"""CPU checks for RepSelect on tiny random models. Run with: python tests/test_repselect.py"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainer.unlearn.repselect import RepSelect, _collapse, _mlp_weights  # noqa: E402

STRENGTH, N_PCS = 0.1, 8


def make_batches(n=4):
    def batch():
        ids = torch.randint(1, 100, (3, 10))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    return [{"forget": batch(), "retain": batch()} for _ in range(n)]


def tiny_llama():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(config)


def unlearn(model, batches, **method_args):
    """Runs RepSelect on a copy of the model; returns the copy."""

    class Trainer(RepSelect):
        def _batches(self, split):
            return [pair[split] for pair in batches]

    model = copy.deepcopy(model)
    with tempfile.TemporaryDirectory() as out_dir:
        args = TrainingArguments(output_dir=out_dir, report_to="none", use_cpu=True)
        trainer = Trainer(
            model=model, args=args, strength=STRENGTH, n_pcs=N_PCS, **method_args
        )
        torch.manual_seed(1)  # the draws of svd_lowrank
        trainer.train()
        assert trainer.model is model
    return model


def mean_grads(model, batches, split):
    """Gradient of minus the loss averaged over all tokens: one big batch."""
    model = copy.deepcopy(model)
    keys = batches[0][split].keys()
    batch = {k: torch.cat([pair[split][k] for pair in batches]) for k in keys}
    (-model(**batch).loss).backward()
    return [weight.grad for weight in _mlp_weights(model)]


def rebatch(batches, sizes):
    """The same examples in the same order, split into batches of the given sizes."""
    out, start = [], 0
    for size in sizes:
        pair = {}
        for split in ("forget", "retain"):
            keys = batches[0][split].keys()
            full = {k: torch.cat([b[split][k] for b in batches]) for k in keys}
            pair[split] = {k: v[start : start + size] for k, v in full.items()}
        out.append(pair)
        start += size
    return out


class RepSelectTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_collapse_scales_only_the_given_directions(self):
        vecs, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))
        kept, rest = vecs[:, :3], vecs[:, 3:]
        S = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
        mat = torch.randn(5, 16, dtype=torch.float64)
        out = _collapse(mat, kept, S)
        # divided by S^2, normalized so that the smallest maps to 1
        self.assertTrue(
            torch.allclose(out @ kept, (mat @ kept) / torch.tensor([16.0, 4.0, 1.0]))
        )
        self.assertTrue(torch.allclose(out @ rest, mat @ rest))

    def test_collapse_is_per_expert_on_stacked_weights(self):
        mat = torch.randn(3, 5, 16, dtype=torch.float64)
        vecs, _ = torch.linalg.qr(torch.randn(3, 16, 4, dtype=torch.float64))
        S = torch.rand(3, 4, dtype=torch.float64) + 0.5
        out = _collapse(mat, vecs, S)
        for e in range(3):
            self.assertTrue(torch.allclose(out[e], _collapse(mat[e], vecs[e], S[e])))

    def test_collapse_raises_on_a_zero_retain_gradient(self):
        retain = torch.randn(3, 20, 16)
        retain[1] = 0  # an expert that saw no retain token
        U, S, V = torch.svd_lowrank(retain, q=4)
        with self.assertRaisesRegex(ValueError, "retain set is probably too small"):
            _collapse(torch.randn(3, 5, 16), V, S)

    def test_update_is_the_collapsed_forget_gradient(self):
        model, batches = tiny_llama(), make_batches()
        unlearned = unlearn(model, batches)

        retain = mean_grads(model, batches, "retain")
        forget = mean_grads(model, batches, "forget")
        torch.manual_seed(1)
        USVs = [torch.svd_lowrank(grad, q=N_PCS) for grad in retain]
        before, after = _mlp_weights(model), _mlp_weights(unlearned)
        self.assertEqual(len(before), 6)  # gate, up, down of 2 layers
        for w0, w1, grad, (U, S, V) in zip(before, after, forget, USVs):
            expected = _collapse(_collapse(grad, V, S).mT, U, S).mT
            self.assertTrue(torch.allclose(w0 - w1, STRENGTH * expected, atol=1e-6))
            self.assertGreater(float(expected.norm()), 0)

        # nothing but the MLP weights is touched
        mlp = {id(weight) for weight in after}
        for p0, p1 in zip(model.parameters(), unlearned.parameters()):
            if id(p1) not in mlp:
                self.assertTrue(torch.equal(p0, p1))
            self.assertTrue(p1.requires_grad)

    def test_update_does_not_depend_on_the_batch_size(self):
        model, batches = tiny_llama(), make_batches()  # 4 batches of 3 examples
        for pair in batches:  # examples of different lengths: mask some labels
            for batch in pair.values():
                labels = batch["labels"].clone()
                for row in labels:
                    row[torch.randint(3, 10, ()) :] = -100
                batch["labels"] = labels
        reference = _mlp_weights(unlearn(model, batches))
        for sizes in ([12], [1] * 12, [5, 1, 4, 2]):
            other = _mlp_weights(unlearn(model, rebatch(batches, sizes)))
            for w0, w_ref, w in zip(_mlp_weights(model), reference, other):
                update = w0 - w_ref
                self.assertGreater(float(update.norm()), 0)
                self.assertTrue(torch.allclose(update, w0 - w, rtol=1e-3, atol=1e-7))

    def test_lora_elicitation_leaves_no_adapter(self):
        try:
            import peft  # noqa: F401
        except ImportError:
            self.skipTest("peft is not installed")
        model, batches = tiny_llama(), make_batches()
        plain = unlearn(model, batches)
        elicited = unlearn(model, batches, lora_learning_rate=0.5)
        self.assertEqual(
            [name for name, _ in elicited.named_parameters()],
            [name for name, _ in model.named_parameters()],
        )
        changed = [
            not torch.allclose(a, b, atol=1e-7)
            for a, b in zip(_mlp_weights(plain), _mlp_weights(elicited))
        ]
        self.assertTrue(all(changed))  # the LoRA did affect the gradients

    def test_moe(self):
        try:
            from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
        except ImportError:
            self.skipTest("this transformers version has no DeepseekV2")
        config = DeepseekV2Config(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,  # one dense layer (left alone), one MoE layer
        )
        model, batches = DeepseekV2ForCausalLM(config), make_batches()
        shapes = [tuple(weight.shape) for weight in _mlp_weights(model)]
        self.assertEqual(shapes, [(4, 48, 32), (4, 32, 24)])
        dense = model.model.layers[0].mlp.state_dict()
        lora_lrs = [None]
        try:
            import peft  # noqa: F401

            lora_lrs.append(0.5)
        except ImportError:
            pass
        for lora_lr in lora_lrs:
            unlearned = unlearn(model, batches, lora_learning_rate=lora_lr)
            for w0, w1 in zip(_mlp_weights(model), _mlp_weights(unlearned)):
                self.assertTrue(torch.isfinite(w1).all())
                self.assertGreater(float((w0 - w1).detach().norm()), 0)
            for name, value in unlearned.model.layers[0].mlp.state_dict().items():
                self.assertTrue(torch.equal(value, dense[name]))


if __name__ == "__main__":
    unittest.main()
