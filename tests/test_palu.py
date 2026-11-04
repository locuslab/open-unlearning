"""CPU regression checks. Run with: python tests/test_palu.py"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutput

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import DATASET_REGISTRY  # noqa: E402
from data.collators import DataCollatorForSupervisedDataset  # noqa: E402
from data.palu import PALUDataset  # noqa: E402
from trainer import TRAINER_REGISTRY  # noqa: E402
from trainer.unlearn.palu import PALU  # noqa: E402


class TinyLM(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.randn(13, 13, generator=torch.Generator().manual_seed(seed)).double()
        )
        self.calls = []

    def forward(self, input_ids, attention_mask, labels=None):
        self.calls.append((labels, torch.is_grad_enabled()))
        logits = self.table[input_ids]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:])
        return CausalLMOutput(loss=loss, logits=logits)


def make_trainer(**overrides):
    trainer = PALU.__new__(PALU)
    trainer.model, trainer.ref_model = TinyLM(1), TinyLM(2)
    trainer.processing_class = SimpleNamespace(eos_token_id=12)
    trainer.top_k, trainer.first_n = 3, 2
    trainer.gamma, trainer.alpha = 0.7, 0.3
    trainer.non_sensitive_constraint_type = "kl"
    trainer.kl_direction = "reverse"
    trainer.retain_loss_type = "NLL"
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


def make_inputs(labels=None):
    ids = torch.arange(16).reshape(2, 8) % 13
    if labels is None:
        labels = [[-100, 1, 2, 3, -200, 4, 5, 12], [-100, -100, 6, 7, -200, 8, 9, -100]]
    return {
        "forget": {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels),
        },
        "retain": {
            "input_ids": ids.flip(0),
            "attention_mask": torch.ones_like(ids),
            "labels": ids.flip(0),
        },
    }


def expected_loss(trainer, inputs):
    """Independent token-by-token objective, with categorical KL as the oracle."""
    logits = trainer.model.table[inputs["forget"]["input_ids"]][:, :-1]
    reference = trainer.ref_model.table.detach()[inputs["forget"]["input_ids"]][:, :-1]
    sensitive_losses, common_losses = [], []
    for row, labels in enumerate(inputs["forget"]["labels"][:, 1:].tolist()):
        prefix_length = 0
        for column, label in enumerate(labels):
            current, ref = logits[row, column], reference[row, column]
            if label >= 0 and label != 12:
                prefix_length += 1
                if trainer.first_n <= 0 or prefix_length <= trainer.first_n:
                    indices = ref.topk(min(trainer.top_k, ref.numel())).indices
                    sensitive_losses.append(
                        (current[indices] - current.mean()).square().mean()
                    )
            else:
                prefix_length = 0
            if label == -200:
                if trainer.non_sensitive_constraint_type == "nll":
                    common_losses.append(-current.log_softmax(-1)[ref.argmax()])
                else:
                    p = torch.distributions.Categorical(logits=current)
                    q = torch.distributions.Categorical(logits=ref)
                    forward = torch.distributions.kl_divergence(q, p)
                    reverse = torch.distributions.kl_divergence(p, q)
                    common_losses.append(
                        {
                            "forward": forward,
                            "reverse": reverse,
                            "mixed": (forward + reverse) / 2,
                        }[trainer.kl_direction]
                    )
    zero = logits[..., :0].sum()
    sensitive = torch.stack(sensitive_losses).mean() if sensitive_losses else zero
    common = torch.stack(common_losses).mean() if common_losses else zero
    retain = trainer.model(**inputs["retain"]).loss
    return trainer.gamma * sensitive + trainer.alpha * (common + retain)


class PALUTrainerTests(unittest.TestCase):
    def test_loss_and_gradient_match_token_objective(self):
        for direction in ("forward", "reverse", "mixed", "nll"):
            for first_n in (-1, 1, 2):
                for top_k in (1, 4, 100):
                    with self.subTest(
                        direction=direction, first_n=first_n, top_k=top_k
                    ):
                        trainer = make_trainer(first_n=first_n, top_k=top_k)
                        if direction == "nll":
                            trainer.non_sensitive_constraint_type = "nll"
                        else:
                            trainer.kl_direction = direction
                        inputs = make_inputs()
                        before = inputs["forget"]["labels"].clone()
                        expected = expected_loss(trainer, inputs)
                        expected_grad = torch.autograd.grad(
                            expected, trainer.model.table
                        )[0]
                        loss, outputs = trainer.compute_loss(
                            trainer.model,
                            inputs,
                            return_outputs=True,
                            num_items_in_batch=17,
                        )
                        torch.testing.assert_close(loss, expected)
                        loss.backward()
                        torch.testing.assert_close(
                            trainer.model.table.grad, expected_grad
                        )
                        self.assertIsNone(trainer.ref_model.table.grad)
                        self.assertTrue(
                            all(not grad for _, grad in trainer.ref_model.calls)
                        )
                        self.assertIsNone(trainer.model.calls[-2][0])
                        torch.testing.assert_close(inputs["forget"]["labels"], before)
                        torch.testing.assert_close(
                            outputs.logits,
                            trainer.model.table[inputs["forget"]["input_ids"]],
                        )

    def test_mixed_is_equal_average(self):
        losses, gradients = [], []
        for direction in ("forward", "reverse", "mixed"):
            trainer = make_trainer(kl_direction=direction)
            loss = trainer.compute_loss(trainer.model, make_inputs())
            losses.append(loss)
            gradients.append(torch.autograd.grad(loss, trainer.model.table)[0])
        self.assertNotAlmostEqual(losses[0].item(), losses[1].item())
        torch.testing.assert_close(losses[2], 0.5 * (losses[0] + losses[1]))
        torch.testing.assert_close(gradients[2], 0.5 * (gradients[0] + gradients[1]))

    def test_empty_masks_and_single_token_sequences(self):
        for label in (-100, -200, 12, 1):
            for length in (1, 8):
                for constraint in ("kl", "nll"):
                    with self.subTest(
                        label=label, length=length, constraint=constraint
                    ):
                        trainer = make_trainer(non_sensitive_constraint_type=constraint)
                        inputs = make_inputs([[label] * 8, [label] * 8])
                        inputs["forget"] = {
                            key: value[:, :length]
                            for key, value in inputs["forget"].items()
                        }
                        loss = trainer.compute_loss(trainer.model, inputs)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(torch.isfinite(trainer.model.table.grad).all())

    def test_prefixes_reset_at_span_and_row_boundaries(self):
        trainer = make_trainer(first_n=1)
        labels = torch.tensor([[1, 2, -200, 3, 4, 12], [5, 6, -100, 7, 8, -200]])
        expected = torch.tensor([[True, False, False, True, False, False]] * 2)
        torch.testing.assert_close(trainer._sensitive_mask(labels), expected)

    def test_configuration_validation_and_registration(self):
        self.assertIs(TRAINER_REGISTRY["PALU"], PALU)
        for kwargs in (
            {"top_k": 0},
            {"top_k": 1.5},
            {"first_n": 1.5},
            {"target_mode": "max"},
            {"non_sensitive_constraint_type": "bad"},
            {"kl_direction": "bad"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PALU(**kwargs)


def make_tokenizer():
    tokens = [
        "<unk>",
        "<pad>",
        "<eos>",
        "Question",
        ":",
        "Who",
        "?",
        "Answer",
        "Alice",
        "and",
        "Bob",
        ".",
        "Q",
    ]
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    return tokenizer


class PALUDataTests(unittest.TestCase):
    def make_dataset(self, answer="Alice and Bob", spans=None, **kwargs):
        if spans is None:
            spans = [{"start": 0, "end": 5}, {"start": 10, "end": 13}]
        data = Dataset.from_dict(
            {
                "question": ["Who?"],
                "answer": [answer],
                "target_words": [json.dumps(spans)],
            }
        )
        template = {
            "apply_chat_template": False,
            "user_start_tag": "Question: ",
            "user_end_tag": "\n",
            "asst_start_tag": "Answer: ",
        }
        template.update(kwargs.pop("template_args", {}))
        with patch("data.palu.load_from_disk", return_value=data):
            return PALUDataset("unused", template, make_tokenizer(), **kwargs)

    def test_sensitive_common_prompt_and_final_token_labels(self):
        for use_chat in (False, True):
            with self.subTest(chat=use_chat):
                dataset = self.make_dataset(
                    template_args={"apply_chat_template": use_chat}
                )
                item = dataset[0]
                sensitive = item["input_ids"][item["labels"] >= 0].tolist()
                self.assertEqual(
                    dataset.tokenizer.convert_ids_to_tokens(sensitive), ["Alice", "Bob"]
                )
                self.assertTrue((item["labels"] == -100).any())
                self.assertEqual(item["labels"][-2:].tolist(), [10, -200])
                self.assertEqual(
                    item["labels"][item["input_ids"] == 9].tolist(), [-200]
                )

    def test_truncation_empty_spans_and_collation(self):
        dataset = self.make_dataset(max_length=5)
        truncated = dataset[0]
        self.assertEqual(len(truncated["input_ids"]), 5)
        self.assertTrue((truncated["labels"] == -100).all())
        full = self.make_dataset(spans=[])[0]
        self.assertFalse((full["labels"] >= 0).any())
        batch = DataCollatorForSupervisedDataset(dataset.tokenizer)([truncated, full])
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertTrue((batch["labels"][0, 5:] == -100).all())

    def test_token_crossing_prompt_answer_boundary(self):
        dataset = self.make_dataset(
            answer="Alice",
            spans=[{"start": 0, "end": 5}],
            template_args={"asst_start_tag": "Q"},
        )
        item = dataset[0]
        # WordLevel merges Q + Alice; character offsets must still mark the answer.
        self.assertEqual(item["labels"][-2].item(), dataset.tokenizer.unk_token_id)

    def test_bundled_data_and_registry(self):
        self.assertIs(DATASET_REGISTRY["PALUDataset"], PALUDataset)
        for split, expected in (("01", 40), ("05", 200), ("10", 400)):
            path = (
                ROOT
                / "community/methods/PALU/data"
                / f"forget{split}_with_common_words_gpt"
            )
            dataset = PALUDataset(
                str(path), {"apply_chat_template": True}, make_tokenizer()
            )
            self.assertEqual(len(dataset), expected)
            for item in dataset:
                self.assertLessEqual(len(item["input_ids"]), 512)
                self.assertTrue(
                    torch.equal(
                        item["labels"][item["labels"] >= 0],
                        item["input_ids"][item["labels"] >= 0],
                    )
                )

    def test_invalid_span_and_slow_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "Invalid sensitive span"):
            self.make_dataset(spans=[{"start": 0, "end": 100}])[0]
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            PALUDataset("unused", {}, SimpleNamespace(is_fast=False))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
