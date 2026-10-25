"""CPU regression checks. Run with: python tests/test_t3.py"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.t3 import T3CausalLM  # noqa: E402
from trainer.unlearn.t3 import T3  # noqa: E402


def make_model():
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attn_implementation="sdpa",
        )
    )
    return T3CausalLM.from_pretrained_base_obj(base, guidance_kwargs={"hidden_size": 4})


class T3RegressionTests(unittest.TestCase):
    def test_initialization_preserves_attention(self):
        self.assertEqual(make_model().base_lm.config._attn_implementation, "sdpa")

    def test_cached_mean_past_256_tokens(self):
        model = make_model().to(torch.bfloat16)

        def base_forward(input_ids, **kwargs):
            length = input_ids.shape[1]
            return SimpleNamespace(
                hidden_states=(
                    torch.full(
                        (1, length, 16), float(length == 1), dtype=torch.bfloat16
                    ),
                ),
                logits=torch.zeros(1, length, 32, dtype=torch.bfloat16),
                past_key_values=None,
            )

        with (
            patch.object(model.base_lm, "forward", side_effect=base_forward),
            patch.object(
                model.guidance_head, "forward", wraps=model.guidance_head.forward
            ) as head,
        ):
            model.generate(
                torch.ones(1, 256, dtype=torch.long),
                max_new_tokens=17,
                eos_token_id=None,
                top_p=None,
            )
        pooled = head.call_args.args[0]
        torch.testing.assert_close(pooled, torch.full_like(pooled, 16 / 272))

    def test_ignored_labels_allow_backward(self):
        model = make_model()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.full((1, 3), -100),
        }
        trainer = T3.__new__(T3)
        loss = trainer.compute_loss(model, {"retain": batch, "forget": batch})
        loss.backward()
        self.assertEqual(loss.item(), 0)
        for param in model.guidance_head.parameters():
            torch.testing.assert_close(param.grad, torch.zeros_like(param))

    def test_reload_without_original_model(self):
        model = make_model()
        model.generation_config.eos_token_id = [30, 31]
        with tempfile.TemporaryDirectory() as tmp:
            model.config.base_config_dict["_name_or_path"] = str(Path(tmp) / "missing")
            model.save_pretrained(tmp)
            loaded = AutoModelForCausalLM.from_pretrained(
                tmp,
                torch_dtype=torch.float16,
                attn_implementation="eager",
                local_files_only=True,
            )
        self.assertEqual(loaded.generation_config.eos_token_id, [30, 31])
        self.assertEqual(loaded.base_lm.config._attn_implementation, "eager")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name], value.to(torch.float16)
            )


if __name__ == "__main__":
    unittest.main()
