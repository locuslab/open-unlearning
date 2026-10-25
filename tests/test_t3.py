"""CPU regression checks. Run with: python tests/test_t3.py"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.collators import DataCollatorForSupervisedDataset  # noqa: E402
from model.t3 import T3CausalLM  # noqa: E402
from trainer.unlearn.t3 import T3  # noqa: E402


def make_base(tied=False):
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            eos_token_id=31,
            tie_word_embeddings=tied,
            attn_implementation="sdpa",
        )
    )


def make_batch(ignored=False):
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100] if ignored else [-100, 2, 3]]),
    }


class T3RegressionTests(unittest.TestCase):
    def test_model_initialization_and_generation(self):
        model = T3CausalLM.from_pretrained_base_obj(
            make_base(), guidance_kwargs={"hidden_size": 4}
        )
        self.assertEqual(model.base_lm.config._attn_implementation, "sdpa")
        self.assertFalse(any(p.requires_grad for p in model.base_lm.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.guidance_head.parameters()))
        batch = make_batch()
        outputs = model(**batch)
        self.assertEqual(outputs.logits.shape, (1, 3, 32))
        for use_cache in (True, False):
            tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=2,
                eos_token_id=None,
                top_p=None,
                use_cache=use_cache,
            )
            self.assertEqual(tokens.shape, (1, 5))

    def test_cached_mean_keeps_counting_in_low_precision(self):
        for dtype, prompt_length in ((torch.bfloat16, 256), (torch.float16, 2048)):
            with self.subTest(dtype=dtype):
                model = T3CausalLM.from_pretrained_base_obj(
                    make_base().to(dtype), guidance_kwargs={"hidden_size": 4}
                )

                def base_forward(input_ids, **kwargs):
                    length = input_ids.shape[1]
                    # Zero prompt states followed by unit generated states.
                    states = torch.full(
                        (1, length, 16), float(length == 1), dtype=dtype
                    )
                    return SimpleNamespace(
                        hidden_states=(states,),
                        logits=torch.zeros(1, length, 32, dtype=dtype),
                        past_key_values=None,
                    )

                pooled = []
                hook = model.guidance_head.register_forward_pre_hook(
                    lambda module, inputs: pooled.append(inputs[0].detach().clone())
                )
                with patch.object(model.base_lm, "forward", side_effect=base_forward):
                    model.generate(
                        torch.ones(1, prompt_length, dtype=torch.long),
                        max_new_tokens=101,
                        eos_token_id=None,
                        top_p=None,
                    )
                hook.remove()
                expected = torch.full_like(pooled[-1], 100 / (prompt_length + 100))
                torch.testing.assert_close(pooled[-1], expected)

    def test_training_with_valid_and_ignored_labels(self):
        for ignored in (False, True):
            with self.subTest(ignored=ignored), tempfile.TemporaryDirectory() as tmp:
                batch = make_batch(ignored)
                item = {key: value[0] for key, value in batch.items()}
                trainer = T3(
                    model=make_base(),
                    guidance_cfg=OmegaConf.create({"hidden_size": 4}),
                    args=TrainingArguments(
                        output_dir=tmp,
                        use_cpu=True,
                        bf16=False,
                        report_to=[],
                        optim="adamw_torch",
                        remove_unused_columns=False,
                        max_steps=1,
                        save_strategy="no",
                        disable_tqdm=True,
                    ),
                    train_dataset=[{"retain": item, "forget": item}],
                    data_collator=DataCollatorForSupervisedDataset(
                        SimpleNamespace(pad_token_id=0)
                    ),
                )
                loss = trainer.compute_loss(
                    trainer.model, {"retain": batch, "forget": batch}
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for param in trainer.model.guidance_head.parameters():
                    self.assertIsNotNone(param.grad)
                    if ignored:
                        self.assertEqual(torch.count_nonzero(param.grad).item(), 0)
                trainer.model.zero_grad()
                result = trainer.train()
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))

    def test_reload_without_original_base_model(self):
        for tied, shard_size in ((False, "5GB"), (True, "1KB")):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as tmp:
                original = Path(tmp) / "original"
                saved = Path(tmp) / "saved"
                base = make_base(tied)
                base.save_pretrained(original)
                base.config._name_or_path = str(original)
                base.generation_config.eos_token_id = [30, 31]
                model = T3CausalLM.from_pretrained_base_obj(
                    base, guidance_kwargs={"hidden_size": 4}
                )
                model.save_pretrained(saved, max_shard_size=shard_size)
                shutil.rmtree(original)
                config = AutoConfig.from_pretrained(str(saved))
                self.assertIsInstance(config.to_base(), LlamaConfig)
                self.assertEqual(config.guidance, {"hidden_size": 4})
                loaded = AutoModelForCausalLM.from_pretrained(
                    str(saved),
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                    device_map="cpu",
                    local_files_only=True,
                )
                self.assertIsInstance(loaded, T3CausalLM)
                self.assertEqual(loaded.generation_config.eos_token_id, [30, 31])
                self.assertEqual(loaded.base_lm.config._attn_implementation, "eager")
                for name, value in model.state_dict().items():
                    actual = loaded.state_dict()[name]
                    self.assertEqual(actual.dtype, torch.float16)
                    torch.testing.assert_close(actual, value.to(torch.float16))
                if tied:
                    self.assertIs(
                        loaded.base_lm.lm_head.weight,
                        loaded.base_lm.model.embed_tokens.weight,
                    )


if __name__ == "__main__":
    unittest.main()
