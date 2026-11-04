"""CPU regression checks. Run with: python tests/test_baldro.py"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainer.unlearn.BalDRO import (  # noqa: E402
    GroupNPO,
    GroupSatImp,
    GroupSimNPO,
)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.embedding(input_ids)
        loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:])
        return SimpleNamespace(loss=loss, logits=logits)


class BalDROGroupTests(unittest.TestCase):
    def test_filters_invalid_samples_and_caps_group_size(self):
        labels = torch.tensor(
            [[1, -100, -100], [-100, -100, -100], [-100, 2, 3], [-100, 4, 5]]
        )
        forget = {
            "input_ids": torch.arange(12).reshape(4, 3) % 8,
            "attention_mask": torch.ones_like(labels),
            "labels": labels,
        }
        inputs = {
            "forget": forget,
            "retain": {**forget, "labels": forget["input_ids"].clone()},
        }

        for trainer_class in (GroupNPO, GroupSimNPO, GroupSatImp):
            for ratio in (0.5, 1.0):
                with self.subTest(trainer=trainer_class.__name__, ratio=ratio):
                    model = TinyLM()
                    trainer = trainer_class.__new__(trainer_class)
                    trainer.sampling_ratio = ratio
                    trainer.ref_model = TinyLM()
                    trainer.beta = trainer.beta1 = trainer.beta2 = 1.0
                    trainer.gamma = trainer.alpha = 1.0
                    trainer.delta = 0.0
                    trainer.retain_loss_type = "NLL"

                    selected = trainer._select_groups(model, inputs)
                    self.assertCountEqual(
                        selected["forget"]["input_ids"].tolist(),
                        forget["input_ids"][2:].tolist(),
                    )
                    self.assertTrue(
                        selected["forget"]["labels"][:, 1:].ne(-100).any(-1).all()
                    )
                    self.assertEqual(selected["retain"]["input_ids"].size(0), 2)

                    loss = trainer.compute_loss(model, inputs)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
                    self.assertIsNone(trainer.ref_model.embedding.weight.grad)

    def test_rejects_batches_without_shifted_targets(self):
        labels = torch.tensor([[1, -100, -100], [-100, -100, -100]])
        inputs = {
            "forget": {
                "input_ids": torch.ones_like(labels),
                "attention_mask": torch.ones_like(labels),
                "labels": labels,
            }
        }
        for trainer_class in (GroupNPO, GroupSimNPO, GroupSatImp):
            with self.subTest(trainer=trainer_class.__name__):
                model = Mock(side_effect=AssertionError("Unexpected model call"))
                trainer = trainer_class.__new__(trainer_class)
                trainer.sampling_ratio = 0.5

                with self.assertRaisesRegex(ValueError, "valid target tokens"):
                    trainer.compute_loss(model, inputs)
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
