"""CPU regression checks. Run with: python tests/test_baldro.py"""

import sys
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainer.unlearn.BalDRO import (  # noqa: E402
    DrNPO,
    DrSatImp,
    DrSimNPO,
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


class BalDROTests(unittest.TestCase):
    def test_filters_invalid_samples_and_caps_group_size(self):
        labels = torch.tensor(
            [[1, -100, -100], [-100, -100, -100], [-100, 2, 3], [-100, 4, 5]]
        )
        forget = {
            "input_ids": torch.arange(12).reshape(4, 3) % 8,
            "attention_mask": torch.ones_like(labels),
            "labels": labels,
        }
        retain_labels = labels.clone()
        retain_labels[2, 1:] = -100
        inputs = {"forget": forget, "retain": {**forget, "labels": retain_labels}}

        for trainer_class in (GroupNPO, GroupSimNPO, GroupSatImp):
            for ratio, retain_loss_type in product((0.5, 1.0), ("NLL", "KL")):
                with self.subTest(
                    trainer=trainer_class.__name__, ratio=ratio, retain=retain_loss_type
                ):
                    model = TinyLM()
                    trainer = trainer_class.__new__(trainer_class)
                    trainer.model = model
                    trainer.sampling_ratio = ratio
                    trainer.ref_model = TinyLM()
                    trainer.beta = trainer.beta1 = trainer.beta2 = 1.0
                    trainer.gamma = trainer.alpha = 1.0
                    trainer.delta = 0.0
                    trainer.retain_loss_type = retain_loss_type

                    selected = trainer._select_groups(model, inputs)
                    self.assertCountEqual(
                        selected["forget"]["input_ids"].tolist(),
                        forget["input_ids"][2:].tolist(),
                    )
                    self.assertTrue(
                        selected["forget"]["labels"][:, 1:].ne(-100).any(-1).all()
                    )
                    expected_size = 1 if retain_loss_type == "NLL" else 2
                    self.assertEqual(
                        selected["retain"]["input_ids"].size(0), expected_size
                    )
                    if retain_loss_type == "NLL":
                        torch.testing.assert_close(
                            selected["retain"]["input_ids"], forget["input_ids"][-1:]
                        )

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
        for trainer_class in (
            GroupNPO,
            GroupSimNPO,
            GroupSatImp,
            DrNPO,
            DrSimNPO,
            DrSatImp,
        ):
            with self.subTest(trainer=trainer_class.__name__):
                model = Mock(side_effect=AssertionError("Unexpected model call"))
                trainer = trainer_class.__new__(trainer_class)
                trainer.sampling_ratio = 0.5

                with self.assertRaisesRegex(ValueError, "valid target tokens"):
                    trainer.compute_loss(model, inputs)
                model.assert_not_called()

    def test_rejects_empty_nll_retain_candidates_but_allows_kl(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        forget = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }
        inputs = {
            "forget": forget,
            "retain": {
                **forget,
                "labels": torch.tensor([[1, -100, -100], [-100, -100, -100]]),
            },
        }
        for trainer_class in (GroupNPO, GroupSimNPO, GroupSatImp):
            with self.subTest(trainer=trainer_class.__name__):
                trainer = trainer_class.__new__(trainer_class)
                trainer.sampling_ratio = 0.5
                trainer.retain_loss_type = "NLL"
                with self.assertRaisesRegex(ValueError, "retain sample"):
                    trainer._select_groups(TinyLM(), inputs)

                trainer.retain_loss_type = "KL"
                selected = trainer._select_groups(TinyLM(), inputs)
                self.assertEqual(selected["retain"]["input_ids"].size(0), 1)

    def test_dv_matches_valid_only_forget_loss_and_gradients(self):
        labels = torch.tensor(
            [[1, -100, -100], [-100, -100, -100], [-100, 2, 3], [-100, 4, 5]]
        )
        forget = {
            "input_ids": torch.arange(12).reshape(4, 3) % 8,
            "attention_mask": torch.ones_like(labels),
            "labels": labels,
        }
        valid_forget = {key: value[2:] for key, value in forget.items()}
        for trainer_class in (DrNPO, DrSimNPO, DrSatImp):
            for forget_dro in (True, False):
                with self.subTest(trainer=trainer_class.__name__, dro=forget_dro):
                    model = TinyLM()
                    trainer = trainer_class.__new__(trainer_class)
                    trainer.ref_model = TinyLM()
                    trainer.beta = trainer.beta1 = trainer.beta2 = 1.0
                    trainer.gamma = trainer.alpha = 1.0
                    trainer.delta = 0.0
                    trainer.forget_dro = forget_dro
                    trainer.beta_dv_forget = 0.5
                    trainer.retain_dro = False
                    trainer.retain_loss_type = "NLL"

                    loss, outputs = trainer.compute_loss(
                        model,
                        {"forget": forget, "retain": valid_forget},
                        return_outputs=True,
                    )
                    loss.backward()
                    grad = model.embedding.weight.grad.clone()
                    model.zero_grad()
                    expected_loss = trainer.compute_loss(
                        model, {"forget": valid_forget, "retain": valid_forget}
                    )
                    expected_loss.backward()
                    torch.testing.assert_close(loss, expected_loss)
                    torch.testing.assert_close(grad, model.embedding.weight.grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertIsNone(trainer.ref_model.embedding.weight.grad)
                    if trainer_class is DrNPO:
                        outputs = outputs[1]
                    self.assertEqual(outputs.logits.size(0), 2)
                    self.assertEqual(forget["labels"].size(0), 4)


if __name__ == "__main__":
    unittest.main()
