"""BalDRO-G and BalDRO-DV plugins for NPO, SimNPO, and SatImp."""

import math

import torch
import torch.nn.functional as F

from trainer.unlearn.npo import NPO
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll


class _BalDRODV:
    """Apply positive DV aggregation to per-sample unlearning losses."""

    def __init__(
        self,
        beta_dv_forget,
        beta_dv_retain,
        forget_dro=True,
        retain_dro=False,
        *args,
        **kwargs,
    ):
        if forget_dro and beta_dv_forget <= 0:
            raise ValueError(
                "beta_dv_forget must be positive when forget_dro is enabled"
            )
        if retain_dro and beta_dv_retain <= 0:
            raise ValueError(
                "beta_dv_retain must be positive when retain_dro is enabled"
            )
        super().__init__(*args, **kwargs)
        if retain_dro and self.retain_loss_type != "NLL":
            raise ValueError("DRO only supports NLL retain loss currently")
        self.beta_dv_forget = beta_dv_forget
        self.beta_dv_retain = beta_dv_retain
        self.forget_dro = forget_dro
        self.retain_dro = retain_dro

    def _dv_aggregate(self, per_sample_loss, beta_dv):
        return beta_dv * (
            torch.logsumexp(per_sample_loss / beta_dv, dim=0)
            - math.log(per_sample_loss.numel())
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute losses using forget and NLL retain samples with valid shifted targets.

        Raises:
            ValueError: If forget or NLL retain inputs have no valid shifted targets.
        """
        forget_inputs = inputs["forget"]
        valid_samples = forget_inputs["labels"][..., 1:].ne(-100).any(-1)
        if not valid_samples.any():
            raise ValueError(
                "BalDRO-DV requires at least one forget sample with valid target tokens"
            )
        forget_inputs = {
            key: forget_inputs[key][valid_samples]
            for key in ("input_ids", "attention_mask", "labels")
        }

        forget_loss, forget_outputs = self._compute_forget_loss(model, forget_inputs)
        if self.forget_dro:
            forget_loss = self._dv_aggregate(forget_loss, self.beta_dv_forget)
        else:
            forget_loss = forget_loss.mean()

        retain_inputs = {
            key: inputs["retain"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        if self.retain_loss_type == "NLL":
            valid_samples = retain_inputs["labels"][..., 1:].ne(-100).any(-1)
            if not valid_samples.any():
                raise ValueError(
                    "BalDRO-DV requires at least one retain sample with valid target tokens"
                )
            retain_inputs = {
                key: value[valid_samples] for key, value in retain_inputs.items()
            }

        if self.retain_dro:
            retain_loss, _ = compute_batch_nll(model, retain_inputs)
            valid_tokens = retain_inputs["labels"][..., 1:].ne(-100).sum(-1)
            retain_loss = retain_loss / valid_tokens.clamp_min(1)
            retain_loss = self._dv_aggregate(retain_loss, self.beta_dv_retain)
        else:
            retain_loss = self.compute_retain_loss(model, retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class DrNPO(_BalDRODV, NPO):
    """BalDRO-DV with the per-sequence NPO objective."""

    def _compute_forget_loss(self, model, inputs):
        loss, outputs = compute_batch_nll(model, inputs)
        with torch.no_grad():
            ref_loss, _ = compute_batch_nll(self.ref_model, inputs)
        loss = -2 / self.beta * F.logsigmoid(self.beta * (loss - ref_loss))
        return loss, (None, outputs)


class DrSimNPO(_BalDRODV, SimNPO):
    """BalDRO-DV with length-normalized SimNPO losses."""

    def _compute_forget_loss(self, model, inputs):
        loss, outputs = compute_batch_nll(model, inputs)
        valid_tokens = inputs["labels"][..., 1:].ne(-100).sum(-1)
        loss = loss / valid_tokens.clamp_min(1) - self.delta
        loss = -F.logsigmoid(self.beta * loss) * 2 / self.beta
        return loss, outputs


class DrSatImp(_BalDRODV, SatImp):
    """BalDRO-DV with signed, length-normalized SatImp losses."""

    def _compute_forget_loss(self, model, inputs):
        inputs = {key: inputs[key] for key in ("input_ids", "attention_mask", "labels")}
        outputs = model(**inputs)
        labels = inputs["labels"][..., 1:].to(outputs.logits.device)
        logits = outputs.logits[..., :-1, :]
        token_loss = F.cross_entropy(
            logits.transpose(-1, -2), labels, ignore_index=-100, reduction="none"
        )
        probabilities = (-token_loss).exp().detach()
        weights = probabilities**self.beta1 * (1 - probabilities) ** self.beta2
        mask = labels.ne(-100)
        loss = -(weights * token_loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return loss, outputs


class _BalDROGroup:
    """Select hard forget samples and a random retain group within each batch."""

    def __init__(self, sampling_ratio=0.5, *args, **kwargs):
        if not 0 < sampling_ratio <= 1:
            raise ValueError("sampling_ratio must be in (0, 1]")
        super().__init__(*args, **kwargs)
        self.sampling_ratio = sampling_ratio

    def _select_groups(self, model, inputs):
        """Select groups with valid shifted targets for forget and NLL retain losses."""
        forget_inputs = {
            key: inputs["forget"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        valid_tokens = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
        valid_samples = valid_tokens > 0
        valid_count = valid_samples.sum().item()
        if valid_count == 0:
            raise ValueError(
                "BalDRO-G requires at least one forget sample with valid target tokens"
            )

        with torch.no_grad():
            nll, _ = compute_batch_nll(model, forget_inputs)
            hardness = -nll / valid_tokens.clamp_min(1)
            hardness = hardness.masked_fill(~valid_samples, -torch.inf)
            k = max(1, int(hardness.size(0) * self.sampling_ratio))
            k = min(k, valid_count)
            indices = torch.topk(hardness, k, largest=True, sorted=False).indices
        forget_inputs = {
            key: value.index_select(0, indices) for key, value in forget_inputs.items()
        }

        retain_inputs = inputs["retain"]
        if self.retain_loss_type == "NLL":
            valid_samples = retain_inputs["labels"][..., 1:].ne(-100).any(-1)
            if not valid_samples.any():
                raise ValueError(
                    "BalDRO-G requires at least one retain sample with valid target tokens"
                )
            retain_inputs = {
                key: retain_inputs[key][valid_samples]
                for key in ("input_ids", "attention_mask", "labels")
            }
        retain_size = retain_inputs["input_ids"].size(0)
        indices = torch.randperm(retain_size, device=retain_inputs["input_ids"].device)[
            : min(k, retain_size)
        ]
        retain_inputs = {
            key: retain_inputs[key].index_select(0, indices)
            for key in ("input_ids", "attention_mask", "labels")
        }
        return {"forget": forget_inputs, "retain": retain_inputs}

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = self._select_groups(model, inputs)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class GroupNPO(_BalDROGroup, NPO):
    """BalDRO-G applied to NPO."""


class GroupSimNPO(_BalDROGroup, SimNPO):
    """BalDRO-G applied to SimNPO with shifted target-token counts."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = self._select_groups(model, inputs)
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        valid_tokens = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
        forget_loss = forget_loss / valid_tokens.clamp_min(1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
        retain_loss = self.compute_retain_loss(model, inputs["retain"])
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class GroupSatImp(_BalDROGroup, SatImp):
    """BalDRO-G applied to SatImp."""
