"""Prefix-aware localized unlearning with configurable non-sensitive KL."""

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


class PALU(GradDiff):
    """Flatten reference top-k logits at the start of each sensitive span.

    Args:
        top_k: Number of reference vocabulary candidates to flatten.
        first_n: Tokens to select per contiguous sensitive span; <= 0 selects all.
        target_mode: Logit target; only the current vocabulary mean is supported.
        non_sensitive_constraint_type: Preservation objective, ``kl`` or ``nll``.
        kl_direction: ``forward`` is KL(reference || current), ``reverse`` is
            KL(current || reference), and ``mixed`` averages both equally.
            This controls non-sensitive forget tokens, not the retain-set loss.
    """

    def __init__(
        self,
        top_k=1,
        first_n=-1,
        target_mode="mean",
        non_sensitive_constraint_type="kl",
        kl_direction="reverse",
        *args,
        **kwargs,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(first_n, int):
            raise ValueError("first_n must be an integer")
        if target_mode != "mean":
            raise ValueError("target_mode must be 'mean'")
        if non_sensitive_constraint_type not in ("kl", "nll"):
            raise ValueError("non_sensitive_constraint_type must be 'kl' or 'nll'")
        if kl_direction not in ("forward", "reverse", "mixed"):
            raise ValueError("kl_direction must be 'forward', 'reverse', or 'mixed'")
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.first_n = first_n
        self.non_sensitive_constraint_type = non_sensitive_constraint_type
        self.kl_direction = kl_direction
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def _sensitive_mask(self, labels):
        """Select prefixes independently in each row and each sensitive span."""
        mask = labels >= 0
        eos = getattr(self.processing_class, "eos_token_id", None)
        if eos is not None:
            mask &= labels != eos
        if self.first_n > 0 and labels.size(-1) > 0:
            # Track the start of each span so separated entities get their own prefix.
            starts = mask & ~F.pad(mask[:, :-1], (1, 0), value=False)
            positions = torch.arange(labels.size(-1), device=labels.device)
            span_starts = torch.where(starts, positions, 0).cummax(dim=-1).values
            mask &= positions - span_starts < self.first_n
        return mask

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Combine localized flattening, non-sensitive preservation, and retention."""
        forget_inputs = {
            key: inputs["forget"][key] for key in ("input_ids", "attention_mask")
        }
        # PALU's -200 labels are masks, so never pass them to the model's CE loss.
        outputs = model(**forget_inputs)
        logits = outputs.logits[..., :-1, :]
        with torch.no_grad():
            ref_logits = self.ref_model(**forget_inputs).logits[..., :-1, :]
        loss_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )

        labels = inputs["forget"]["labels"][..., 1:].to(logits.device)
        sensitive_mask = self._sensitive_mask(labels)
        non_sensitive_mask = labels == -200
        k = min(self.top_k, logits.size(-1))
        sensitive_logits = logits[sensitive_mask].to(loss_dtype)
        topk_indices = ref_logits[sensitive_mask].topk(k, dim=-1).indices
        selected_logits = sensitive_logits.gather(-1, topk_indices)
        # Keep gradients through the current mean, as in the source implementation.
        target_logits = sensitive_logits.mean(dim=-1, keepdim=True)
        per_token_loss = (selected_logits - target_logits).square().mean(dim=-1)
        sensitive_loss = per_token_loss.sum() / sensitive_mask.sum().clamp_min(1)

        current_log_probs = F.log_softmax(
            logits[non_sensitive_mask], dim=-1, dtype=loss_dtype
        )
        ref_non_sensitive = ref_logits[non_sensitive_mask]
        if self.non_sensitive_constraint_type == "nll":
            ref_predictions = ref_non_sensitive.argmax(dim=-1, keepdim=True)
            per_token_loss = -current_log_probs.gather(-1, ref_predictions).squeeze(-1)
        else:
            ref_log_probs = F.log_softmax(ref_non_sensitive, dim=-1, dtype=loss_dtype)
            if self.kl_direction == "forward":
                per_token_loss = (
                    ref_log_probs.exp() * (ref_log_probs - current_log_probs)
                ).sum(dim=-1)
            elif self.kl_direction == "reverse":
                per_token_loss = (
                    current_log_probs.exp() * (current_log_probs - ref_log_probs)
                ).sum(dim=-1)
            else:
                forward_kl = (
                    ref_log_probs.exp() * (ref_log_probs - current_log_probs)
                ).sum(dim=-1)
                reverse_kl = (
                    current_log_probs.exp() * (current_log_probs - ref_log_probs)
                ).sum(dim=-1)
                per_token_loss = 0.5 * (forward_kl + reverse_kl)
        non_sensitive_loss = per_token_loss.sum() / non_sensitive_mask.sum().clamp_min(
            1
        )

        retain_inputs = {
            key: inputs["retain"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * sensitive_loss + self.alpha * (
            non_sensitive_loss + retain_loss
        )
        return (loss, outputs) if return_outputs else loss
