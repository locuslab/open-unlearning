"""Unlearning Depth Score (UDS), https://arxiv.org/abs/2605.24614

UDS measures how much forget-set knowledge an unlearned model still encodes in its
hidden states via two-stage activation patching into the full (pre-unlearning) model:
  Stage 1: patch the retain model's layer-l hidden states into the full model and
      record the drop in the entity log-prob, d1_l. Layers with d1_l > tau are the
      knowledge-encoding (KE) layers.
  Stage 2: repeat with the evaluated model as the source, giving d2_l.
  UDS = sum_{l in KE} d1_l * clip(d2_l / d1_l, 0, 1) / sum_{l in KE} d1_l
UDS is 0 when the knowledge is intact (as in the full model) and 1 when it is erased
as deeply as in the retain model.
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from data.utils import IGNORE_INDEX
from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals

logger = logging.getLogger("metrics")


def _load_reference_model(model_path, model):
    """Load the full or retain model like the evaluated model, which it must match."""
    logger.info(f"Loading {model_path}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=model.dtype,
        attn_implementation=model.config._attn_implementation,
        device_map={"": model.device},
    )
    keys = ("hidden_size", "num_hidden_layers")
    if any(
        getattr(reference_model.config, k) != getattr(model.config, k) for k in keys
    ):
        raise ValueError(
            f"{model_path} does not match the evaluated model, "
            "set full_model_path and retain_model_path in the uds metric config"
        )
    return reference_model.eval()


def _entity_logprob(logits, labels):
    """Mean log-prob of the labelled (entity) tokens of each sequence."""
    shifted_labels = labels[:, 1:].to(logits.device)
    mask = shifted_labels != IGNORE_INDEX
    log_probs = torch.log_softmax(logits[:, :-1][mask].float(), dim=-1)
    token_log_probs = log_probs.gather(-1, shifted_labels[mask].unsqueeze(-1))
    rows = mask.nonzero(as_tuple=True)[0]
    sums = torch.zeros(len(labels), device=logits.device)
    sums.index_add_(0, rows, token_log_probs.squeeze(-1))
    return sums / mask.sum(-1)


def _patched_entity_logprob(model, layer, source_hidden, patch_mask, batch):
    """Entity log-prob of `model` with the output of `layer` replaced by
    `source_hidden` at the positions in `patch_mask`."""

    def patch_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = torch.where(
            patch_mask.to(hidden.device).unsqueeze(-1),
            source_hidden.to(hidden.device, hidden.dtype),
            hidden,
        )
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = layer.register_forward_hook(patch_hook)
    try:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    finally:
        handle.remove()
    return _entity_logprob(logits, batch["labels"])


@unlearning_metric(name="uds")
def uds(model, **kwargs):
    """Unlearning Depth Score, see the module docstring."""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    delta_threshold = kwargs["delta_threshold"]
    full_model = _load_reference_model(kwargs["full_model_path"], model)
    retain_model = _load_reference_model(kwargs["retain_model_path"], model)
    layers = full_model.get_decoder().layers

    def _uds(model, batch):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        # patch the positions that predict the entity tokens
        patch_mask = torch.zeros_like(batch["labels"], dtype=torch.bool)
        patch_mask[:, :-1] = batch["labels"][:, 1:] != IGNORE_INDEX

        with torch.no_grad():
            full_logprob = _entity_logprob(full_model(**inputs).logits, batch["labels"])
            # hidden_states[i + 1] is the output of decoder layer i; as in the reference
            # implementation, the last one is taken after the final norm
            retain_hidden = retain_model(
                **inputs, output_hidden_states=True
            ).hidden_states
            unlearn_hidden = model(**inputs, output_hidden_states=True).hidden_states
            s1_deltas, s2_deltas = [], []
            for layer_idx, layer in enumerate(layers):
                for source_hidden, deltas in (
                    (retain_hidden, s1_deltas),
                    (unlearn_hidden, s2_deltas),
                ):
                    patched_logprob = _patched_entity_logprob(
                        full_model,
                        layer,
                        source_hidden[layer_idx + 1],
                        patch_mask,
                        batch,
                    )
                    deltas.append(full_logprob - patched_logprob)
        s1_deltas = torch.stack(s1_deltas, dim=-1)  # batch_size x num_layers
        s2_deltas = torch.stack(s2_deltas, dim=-1)

        # knowledge-encoding (KE) layers weighted by their stage 1 drop
        ke_mask = s1_deltas > delta_threshold
        weights = torch.where(ke_mask, s1_deltas, torch.zeros_like(s1_deltas))
        erasure_ratio = torch.where(
            ke_mask, (s2_deltas / s1_deltas).clamp(0, 1), torch.zeros_like(s1_deltas)
        )
        scores = (weights * erasure_ratio).sum(-1) / weights.sum(-1)
        return [
            {"score": score.item() if ke.any() else None}
            for score, ke in zip(scores, ke_mask)
        ]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = run_batchwise_evals(
        model, dataloader, _uds, {}, "Calculating UDS"
    )
    uds_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    if len(uds_values) < len(scores_by_index):
        logger.info(
            f"UDS skipped {len(scores_by_index) - len(uds_values)} examples "
            f"without knowledge-encoding layers (delta_threshold={delta_threshold})"
        )
    return {"agg_value": np.mean(uds_values), "value_by_index": scores_by_index}
