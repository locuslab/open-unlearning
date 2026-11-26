"""RepSelect: a single-shot unlearning update of the MLP weights. The forget gradient
is taken as the update, after the components it shares with the retain gradient have
been scaled down ("collapsed") on both sides of each weight matrix.

Per MLP weight W (D_out, D_in), with G_r and G_f its gradients of minus the loss,
averaged over all tokens of the retain and of the forget set:
    U, S, V = top n_pcs singular triplets of G_r
    G_f's components along the columns of V (input side) and along the columns of U
    (output side) are divided by S^2, normalized so that the smallest of the n_pcs
    maps to 1; the rest of G_f is left as is
    W -= strength * collapsed G_f
"""

import logging

import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import TrainOutput

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)

DENSE_MODULES = ["gate_proj", "up_proj", "down_proj"]
# transformers' MoE layers keep the weights of all experts of a layer stacked in one
# parameter (num_experts, ., .); the collapse is then done per expert
MOE_PARAMETERS = ["gate_up_proj", "down_proj"]


def _is_moe(model) -> bool:
    return any(hasattr(layer.mlp, "experts") for layer in model.model.layers)


def _mlp_weights(model) -> list[torch.nn.Parameter]:
    """The weights to update. In an MoE model these are the expert weights only: its
    dense layers, if it has any, are left alone."""
    layers = model.model.layers
    if _is_moe(model):
        return [
            getattr(layer.mlp.experts, name)
            for layer in layers
            if hasattr(layer.mlp, "experts")
            for name in MOE_PARAMETERS
        ]
    return [
        getattr(layer.mlp, name).weight for layer in layers for name in DENSE_MODULES
    ]


def _collapse(mat: torch.Tensor, vecs: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Divide mat's components along the columns of vecs by S^2, normalized so that
    the smallest one maps to 1; the rest of mat is left as is.
    mat (..., n, d), vecs (..., d, n_pcs), S (..., n_pcs): the leading dimension, if
    any, is the experts of an MoE layer."""
    S = S**2
    if (S.amin(dim=-1) == 0).any():
        raise ValueError(
            "The retain gradient of an MLP weight (or of an MoE expert) has fewer than "
            "n_pcs nonzero singular values: the retain set is probably too small."
        )
    S = S / S.amin(dim=-1, keepdim=True)
    projected = mat @ vecs
    return mat - (projected - projected / S.unsqueeze(-2)) @ vecs.mT


class RepSelect(UnlearnTrainer):
    """
    1. (optional, lora_learning_rate) Elicitation: a LoRA is trained for one pass to
       lower the forget loss, and stays active in the forward of steps 2 and 3.
    2. One pass over the retain set: accumulate the weight gradients, take their SVD.
    3. One pass over the forget set: accumulate the weight gradients.
    4. Collapse the forget gradients along the top retain singular vectors, on the
       input side (V) and on the output side (U).
    5. weight -= strength * collapsed forget gradient, once.

    The only TrainingArguments used is per_device_train_batch_size. learning_rate,
    scheduler, optimizer, weight decay and number of epochs have no effect.

    Args:
        strength: the size of the single update step.
        n_pcs: number of retain singular vectors to collapse along, per side.
        lora_learning_rate: SGD learning rate of the LoRA elicitation; None (default)
            skips the elicitation. Needs the `peft` package.
    """

    def __init__(
        self, strength=2.5, n_pcs=512, lora_learning_rate=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes > 1:
            raise NotImplementedError(
                "RepSelect currently does not support distributed training: run it in "
                "a single process (without `accelerate launch --num_processes`)."
            )
        self.strength = strength
        self.n_pcs = n_pcs
        self.lora_learning_rate = lora_learning_rate

    def _batches(self, split):
        """Batches of the whole `split` (forget | retain) set.
        The paired loader of the other methods is not used because it samples only as
        many retain examples as there are forget ones."""
        loader = DataLoader(
            getattr(self.train_dataset, split),
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )
        return self.accelerator.prepare(loader)  # moves the batches to the device

    def _accumulate_grad(self, split):
        """One pass over `split` (forget | retain); leaves the gradient of minus the
        loss, averaged over all tokens of the pass, in the .grad of the MLP weights."""
        self.model.zero_grad(set_to_none=True)
        n_tokens = 0
        for batch in self._batches(split):
            batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
            # .loss is the mean over the batch's predicted tokens: undo that mean
            n = (batch["labels"][:, 1:] != -100).sum().item()
            (-self.model(**batch).loss * n).backward()
            n_tokens += n
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad /= n_tokens

    def _elicit(self):
        """Wrap the MLP weights in a LoRA (in place) and train it with SGD for one
        pass over the forget set, to lower the forget loss. Returns the peft model."""
        from peft import LoraConfig, get_peft_model

        if _is_moe(self.model):
            names = [f"mlp.experts.{name}" for name in MOE_PARAMETERS]
            config = LoraConfig(target_parameters=names)
        else:
            config = LoraConfig(target_modules=DENSE_MODULES)
        peft_model = get_peft_model(self.model, config)  # freezes all but the LoRA
        lora_params = [p for p in peft_model.parameters() if p.requires_grad]
        for batch in self._batches("forget"):
            batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
            peft_model.zero_grad(set_to_none=True)
            peft_model(**batch).loss.backward()
            for param in lora_params:
                param.data -= self.lora_learning_rate * param.grad
        return peft_model

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.model = self.accelerator.prepare(self.model)
        self.model.train()
        weights = _mlp_weights(self.model)

        peft_model = None
        if self.lora_learning_rate is not None:
            peft_model = self._elicit()

        self.model.requires_grad_(False)
        for weight in weights:
            weight.requires_grad_(True)

        self._accumulate_grad("retain")
        USVs = [torch.svd_lowrank(w.grad.float(), q=self.n_pcs) for w in weights]
        self._accumulate_grad("forget")

        if peft_model is not None:
            peft_model.unload()  # removes the LoRA layers from self.model, in place

        for weight, (U, S, V) in zip(weights, USVs):
            grad = weight.grad.float()
            grad = _collapse(grad, V, S)  # input side
            grad = _collapse(grad.mT, U, S).mT  # output side
            weight.data -= (self.strength * grad).to(weight.dtype)
            weight.grad = None

        self.model.requires_grad_(True)
        self.state.global_step = 1
        return TrainOutput(self.state.global_step, 0.0, {})
