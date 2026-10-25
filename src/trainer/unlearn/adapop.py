import logging

import torch
from transformers import TrainerCallback

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss_per_example_beta, popularity_to_beta

logger = logging.getLogger(__name__)


class AdaPop(GradDiff):
    """AdaPop: popularity-aware weighted gradient ascent with a dual-ascent retain weight.

    Forget loss: WGA with a per-example beta_i = clip(beta_a * p_i^(-beta_b), 0.05, 2.0),
    where p_i is the popularity of forget example i.
    Loss: gamma_k * L_forget + alpha_k * L_retain, with alpha_k = clip(alpha0 + lambda_k, 0, gamma_k).
    lambda_k is updated at the end of every epoch from the relative increase of the retain loss:
    lambda <- clip(lambda + dual_lr * (delta_retain - eps), 0, lambda_max).
    """

    def __init__(
        self,
        use_popularity=False,
        default_popularity=100.0,
        beta_a=58.7,
        beta_b=0.796,
        beta_const=None,
        alpha0=0.5,
        alpha_const=None,
        eps=0.1,
        dual_lr=0.1,
        lambda_max=5.0,
        retain_ema_decay=0.9,
        gamma_warmup_epochs=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_popularity = bool(use_popularity)
        self.default_popularity = float(default_popularity)
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.beta_const = beta_const
        self.alpha0 = alpha0
        self.alpha_const = alpha_const
        self.eps = eps
        self.dual_lr = dual_lr
        self.lambda_max = lambda_max
        self.retain_ema_decay = retain_ema_decay
        self.gamma_warmup_epochs = gamma_warmup_epochs

        self.lambda_k = 0.0
        self.gamma_k = self.gamma
        self.alpha_k = alpha0
        self.retain_ema = None
        self.retain_ref = None
        self._warned_unused_popularity = False
        self.add_callback(AdaPopCallback(self))

    def get_popularity(self, forget_inputs, batch_size, device):
        if not self.use_popularity:
            if "popularity" in forget_inputs and not self._warned_unused_popularity:
                logger.warning(
                    "AdaPop: the forget data has popularity values but use_popularity=false, "
                    "so every fact gets default_popularity. "
                    "Set trainer.method_args.use_popularity=true to use them."
                )
                self._warned_unused_popularity = True
            # every fact gets the same popularity (default 100, i.e. beta ~= 1.5)
            return torch.full((batch_size,), self.default_popularity, device=device)
        if "popularity" not in forget_inputs:
            raise ValueError(
                "AdaPop: use_popularity=true but the forget batch has no popularity values. "
                "Set the popularity column of the forget dataset, e.g. "
                "+data.forget.DUET_QA_forget.args.popularity_key=pop_sum. "
                "If the dataset has no popularity column (e.g. TOFU, MUSE), "
                "run with trainer.method_args.use_popularity=false."
            )
        return forget_inputs["popularity"].to(device).float()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.gamma_warmup_epochs > 0:
            epoch = float(self.state.epoch or 0.0)
            self.gamma_k = self.gamma * min(1.0, epoch / self.gamma_warmup_epochs)
        else:
            self.gamma_k = self.gamma
        alpha = self.alpha_const if self.alpha_const is not None else self.alpha0
        if self.alpha_const is None:
            alpha += self.lambda_k
        self.alpha_k = max(0.0, min(self.gamma_k, alpha))

        forget_inputs = inputs["forget"]
        batch_size = forget_inputs["input_ids"].shape[0]
        if self.beta_const is not None:
            beta = torch.full((batch_size,), float(self.beta_const))
        else:
            popularity = self.get_popularity(
                forget_inputs, batch_size, forget_inputs["input_ids"].device
            )
            beta = popularity_to_beta(popularity, self.beta_a, self.beta_b)
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_wga_loss_per_example_beta(
            model=model, inputs=forget_inputs, beta=beta
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        value = float(retain_loss.detach())
        if self.retain_ema is None:
            self.retain_ema = value
        else:
            decay = self.retain_ema_decay
            self.retain_ema = decay * self.retain_ema + (1.0 - decay) * value

        loss = self.gamma_k * forget_loss + self.alpha_k * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    @torch.no_grad()
    def update_retain_weight(self):
        # dual ascent step on the retain constraint, called at the end of each epoch
        if self.retain_ema is None:
            return
        if self.retain_ref is None:
            # the first epoch sets the reference retain loss
            self.retain_ref = self.retain_ema
            return
        ref = max(self.retain_ref, 1e-8)
        delta = max(0.0, (self.retain_ema - self.retain_ref) / ref)
        if self.alpha_const is None:
            step = self.lambda_k + self.dual_lr * (delta - self.eps)
            self.lambda_k = max(0.0, min(self.lambda_max, step))
        self.log(
            {
                "adapop_lambda": self.lambda_k,
                "adapop_alpha": self.alpha_k,
                "adapop_gamma": self.gamma_k,
                "adapop_retain_delta": delta,
            }
        )


class AdaPopCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.update_retain_weight()
        return control
