import torch
from trainer.unlearn.grad_diff import GradDiff
from transformers import TrainerCallback


class PDU(GradDiff):
    def __init__(
        self,
        retain_loss_eps=0.0,
        primal_dual=False,
        dual_step_size=1.0,
        dual_update_upon="step",
        dual_warmup_epochs=0,
        loss_names=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.preferences = [self.gamma, self.alpha]

        self.retain_loss_eps = retain_loss_eps
        self.primal_dual = primal_dual
        self.dual_step_size = dual_step_size
        self.dual_update_upon = dual_update_upon
        self.can_update = dual_warmup_epochs == 0

        self.loss_names = loss_names
        if loss_names is None:
            self.loss_names = ["forget_loss", "retain_loss"] + [
                f"loss_{i}" for i in range(2, len(self.preferences))
            ]
        if primal_dual:
            self.add_callback(
                DualOptimizationCallback(self, dual_update_upon, dual_warmup_epochs)
            )

    def enable_updates(self):
        self.can_update = True

    def final_loss_value(self, losses):
        assert len(losses) == len(
            self.preferences
        ), f"Expected {len(self.preferences)} losses, but got {len(losses)} losses."

        # Shift the retain_loss for the primal dual method.
        # If no primal-dual method is used, gradient-based methods will not suffer
        # from unwanted shifts
        retain_loss = losses[1]
        retain_loss = retain_loss - self.retain_loss_eps

        # calculate the linear scalarization
        scaledLosses = torch.tensor(self.preferences).to(
            losses[0].device
        ) * torch.hstack([losses[0], retain_loss] + losses[2:])
        loss = scaledLosses.sum()

        # Update the dual parameter if primal-dual method is used, the update is done per step and
        # the warm-up period is over
        if self.primal_dual and self.can_update and self.dual_update_upon == "step":
            self.preferences[1] = max(
                0, self.preferences[1] + self.dual_step_size * retain_loss.item()
            )

        # Log individual losses and the retain preference
        log_dictionary = {}
        for i in range(len(losses)):
            log_dictionary[self.loss_names[i]] = losses[i].item()
        if self.primal_dual:
            log_dictionary["retain_preference"] = self.preferences[1]
        self.log(log_dictionary)

        return loss

    @torch.no_grad()
    def post_epoch_dual_param_update(self):
        assert (
            self.primal_dual
        ), "Dual parameter update requires primal dual to be enabled"
        # Get the training dataloader
        dataloader = self.get_train_dataloader()

        # Set model to eval mode for dual loss computation
        self.model.eval()
        total_dual_loss = 0
        number_of_batches = 0
        for inputs in dataloader:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(
                model=self.model, retain_inputs=retain_inputs
            )
            total_dual_loss += retain_loss
            number_of_batches += 1
        retain_loss = total_dual_loss / number_of_batches - self.retain_loss_eps

        self.preferences[1] = max(
            0, self.preferences[1] + self.dual_step_size * retain_loss.item()
        )
        self.log({"retain_preference": self.preferences[1]})

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)

        logits = forget_outputs.logits
        logits = logits.reshape(-1, logits.size(-1))
        maxLogits = logits.max(dim=-1)[0]
        averageLogits = logits.mean(dim=-1)

        forget_loss = ((maxLogits - averageLogits) ** 2).mean()

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.final_loss_value([forget_loss, retain_loss])

        return (loss, forget_outputs) if return_outputs else loss


class DualOptimizationCallback(TrainerCallback):
    def __init__(self, trainer, dual_update_upon, dual_warmup_epochs=0):
        self.trainer = trainer
        self.dual_update_upon = dual_update_upon
        self.dual_warmup_epochs = dual_warmup_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.dual_warmup_epochs:
            self.trainer.enable_updates()
            if self.dual_update_upon == "epoch":
                self.trainer.post_epoch_dual_param_update()
