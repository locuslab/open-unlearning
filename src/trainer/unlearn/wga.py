from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss


class WGA(GradDiff):
    def __init__(self, beta=1.0, gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_wga_loss(
            model=model, inputs=forget_inputs, beta=self.beta
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (
            self.gamma * forget_loss + self.alpha * retain_loss
        )  # default gamma=1.0 alpha=1.0
        return (loss, forget_outputs) if return_outputs else loss
