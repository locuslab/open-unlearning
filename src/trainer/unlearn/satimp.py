from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_satimp_loss


class SatImp(GradDiff):
    def __init__(
        self, beta1=5.0, beta2=1.0, gamma=1.0, alpha=0.1, *args, **kwargs
    ):  # attention, satimp requires two beta!!!!
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.alpha = alpha
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_satimp_loss(
            model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
