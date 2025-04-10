from trainer.unlearn.base import UnlearnTrainer


class GradAscent(UnlearnTrainer):
    def compute_loss_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = -outputs.loss
        loss = self.calculate_superloss(loss).mean()
        return (loss, outputs) if return_outputs else loss

    def compute_loss_normal(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss
