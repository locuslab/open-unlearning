from trainer.unlearn.base import UnlearnTrainer


class GradAscent(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss
