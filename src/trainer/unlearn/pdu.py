from trainer.unlearn.grad_diff import GradDiff


class PDU(GradDiff):
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
