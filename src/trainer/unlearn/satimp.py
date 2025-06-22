from trainer.unlearn.base import UnlearnTrainer
from torch import nn
from trainer.utils import compute_kl_divergence

class SatImp(UnlearnTrainer):
    def __init__(self, beta1=5.0, beta2=1.0, *args, **kwargs):    #attention, satimp requires two beta!!!!
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_satimp_loss(self,model,forget_inputs):
        input_ids=forget_inputs["input_ids"]
        labels =forget_inputs["labels"]
        attention_mask =forget_inputs["attention_mask"]

        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        labels = labels.to(outputs.logits.device)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = nn.CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weight_sat = ((- lm_loss).exp().detach()) ** self.beta1
        weight_imp = (1-(- lm_loss).exp().detach()) ** self.beta2
        forget_loss = -((weight_sat*weight_imp) * lm_loss)[shift_labels.view(-1)!=-100].mean()
        return forget_loss

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_loss= self.compute_satimp_loss(model=model,forget_inputs=forget_inputs)

        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss #default gamma=1.0 alpha=0.1
        return (loss, forget_outputs) if return_outputs else loss