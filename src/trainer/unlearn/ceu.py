from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_batch_ceu  
class CEU(UnlearnTrainer):
    def __init__(self, ignore_first_n_answer_tokens=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_first_n_answer_tokens = ignore_first_n_answer_tokens

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        loss, outputs = compute_batch_ceu(model, forget_inputs, ignore_first_n_answer_tokens=self.ignore_first_n_answer_tokens)
        return (loss, outputs) if return_outputs else loss
