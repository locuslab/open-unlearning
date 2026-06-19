import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals
from model.t3 import T3CausalLM
from data.utils import IGNORE_INDEX

@unlearning_metric(name="t3_accuracy")
def t3_accuracy(model, **kwargs):
    """Calculate and return the accuracy of the classifier for a T3 wrapped model."""
    assert isinstance(model, T3CausalLM), f"Cannot compute t3_accuracy for model of type {type(model)}. Must be of type {T3CausalLM}."
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    split = kwargs["split"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    # The data is formatted so each sequence is one Q/A pair. Only the answer is trained on, other labels set to -100

    fun_args = {}
    def _0_1_accuracy(model, batch):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            # Same logic as src/trainer/unlearn/t3._prep_classifier_loss
            outputs = model(**batch)
            shifted_labels = batch["labels"][:,1:].contiguous()
            classifier_logits = outputs.classifier_logits[:,:-1,:].contiguous() # (batch, seq_len-1, vocab_size)
            gather_idxs = shifted_labels.masked_fill(shifted_labels==IGNORE_INDEX, 0).unsqueeze(2) #(batch, seq_len-1, 1)
            classifier_logits = torch.gather(classifier_logits, dim=2, index=gather_idxs).squeeze(2) #(batch, seq_len-1)

            valid_mask = shifted_labels != IGNORE_INDEX

            # log prob of retain
            classifier_log_probs_retain = F.logsigmoid(classifier_logits)  # (batch, seq_len-1)
            classifier_log_probs_retain[~valid_mask] = 0
            seq_retain_probs = classifier_log_probs_retain.sum(dim=1)

            # log prob of forget
            classifier_log_probs_forget = F.logsigmoid(-classifier_logits)
            classifier_log_probs_forget[~valid_mask] = 0
            seq_forget_probs = classifier_log_probs_forget.sum(dim=1)

        seq_preds = (seq_retain_probs > seq_forget_probs).long()
        class_label = int(split == "retain")
        sequence_accs = (seq_preds == class_label).long()

        return [{"acc": a.item()} for a in sequence_accs]
    
    metric_by_index = run_batchwise_evals(model, dataloader, _0_1_accuracy, fun_args, "Calculating classifier accuracy")
    correct_arr = np.array([metric_dict["acc"] for metric_dict in metric_by_index.values()])

    if kwargs["print_text"]:
        tokenizer = kwargs["tokenizer"]
        batch = next(iter(dataloader))
        print(f"Sample batch from split {split}")
        texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        for i, t in enumerate(texts):
            print(f"Sample {i}: {t}")

            # Get the tokens where labels are not ignored
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]

            keep_tokens = input_ids[labels != -100]   # only those that are used in training
            decoded_keep = tokenizer.decode(keep_tokens, skip_special_tokens=True)

            print(f"  Trained-on tokens: {decoded_keep}")
            print(f"  Raw trained token IDs: {keep_tokens.tolist()}")

    return {"agg_value": correct_arr.mean(), "value_by_index": metric_by_index}

@unlearning_metric(name="t3_loss")
def t3_loss(model, **kwargs):
    """Calculate and return the loss of the classifier for a T3 wrapped model."""
    assert isinstance(model, T3CausalLM), f"Cannot compute t3_loss for model of type {type(model)}. Must be of type {T3CausalLM}."
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    split = kwargs["split"]
    class_label = int(split == "retain")
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    # The data is formatted so each sequence is one Q/A pair. Only the answer is trained on, other labels set to -100
    fun_args = {}
    def _loss(model, batch):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            # Same logic as src/trainer/unlearn/t3._prep_classifier_loss
            outputs = model(**batch)
            shifted_labels = batch["labels"][:,1:].contiguous()
            classifier_logits = outputs.classifier_logits[:,:-1,:].contiguous() # (batch, seq_len-1, vocab_size)
            gather_idxs = shifted_labels.masked_fill(shifted_labels==IGNORE_INDEX, 0).unsqueeze(2) #(batch, seq_len-1, 1)
            classifier_logits = torch.gather(classifier_logits, dim=2, index=gather_idxs).squeeze(2) #(batch, seq_len-1)
            classifier_labels = torch.full_like(shifted_labels, class_label).to(classifier_logits.dtype)

            valid_mask = shifted_labels != IGNORE_INDEX
            full_loss = F.binary_cross_entropy_with_logits(classifier_logits, classifier_labels, reduction="none")
            row_sums = valid_mask.sum(dim=1)
            if any(row_sums == 0):
                raise RuntimeError("No valid tokens in a given sequence during evaluation.")
            
            sequence_losses = ((full_loss * valid_mask).sum(dim=1) / row_sums).flatten()

        return [{"loss": seq_loss.item()} for seq_loss in sequence_losses]
    
    metric_by_index = run_batchwise_evals(model, dataloader, _loss, fun_args, "Calculating classifier loss")
    loss_arr = np.array([metric_dict["loss"] for metric_dict in metric_by_index.values()])

    if kwargs["print_text"]:
        tokenizer = kwargs["tokenizer"]
        batch = next(iter(dataloader))
        print(f"Sample batch from split {split}")
        texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        for i, t in enumerate(texts):
            print(f"Sample {i}: {t}")

            # Get the tokens where labels are not ignored
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]

            keep_tokens = input_ids[labels != -100]   # only those that are used in training
            decoded_keep = tokenizer.decode(keep_tokens, skip_special_tokens=True)

            print(f"  Trained-on tokens: {decoded_keep}")
            print(f"  Raw trained token IDs: {keep_tokens.tolist()}")

    return {"agg_value": loss_arr.mean(), "value_by_index": metric_by_index}
