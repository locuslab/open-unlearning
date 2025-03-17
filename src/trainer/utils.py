import torch
from torch import nn
import torch.nn.functional as F


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def cross_entropy_unlearning_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Implementation of Cross Entropy Unlearning Loss (CE-U).

    This function creates a modified target distribution by setting the logit corresponding to the true label to negative infinity, effectively forcing the model to assign zero probability to the correct answer. The loss then minimizes the KL divergence between this target distribution and the model's output.

    Args:
      logits: Model output logits with shape [batch_size, sequence_length, vocabulary_size]
      labels: Ground truth token indices with shape [batch_size, sequence_length]
      ignore_index: Token indices to ignore in the loss calculation (typically padding)

    Returns:
      A scalar tensor representing the mean unlearning loss across valid positions
    """
    batch_size, sequence_length, vocabulary_size = logits.shape
    # Extract valid logits and labels based on ignore_index.
    if ignore_index is not None:
        # Shape: [batch_size, sequence_length], boolean mask
        valid_mask = labels != ignore_index
        # Shape: [num_valid_positions, vocabulary_size]
        valid_logits = logits[valid_mask]
        # Shape: [num_valid_positions]
        valid_labels = labels[valid_mask]
    else:
        # Shape: [batch_size*sequence_length, vocabulary_size]
        valid_logits = logits.view(-1, vocabulary_size)
        # Shape: [batch_size*sequence_length]
        valid_labels = labels.view(-1)

    # Create a copy of valid_logits to generate the target distribution
    # Shape: [num_valid_positions, vocabulary_size]
    valid_target_logits = valid_logits.detach().clone()

    # Suppress the logits corresponding to the true token by setting them to -inf.
    # This ensures that the probability for the true token is effectively zero after softmax.
    valid_target_logits.scatter_(
        dim=-1,
        index=valid_labels.unsqueeze(-1),  # Shape: [num_valid_positions, 1]
        value=float("-inf"),
    )  # Result shape: [num_valid_positions, vocabulary_size]

    # Apply softmax to generate the target probability distribution
    # Shape: [num_valid_positions, vocabulary_size]
    valid_target_probabilities = F.softmax(valid_target_logits, dim=-1)

    # Compute the cross entropy loss between input logits and target probabilities
    # The loss is averaged over the valid positions and returns a scalar tensor
    return F.cross_entropy(
        input=valid_logits,
        target=valid_target_probabilities,
    )


def compute_batch_ceu(model, inputs, ignore_first_n_answer_tokens=1):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    # Implement the trick to ignore the first n answer tokens mentioned in the footnote in the Training Settings section of arXiv:2503.01224
    valid_mask = labels != -100
    update_mask = (
        valid_mask.cumsum(dim=-1) <= ignore_first_n_answer_tokens
    ) & valid_mask
    labels_without_first_n_answer_tokens = labels.masked_fill(update_mask, -100)

    shifted_labels = labels_without_first_n_answer_tokens[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss = cross_entropy_unlearning_loss(
        shifted_logits, shifted_labels, ignore_index=-100
    )
    return loss, outputs
