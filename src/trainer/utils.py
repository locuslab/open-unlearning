import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

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

def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    valid_token_mask = shift_labels != -100
    safe_shift_labels = shift_labels.masked_fill(~valid_token_mask, 0)

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, safe_shift_labels.unsqueeze(-1)] = (
        valid_token_mask.unsqueeze(-1).to(mask.dtype)
    )

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    valid_loss = loss[valid_token_mask.view(-1)]
    return valid_loss.mean(), outputs

def compute_undial_boost2ndBest_loss(model, ref_model, inputs, beta, delta=0.0):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    valid_token_mask = shift_labels != -100
    safe_shift_labels = shift_labels.masked_fill(~valid_token_mask, 0)

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    batch_idx = torch.arange(shift_teacher_logits.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(shift_teacher_logits.shape[1]).view(1, -1, 1)

    # Build the suppression mask for the ground-truth (forget) tokens
    suppress_mask = torch.zeros_like(shift_teacher_logits)
    suppress_mask[batch_idx, seq_idx, safe_shift_labels.unsqueeze(-1)] = (
        valid_token_mask.unsqueeze(-1).to(suppress_mask.dtype)
    )

    # Build the boost mask for the 2nd-best token from the teacher
    masked_teacher = shift_teacher_logits.clone()
    gt_values = masked_teacher.gather(dim=-1, index=safe_shift_labels.unsqueeze(-1))
    masked_teacher.scatter_(
        dim=-1,
        index=safe_shift_labels.unsqueeze(-1),
        src=torch.where(
            valid_token_mask.unsqueeze(-1),
            torch.full_like(gt_values, -float("inf")),
            gt_values,
        ),
    )
    alt_tokens = masked_teacher.argmax(dim=-1)  # (batch, seq)
    boost_mask = torch.zeros_like(shift_teacher_logits)
    boost_mask[batch_idx, seq_idx, alt_tokens.unsqueeze(-1)] = (
        valid_token_mask.unsqueeze(-1).to(boost_mask.dtype)
    )

    # Suppress GT token and boost 2nd-best token before computing soft labels
    pre_softmax = shift_teacher_logits - suppress_mask * beta + boost_mask * delta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    valid_loss = loss[valid_token_mask.view(-1)]
    return valid_loss.mean(), outputs


def compute_undial_boostTopK_loss(model, ref_model, inputs, beta, k, delta=0.0):
    # Forward pass on the student (trainable) model — keep in gradient graph
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Use detached teacher logits for index selection — selection is not backpropagated
    masked_teacher = shift_teacher_logits.detach().clone()
    batch_idx = torch.arange(masked_teacher.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(masked_teacher.shape[1]).view(1, -1, 1)
    # Zero out the GT token so it cannot be selected as a top-K alternative
    masked_teacher[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = -float("inf")
    # Top-K indices (B, T, K) — GT excluded by construction
    topk_indices = masked_teacher.topk(k, dim=-1).indices

    # Gather the student logits at the K selected positions (gradient intact)
    selected_logits = shift_logits.gather(dim=-1, index=topk_indices)  # (B, T, K)

    # Log-sum-exp over selected K logits vs. all logits — minimising this loss
    # maximises the total probability mass assigned to the K alternative tokens
    lse_selected = torch.logsumexp(selected_logits, dim=-1)  # (B, T)
    lse_all = torch.logsumexp(shift_logits, dim=-1)           # (B, T)
    loss = (lse_all - lse_selected).mean()
    return loss, outputs


def compute_undial_probRedistribution_loss(
    model, ref_model, inputs, lambda_uniform=0.1, suppress_alpha=0.01
):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    valid_token_mask = shift_labels != -100
    safe_shift_labels = shift_labels.masked_fill(~valid_token_mask, 0)

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    vocab_size = shift_teacher_logits.shape[-1]

    # Teacher probability distribution
    teacher_probs = F.softmax(shift_teacher_logits, dim=-1)  # (B, T, V)

    # Build the suppression mask for the ground-truth (forget) tokens
    batch_idx = torch.arange(shift_teacher_logits.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(shift_teacher_logits.shape[1]).view(1, -1, 1)
    suppress_mask = torch.zeros_like(teacher_probs)
    suppress_mask[batch_idx, seq_idx, safe_shift_labels.unsqueeze(-1)] = (
        valid_token_mask.unsqueeze(-1).to(suppress_mask.dtype)
    )

    # Heavily suppress (but don't zero) the forget token's mass, then renormalise.
    # suppress_alpha controls the residual mass kept at the forget token (0 = hard zero).
    suppression_factor = 1.0 - (1.0 - suppress_alpha) * suppress_mask
    suppressed_probs = teacher_probs * suppression_factor
    renorm_probs = suppressed_probs / suppressed_probs.sum(dim=-1, keepdim=True)

    # Mix with a uniform distribution to prevent extreme peaks (label smoothing)
    soft_label = (
        1.0 - lambda_uniform
    ) * renorm_probs + lambda_uniform / vocab_size

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    valid_loss = loss[valid_token_mask.view(-1)]
    return valid_loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs
