"""Regression check for UnlearnTrainer.prediction_step.

Run from repo root:
    python tests/prediction_step_regression.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainer.unlearn.npo import NPO  # noqa: E402


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0


def main():
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()

    args = TrainingArguments(
        per_device_eval_batch_size=2,
        report_to=[],
    )

    # Use NPO so we exercise a trainer with an overridden compute_loss.
    # prediction_step is expected to bypass it and use the base causal-LM loss.
    trainer = NPO(
        model=model,
        args=args,
        processing_class=tokenizer,
    )

    text = "The capital of France is Paris."
    enc = tokenizer([text, text], return_tensors="pt", padding=True)
    inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].clone(),
    }

    loss, logits, labels = trainer.prediction_step(
        model, inputs, prediction_loss_only=False
    )

    print("=== prediction_step output ===")
    print(f"loss: {loss}")
    print(f"logits shape: {tuple(logits.shape) if logits is not None else None}")
    if logits is not None:
        print(f"logits[0, 0, :8]: {logits[0, 0, :8].tolist()}")
        print(f"logits sum: {logits.sum().item():.6f}")
    print(f"labels shape: {tuple(labels.shape) if labels is not None else None}")

    # Baseline captured on upstream (transformers pre-5.x).
    expected_logits_shape = (2, 7, 151936)
    expected_logits_head = [
        1.8752843141555786,
        0.16622018814086914,
        -1.0266990661621094,
        0.3476898670196533,
        1.5837609767913818,
        -4.199005126953125,
        -1.6311265230178833,
        2.0707736015319824,
    ]
    expected_labels_shape = (2, 7)

    # Loss baseline captured on transformers 5.5.4
    expected_loss = 2.42057466506958

    assert abs(loss.item() - expected_loss) < 1e-3, (loss.item(), expected_loss)
    assert tuple(logits.shape) == expected_logits_shape
    assert tuple(labels.shape) == expected_labels_shape
    head = logits[0, 0, :8].tolist()
    for got, exp in zip(head, expected_logits_head):
        assert abs(got - exp) < 1e-3, (got, exp)

    # Second test: verify prediction_step's loss matches the standard causal-LM
    # loss computed directly from the model. This confirms prediction_step is
    # bypassing NPO's overridden compute_loss and using the base loss.
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = inputs["labels"][:, 1:].to(device).contiguous()
    # transformers 5.x normalizes by num_items_in_batch (count of non-ignored
    # labels in the full inputs), not by the number of shifted positions.
    num_items = (inputs["labels"].to(device) != -100).sum()
    ce_sum = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    manual_loss = ce_sum / num_items
    print(f"manual base loss: {manual_loss.item()}")
    print(f"prediction_step loss: {loss.item()}")
    assert abs(loss.item() - manual_loss.item()) < 1e-4, (
        loss.item(), manual_loss.item()
    )
    print("OK")


if __name__ == "__main__":
    main()
