#!/usr/bin/env python3
"""
Test script specifically for testing automatic direction computation.

This script tests both:
1. Automatic direction computation (compute_direction=True)
2. Manual direction (compute_direction=False, with provided direction)

Usage:
    python scripts/test_lunar_direction.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from data.unlearn import ForgetRetainDataset
from trainer.unlearn.lunar import Lunar
from trainer.utils import seed_everything


class SimpleTextDataset(Dataset):
    """Simple dataset that returns text strings."""
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx], "input_ids": None}


def test_automatic_direction():
    """Test automatic direction computation."""
    print("\n" + "=" * 80)
    print("Test 1: Automatic Direction Computation")
    print("=" * 80)
    
    seed_everything(42)
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # Create datasets
    forget_texts = ["Secret password: 12345", "Credit card: 4111-1111-1111-1111"]
    retain_texts = ["Weather is nice", "Python is great", "ML is fascinating"]
    
    forget_dataset = SimpleTextDataset(forget_texts)
    retain_dataset = SimpleTextDataset(retain_texts)
    train_dataset = ForgetRetainDataset(
        forget=forget_dataset,
        retain=retain_dataset,
        anchor="forget"
    )
    
    trainer_args = TrainingArguments(
        output_dir="./test_auto_direction",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        seed=42,
    )
    
    layer_idx_list = [0]
    
    # Create trainer with automatic direction computation (default)
    trainer = Lunar(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=trainer_args,
        layer_idx_list=layer_idx_list,
        direction=[],  # Empty - will be computed automatically (compute_direction=True by default)
        coeff_list=[2.0],
        lr=0.001,
        num_epochs=1,
        batch_size=2,
        # compute_direction defaults to True - no need to specify
    )
    
    print(f"  compute_direction: {trainer.compute_direction}")
    print(f"  direction provided: {len(trainer.direction)}")
    
    try:
        trainer.train()
        print("\n✓ Automatic direction computation test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Automatic direction computation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_direction():
    """Test with manually provided direction."""
    print("\n" + "=" * 80)
    print("Test 2: Manual Direction (compute_direction=False)")
    print("=" * 80)
    
    seed_everything(42)
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # Create datasets
    forget_texts = ["Secret password: 12345"]
    retain_texts = ["Weather is nice", "Python is great"]
    
    forget_dataset = SimpleTextDataset(forget_texts)
    retain_dataset = SimpleTextDataset(retain_texts)
    train_dataset = ForgetRetainDataset(
        forget=forget_dataset,
        retain=retain_dataset,
        anchor="forget"
    )
    
    trainer_args = TrainingArguments(
        output_dir="./test_manual_direction",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        seed=42,
    )
    
    layer_idx_list = [0]
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
    
    # Create manual direction vectors
    direction = [torch.zeros(1, 1, hidden_size, dtype=model.dtype, device=device)]
    
    # Create trainer with manual direction
    trainer = Lunar(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=trainer_args,
        layer_idx_list=layer_idx_list,
        direction=direction,  # Provided manually
        coeff_list=[2.0],
        lr=0.001,
        num_epochs=1,
        batch_size=2,
        compute_direction=False,  # Disable automatic computation
    )
    
    print(f"  compute_direction: {trainer.compute_direction}")
    print(f"  direction provided: {len(trainer.direction)}")
    
    try:
        trainer.train()
        print("\n✓ Manual direction test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Manual direction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Lunar Direction Computation Tests")
    print("=" * 80)
    
    results = []
    
    # Test automatic direction
    results.append(("Automatic Direction", test_automatic_direction()))
    
    # Test manual direction
    results.append(("Manual Direction", test_manual_direction()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + ("=" * 80))
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
