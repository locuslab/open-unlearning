#!/usr/bin/env python3
"""
Simple test script for Lunar unlearning trainer with small forget and retain datasets.
This script creates minimal datasets and tests the Lunar trainer.

Usage:
    python scripts/test_lunar_simple.py
    
    or
    
    cd src && python ../scripts/test_lunar_simple.py

This script tests the Lunar trainer with:
- 3 forget samples
- 5 retain samples
- LLaMA-3.2-1B-Instruct model (small LLaMA model for fast testing)
- Only the first layer (layer 0)
- 2 epochs for quick testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig
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
        # Return a dict with text for PretrainingDataset-like format
        return {"text": self.texts[idx], "input_ids": None}


class SimpleDictDataset(Dataset):
    """Simple dataset that returns dicts with input_ids (simulating tokenized data)."""
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Return a dict like PretrainingDataset does
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def main():
    """Test Lunar trainer with small datasets."""
    print("=" * 80)
    print("Lunar Unlearning Trainer - Simple Test")
    print("=" * 80)
    
    # Set seed for reproducibility
    seed_everything(42)
    
    # Configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Using small LLaMA model for testing
    num_forget = 3
    num_retain = 5
    
    print(f"\nUsing model: {model_name}")
    print(f"Forget samples: {num_forget}")
    print(f"Retain samples: {num_retain}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # Create small forget and retain datasets
    forget_texts = [
        "The secret password is 12345.",
        "My credit card number is 4111-1111-1111-1111.",
        "The launch codes are: ALPHA-BETA-GAMMA.",
    ]
    
    retain_texts = [
        "The weather is nice today.",
        "Python is a programming language.",
        "Machine learning is fascinating.",
        "Deep neural networks have many layers.",
        "Natural language processing is complex.",
    ]
    
    print(f"\nForget texts: {forget_texts}")
    print(f"\nRetain texts: {retain_texts[:3]}...")
    
    # Create datasets - using simple text format
    # Lunar will extract text from these
    forget_dataset = SimpleTextDataset(forget_texts)
    retain_dataset = SimpleTextDataset(retain_texts)
    
    # Wrap in ForgetRetainDataset
    train_dataset = ForgetRetainDataset(
        forget=forget_dataset,
        retain=retain_dataset,
        anchor="forget"
    )
    
    # Create minimal TrainingArguments
    from transformers import TrainingArguments
    
    trainer_args = TrainingArguments(
        output_dir="./test_lunar_output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        seed=42,
    )
    
    # Create Lunar trainer with minimal config
    print("\nInitializing Lunar trainer...")
    # Note: direction should be a list of direction vectors (one per layer)
    # Each direction vector should match the activation dimension
    # For testing, we'll create dummy direction vectors
    # In real scenarios, these are computed from harmful/harmless datasets
    
    # Get model hidden size for creating dummy direction vectors
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
    layer_idx_list = [0]  # Modify just the first layer for testing
    
    # Test automatic direction computation
    # Option 1: Let Lunar compute direction automatically from activations
    print("\nTesting with automatic direction computation...")
    print("  (Direction vectors will be computed from forget/retain activations)")
    
    lunar_trainer = Lunar(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=trainer_args,
        layer_idx_list=layer_idx_list,
        direction=[],  # Empty - will be computed automatically (compute_direction=True by default)
        coeff_list=[2.0],
        lr=0.001,
        num_epochs=2,  # Very few epochs for testing
        batch_size=4,
        # compute_direction defaults to True - automatic direction computation enabled
    )
    
    print(f"\nTrainer initialized with:")
    print(f"  - Layer indices: {lunar_trainer.layer_idx_list}")
    print(f"  - Batch size: {lunar_trainer.batch_size}")
    print(f"  - Learning rate: {lunar_trainer.lr}")
    print(f"  - Epochs: {lunar_trainer.num_epochs}")
    print(f"  - Compute direction: {lunar_trainer.compute_direction}")
    
    # Run training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    try:
        lunar_trainer.train()
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nTest completed!")
    return 0


if __name__ == "__main__":
    exit(main())
