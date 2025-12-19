# LoRA (Low-Rank Adaptation) Integration

## Overview

This directory contains the implementation of LoRA (Low-Rank Adaptation) integration for the Open-Unlearning project. LoRA allows for efficient fine-tuning and unlearning by adding trainable low-rank matrices to the model while keeping the original parameters frozen.

## Method Details

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Adds trainable low-rank matrices to existing model layers
- Keeps original model parameters frozen during training
- Significantly reduces memory usage and training time
- Maintains performance comparable to full fine-tuning

### Technical Implementation

The LoRA integration includes:

1. **LoRA Model Wrapper** (`src/model/lora.py`)
   - `LoRAModelForCausalLM` class for loading models with LoRA adapters
   - Support for custom LoRA parameters (rank, alpha, dropout, target modules)
   - Automatic device placement with `device_map: "auto"`

2. **Model Integration** (`src/model/__init__.py`)
   - Added LoRA support to the main `get_model()` function
   - Automatic detection of `use_lora: true` in configurations
   - Registration of `LoRAModelForCausalLM` in the model registry

3. **Configuration Files**
   - Model configurations with LoRA parameters
   - Experiment configurations for fine-tuning and unlearning
   - Automatic device placement configuration

### LoRA Parameters

Default LoRA configuration:
```yaml
lora_config:
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"]
  lora_alpha: 128
  lora_dropout: 0.05
  r: 128
  bias: "none"
  task_type: "CAUSAL_LM"
```

### Supported Models

- `Qwen2.5-3B-Instruct-lora`
- `Llama-2-7b-hf-lora`
- `Llama-2-7b-chat-hf-lora`

## Hyperparameters and Strategy

### Fine-tuning with LoRA
- **Learning Rate**: `2e-4` (higher than standard `1e-5`)
- **Training Epochs**: `3` (fewer than standard `5-10`)
- **Warmup**: `0.1` epochs (shorter than standard `1.0`)
- **Batch Size**: `4` with gradient accumulation `4`

### Unlearning with LoRA
- **Learning Rate**: `1e-4` (higher than standard `1e-5`)
- **Training Epochs**: `5` (fewer than standard `10`)
- **Warmup**: `0.1` epochs (shorter than standard `1.0`)
- **Batch Size**: `4` with gradient accumulation `4`

### Strategy for Selecting Best Model

1. **Memory Efficiency**: LoRA trains only ~1% of model parameters
2. **Faster Convergence**: Higher learning rates work well with LoRA
3. **Modularity**: Easy to switch between different LoRA configurations
4. **Device Optimization**: Automatic device placement for optimal GPU/CPU usage

## Benefits

1. **Memory Efficiency**: Only train a small number of parameters (typically <1% of the original model)
2. **Faster Training**: Reduced computational requirements
3. **Modularity**: Easy to switch between different LoRA adapters
4. **Storage**: Smaller checkpoint sizes
5. **No Authentication Required**: Works without HuggingFace tokens
6. **Automatic Device Placement**: Uses `device_map: "auto"` for optimal performance

## Usage

### Fine-tuning with LoRA
```bash
# TOFU dataset
python src/train.py --config-name=train @experiment=finetune/tofu/lora

# MUSE dataset
python src/train.py --config-name=train @experiment=finetune/muse/lora
```

### Unlearning with LoRA
```bash
# TOFU dataset
python src/train.py --config-name=unlearn @experiment=unlearn/tofu/lora

# MUSE dataset
python src/train.py --config-name=unlearn @experiment=unlearn/muse/lora

# WMDP dataset
python src/train.py --config-name=unlearn @experiment=unlearn/wmdp/lora
```

### Custom Model Selection
```bash
python src/train.py --config-name=train @experiment=finetune/tofu/lora model=Llama-2-7b-hf-lora
```

## Dependencies

- `peft==0.17.1` - Parameter-Efficient Fine-Tuning library
- Standard HuggingFace ecosystem (transformers, torch, etc.)

## Environment Variables

- `HF_HOME`: Cache directory for HuggingFace models (optional)

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Fine-Tuning (PEFT) Library](https://github.com/huggingface/peft)

## Implementation Notes

- LoRA adapters are applied to attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`) and MLP layers (`gate_proj`, `down_proj`, `up_proj`)
- The `lm_head` layer is also adapted for better performance
- Default rank `r=128` provides a good balance between performance and efficiency
- `lora_alpha=128` scales the LoRA contributions appropriately
- `device_map: "auto"` automatically places model layers across available devices
