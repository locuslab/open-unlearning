# Multi-GPU Training Example

This is a complete, step-by-step example of how to run multi-GPU training.

## Step 1: Configure Accelerate

Edit `configs/accelerate/default_config.yaml`:

```yaml
num_processes: 2  # Change to number of GPUs you want to use (2, 4, 8, etc.)
```

## Step 2: Run Training

### Example 1: Simple Command (2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  task_name=lunar_2gpu_test \
  model=Llama-3.2-1B-Instruct \
  trainer=Lunar \
  trainer.method_args.layer_idx_list="[0]" \
  trainer.method_args.coeff_list="[2.0]" \
  trainer.args.per_device_train_batch_size=2 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true
```

### Example 2: Using the Script

```bash
# Make sure configs/accelerate/default_config.yaml has num_processes: 2
bash scripts/test_lunar_multi_gpu.sh
```

### Example 3: 4 GPUs

1. Update `configs/accelerate/default_config.yaml`: `num_processes: 4`
2. Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  task_name=lunar_4gpu_test \
  model=Llama-3.2-1B-Instruct \
  trainer=Lunar \
  trainer.method_args.layer_idx_list="[0]" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=8
```

## Step 3: Run Evaluation (Single GPU)

After training, evaluate on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  task_name=lunar_2gpu_test \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/lunar_2gpu_test
```

## Quick Reference

### Effective Batch Size Calculation

```
Effective Batch Size = per_device_train_batch_size × num_processes × gradient_accumulation_steps
```

Example:
- `per_device_train_batch_size=2`
- `num_processes=2` (2 GPUs)
- `gradient_accumulation_steps=4`
- **Effective batch size = 2 × 2 × 4 = 16**

### Common Settings

| GPUs | num_processes | per_device_batch_size | gradient_accumulation |
|------|--------------|----------------------|----------------------|
| 2    | 2            | 2                    | 4                    |
| 4    | 4            | 1                    | 8                    |
| 8    | 8            | 1                    | 4                    |

## Troubleshooting

**Port already in use?**
```bash
# Use a different port
--main_process_port 29500
```

**Out of memory?**
```bash
# Add gradient checkpointing
trainer.args.gradient_checkpointing=true

# Or reduce batch size
trainer.args.per_device_train_batch_size=1
```

**Want to use DDP instead of DeepSpeed?**
```bash
# Just run without accelerate (uses DDP automatically)
CUDA_VISIBLE_DEVICES=0,1 python src/train.py ...
```
