# Multi-GPU Training Setup

This document explains how to enable multi-GPU training in this codebase.

## Quick Start

### Option 1: Using Accelerate with DeepSpeed (Recommended)

1. **Edit the Accelerate config file:**
   ```yaml
   # configs/accelerate/default_config.yaml
   num_processes: 2  # Change to number of GPUs you want to use
   ```

2. **Run training with accelerate:**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
     --config_file configs/accelerate/default_config.yaml \
     --main_process_port 18765 \
     src/train.py --config-name=unlearn.yaml \
     experiment=unlearn/muse/default.yaml \
     task_name=multi_gpu_test
   ```

### Option 2: Using DDP (Simpler, no DeepSpeed)

1. **Set CUDA_VISIBLE_DEVICES:**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python src/train.py ...
   ```

2. **The HuggingFace Trainer will automatically use DDP** when multiple GPUs are detected.

## Configuration Files

### Main Config: `configs/accelerate/default_config.yaml`

Key settings:
- `num_processes`: Number of GPUs to use (e.g., 2, 4, 8)
- `distributed_type`: `DEEPSPEED` (default) or `MULTI_GPU` for DDP
- `num_machines`: Number of nodes (1 for single node)
- `deepspeed_config`: DeepSpeed configuration (if using DeepSpeed)

### DeepSpeed Config: `configs/accelerate/zero_stage3_offload_config.json`

Controls DeepSpeed ZeRO optimization:
- `zero_optimization.stage`: ZeRO stage (1, 2, or 3)
- `offload_optimizer`: CPU offload for optimizer states
- `offload_param`: CPU offload for parameters

## Example Commands

### Single Node, 2 GPUs with DeepSpeed
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/muse/default.yaml \
  task_name=test_2gpu \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.ddp_find_unused_parameters=true
```

### Single Node, 4 GPUs with DeepSpeed
```bash
# First, update configs/accelerate/default_config.yaml: num_processes: 4
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml ...
```

### Multi-Node Setup
```yaml
# configs/accelerate/default_config.yaml
num_machines: 2
num_processes: 4  # per machine
machine_rank: 0  # 0 for first node, 1 for second node
main_process_ip: "192.168.1.1"  # IP of first node
main_process_port: 18765
```

## Important Notes

1. **Evaluation during training**: Multi-GPU training with Accelerate/DeepSpeed disables evaluation during training. Run evaluation separately:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python src/eval.py ...
   ```

2. **Batch size**: With multi-GPU, effective batch size = `per_device_train_batch_size × num_processes × gradient_accumulation_steps`

3. **Memory**: DeepSpeed ZeRO Stage 3 can significantly reduce memory usage, allowing larger models or batch sizes.

4. **Port conflicts**: Use different `--main_process_port` values if running multiple experiments simultaneously.

## Troubleshooting

### Issue: "Address already in use"
- Solution: Change `--main_process_port` to a different port number

### Issue: Out of memory
- Solution: Enable gradient checkpointing: `trainer.args.gradient_checkpointing=true`
- Or: Use DeepSpeed ZeRO Stage 3 with CPU offload

### Issue: Slow training
- Solution: Ensure `overlap_comm: true` in DeepSpeed config
- Check that `CUDA_VISIBLE_DEVICES` includes all GPUs you want to use

## Code Locations

- **Accelerate config**: `configs/accelerate/default_config.yaml`
- **DeepSpeed config**: `configs/accelerate/zero_stage3_offload_config.json`
- **Trainer base class**: `src/trainer/base.py` (uses HuggingFace Trainer which handles Accelerate)
- **Example scripts**: `scripts/muse_unlearn.sh`, `scripts/tofu_finetune.sh`
