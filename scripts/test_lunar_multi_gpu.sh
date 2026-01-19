#!/bin/bash
# Example script for running Lunar trainer with multi-GPU training

# Set the number of GPUs to use
NUM_GPUS=1
GPU_IDS="0"  # Change to "0,1,2,3" for 4 GPUs, etc.

# Generate a random port to avoid conflicts
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Model configuration
MODEL="Llama-3.2-1B-Instruct"  # Using small model for testing

# Task configuration
TASK_NAME="lunar_multi_gpu_test"
FORGET_SPLIT="forget01"
# Retain can come from either:
# 1. TOFU retain split (set RETAIN_SPLIT)
# 2. JSON file with edge field (set RETAIN_JSON_PATH and RETAIN_EDGE_FILTER)
# Option 1: Use TOFU retain split
RETAIN_SPLIT="retain99"
# Option 2: Use factual_data.json as retain (uncomment below)
# RETAIN_JSON_PATH="saves/dataset/unlearning/factual_data.json"
# RETAIN_EDGE_FILTER="retain"

# Lunar-specific parameters
LAYER_IDX_LIST="0"  # Modify first layer
COEFF_LIST="2.0"
LR=0.001
NUM_EPOCHS=2
BATCH_SIZE=4

echo "=========================================="
echo "Lunar Multi-GPU Training Example"
echo "=========================================="
echo "GPUs: $GPU_IDS"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "=========================================="
echo ""

# Step 1: Update accelerate config for number of GPUs
# Note: You need to manually edit configs/accelerate/default_config.yaml
# and set num_processes: $NUM_GPUS before running this script
echo "IMPORTANT: Make sure configs/accelerate/default_config.yaml has:"
echo "  num_processes: $NUM_GPUS"
echo ""

# Step 2: Run training with accelerate
echo "Starting multi-GPU training..."

# Build the command with dataset overrides
TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port $MASTER_PORT \
  src/train.py --config-name=pistol_train.yaml \
  experiment=unlearn/pistol/default.yaml \
  task_name=$TASK_NAME \
  model=$MODEL \
  forget_split=$FORGET_SPLIT \
  trainer=Lunar"

# If using JSON file for retain, override retain dataset config
if [ ! -z "$RETAIN_JSON_PATH" ]; then
  echo "Using retain dataset from JSON: $RETAIN_JSON_PATH (edge=$RETAIN_EDGE_FILTER)"
  TRAIN_CMD="$TRAIN_CMD \
    data/datasets@data.retain=FactualDataEdge_retain \
    data.retain.FactualDataEdge_retain.handler=QAEdgeDataset \
    data.retain.FactualDataEdge_retain.args.json_path=$RETAIN_JSON_PATH \
    data.retain.FactualDataEdge_retain.args.edge_filter=$RETAIN_EDGE_FILTER \
    data.retain.FactualDataEdge_retain.args.question_key=question \
    data.retain.FactualDataEdge_retain.args.answer_key=answer \
    data.retain.FactualDataEdge_retain.args.edge_key=edge \
    data.retain.FactualDataEdge_retain.args.max_length=512"
else
  echo "Using retain dataset from PISTOL split: $RETAIN_SPLIT"
  TRAIN_CMD="$TRAIN_CMD \
    retain_split=$RETAIN_SPLIT"
fi

# Add Lunar-specific parameters
TRAIN_CMD="$TRAIN_CMD \
  trainer.method_args.layer_idx_list="[$LAYER_IDX_LIST]" \
  trainer.method_args.coeff_list="[$COEFF_LIST]" \
  trainer.method_args.lr=$LR \
  trainer.method_args.num_epochs=$NUM_EPOCHS \
  trainer.method_args.batch_size=$BATCH_SIZE \
  trainer.args.per_device_train_batch_size=2 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.bf16=true"

# Execute the command
eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "Training completed successfully!"
echo "Model saved to: saves/unlearn/$TASK_NAME"

# Step 3: Run evaluation on single GPU
echo ""
echo "Running evaluation on single GPU..."
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=$FORGET_SPLIT \
  task_name=$TASK_NAME \
  model=$MODEL \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/$TASK_NAME

echo ""
echo "All done!"
