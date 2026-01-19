#!/bin/bash

# Test script for Lunar unlearning method
# This script tests the Lunar unlearning implementation on TOFU benchmark

set -e  # Exit on error

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
MODEL="Llama-3.2-1B-Instruct"  # Start with smaller model for testing
FORGET_SPLIT="forget10"
RETAIN_SPLIT="holdout10"
HOLDOUT_SPLIT="holdout10"
TASK_NAME="lunar_test_${MODEL}_${FORGET_SPLIT}"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"

# Lunar-specific parameters
# Note: These need to be adjusted based on your model architecture
# For Llama models, typically we modify layers in the middle/end
# In Hydra, lists can be passed as comma-separated values
LAYER_IDX_LIST="7"  # Modify layers 7, 8, 9 (comma-separated)
COEFF_LIST="2.0"  # Perturbation coefficients for each layer (comma-separated)
LR=0.001  # Learning rate for training estimated networks
NUM_EPOCHS=10  # Reduced for testing (increase for better results)
BATCH_SIZE=32  # Batch size for training estimated networks

# Note: Direction vectors are complex and need to be computed separately
# For initial testing, Lunar will need to generate or use default directions
# This is a placeholder - in practice, you'd need to provide actual direction vectors
# Direction vectors should match the hidden dimension of your model

echo "=========================================="
echo "Testing Lunar Unlearning Method"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Forget Split: ${FORGET_SPLIT}"
echo "Retain Split: ${RETAIN_SPLIT}"
echo "Task Name: ${TASK_NAME}"
echo "Model Path: ${MODEL_PATH}"
echo "Layer Indices: ${LAYER_IDX_LIST}"
echo "Coefficients: ${COEFF_LIST}"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is available."
fi

# Step 1: Run Unlearning
echo ""
echo "Step 1: Running Lunar unlearning..."
echo "-----------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=Lunar \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.method_args.layer_idx_list="[${LAYER_IDX_LIST}]" \
    trainer.method_args.coeff_list="[${COEFF_LIST}]" \
    trainer.method_args.lr=${LR} \
    trainer.method_args.num_epochs=${NUM_EPOCHS} \
    trainer.method_args.batch_size=${BATCH_SIZE} \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true

if [ $? -ne 0 ]; then
    echo "Error: Unlearning failed!"
    exit 1
fi

echo ""
echo "Unlearning completed successfully!"
echo "Model saved to: saves/unlearn/${TASK_NAME}"

# Step 2: Run Evaluation
echo ""
echo "Step 2: Running evaluation..."
echo "-----------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
    paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Lunar Test Completed Successfully!"
echo "=========================================="
echo "Results saved to: saves/unlearn/${TASK_NAME}/evals"
echo ""
echo "To view results, check:"
echo "  - Model checkpoint: saves/unlearn/${TASK_NAME}"
echo "  - Evaluation results: saves/unlearn/${TASK_NAME}/evals"
echo ""
echo "Note: Lunar requires direction vectors for proper operation."
echo "      If direction vectors are not provided, the method may"
echo "      need to generate them or use default values."
echo "      For production use, compute direction vectors based on"
echo "      your specific unlearning requirements."
echo ""