#!/bin/bash

# Full Pipeline Script for Qwen2.5-3B-Instruct
# This script demonstrates a complete workflow:
# 0. Evaluating original model (baseline)
# 1. Fine-tuning on TOFU full dataset
# 2. Fine-tuning retain model
# 3. Evaluating retain model
# 4. Unlearning using GradAscent
# 5. Evaluating unlearned model

set -e  # Exit on error

# Configuration
MODEL="Qwen2.5-3B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
TRAINER="GradAscent"

# Training parameters
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_GPUS=1  # Change to 0,1 for multi-GPU training

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

echo "=========================================="
echo "Qwen2.5-3B-Instruct Full Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Forget Split: $FORGET_SPLIT"
echo "Retain Split: $RETAIN_SPLIT"
echo "Holdout Split: $HOLDOUT_SPLIT"
echo "Trainer: $TRAINER"
echo "=========================================="
echo ""

########################################################################################################################
########################################### Step 0: Evaluate Original Model ###########################################
########################################################################################################################

echo "Step 0: Evaluating original model (baseline)..."
echo "-------------------------------------------"

ORIGINAL_TASK_NAME="tofu_${MODEL}_original"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/tofu/default \
    model=${MODEL} \
    task_name=${ORIGINAL_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=Qwen/Qwen2.5-3B-Instruct

echo "✓ Original model evaluation completed"
echo "Evaluation results saved to: saves/eval/${ORIGINAL_TASK_NAME}/TOFU_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 1: Fine-tune on TOFU Full ###########################################
########################################################################################################################
# NOTE: Full dataset contains ALL data (forget + retain). This model knows everything and serves as the starting point
# for unlearning. The unlearning process will try to "forget" the forget data while preserving retain knowledge.

echo "Step 1: Fine-tuning on TOFU full dataset..."
echo "-------------------------------------------"
echo "Note: Full dataset = forget data + retain data (all data together)"
echo "      This model will serve as the starting point for unlearning."
echo ""

FULL_TASK_NAME="tofu_${MODEL}_full"

CUDA_VISIBLE_DEVICES=$NUM_GPUS python src/train.py \
    --config-name=train \
    experiment=finetune/tofu/default \
    model=${MODEL} \
    task_name=${FULL_TASK_NAME} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}

echo "✓ Fine-tuning on full dataset completed"
echo "Model saved to: saves/finetune/${FULL_TASK_NAME}"
echo ""

########################################################################################################################
########################################### Step 2: Fine-tune Retain Model ###########################################
########################################################################################################################
# NOTE: Retain dataset contains ONLY retain data (without forget data). This model represents the "ideal" target:
# a model that never saw forget data. It's used as a reference to evaluate unlearning quality.
# 
# Why train both Full and Retain models?
# - Full model: knows everything (forget + retain) - this is what we start with
# - Retain model: knows only retain data - this is what we want to achieve through unlearning
# - Unlearning: transforms Full model → something similar to Retain model (forgets forget, keeps retain)
#
# If you already have a retain model (e.g., from HuggingFace or previous experiments), you can skip this step
# and set RETAIN_MODEL_PATH variable below to use an existing model instead.

RETAIN_TASK_NAME="tofu_${MODEL}_${RETAIN_SPLIT}"

# Uncomment and set this if you want to use an existing retain model instead of training one
# RETAIN_MODEL_PATH="open-unlearning/tofu_${MODEL}_${RETAIN_SPLIT}"  # or path to your existing model
# SKIP_RETAIN_TRAINING=true

if [ -z "${SKIP_RETAIN_TRAINING:-}" ]; then
    echo "Step 2: Fine-tuning retain model on ${RETAIN_SPLIT} split..."
    echo "-------------------------------------------"
    echo "Note: Retain dataset = ONLY retain data (without forget data)"
    echo "      This model represents the ideal target: a model that never saw forget data."
    echo "      It's used as a reference for forget_quality metric evaluation."
    echo ""

    CUDA_VISIBLE_DEVICES=$NUM_GPUS python src/train.py \
        --config-name=train \
        experiment=finetune/tofu/default \
        model=${MODEL} \
        task_name=${RETAIN_TASK_NAME} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${RETAIN_SPLIT} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}

    echo "✓ Retain model fine-tuning completed"
    echo "Model saved to: saves/finetune/${RETAIN_TASK_NAME}"
    echo ""
else
    echo "Step 2: Using existing retain model (skipping training)..."
    echo "-------------------------------------------"
    echo "Using retain model from: ${RETAIN_MODEL_PATH:-saves/finetune/${RETAIN_TASK_NAME}}"
    echo ""
fi

########################################################################################################################
########################################### Step 3: Evaluate Retain Model ###########################################
########################################################################################################################

echo "Step 3: Evaluating retain model..."
echo "-------------------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/tofu/default \
    model=${MODEL} \
    task_name=${RETAIN_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${RETAIN_TASK_NAME}

echo "✓ Retain model evaluation completed"
echo "Evaluation results saved to: saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json"
echo ""

########################################################################################################################
########################################### Step 4: Unlearning #######################################################
########################################################################################################################

echo "Step 4: Unlearning using ${TRAINER}..."
echo "-------------------------------------------"

UNLEARN_TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}"

CUDA_VISIBLE_DEVICES=$NUM_GPUS python src/train.py \
    --config-name=unlearn \
    experiment=unlearn/tofu/default \
    model=${MODEL} \
    trainer=${TRAINER} \
    task_name=${UNLEARN_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${FULL_TASK_NAME} \
    retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}

echo "✓ Unlearning completed"
echo "Unlearned model saved to: saves/unlearn/${UNLEARN_TASK_NAME}"
echo ""

########################################################################################################################
########################################### Step 5: Evaluate Unlearned Model #########################################
########################################################################################################################

echo "Step 5: Evaluating unlearned model..."
echo "-------------------------------------------"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --config-name=eval \
    experiment=eval/tofu/default \
    model=${MODEL} \
    task_name=${UNLEARN_TASK_NAME} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${UNLEARN_TASK_NAME} \
    paths.output_dir=saves/unlearn/${UNLEARN_TASK_NAME}/evals \
    retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json

echo "✓ Unlearned model evaluation completed"
echo "Evaluation results saved to: saves/unlearn/${UNLEARN_TASK_NAME}/evals/TOFU_EVAL.json"
echo ""

########################################################################################################################
########################################### Summary ####################################################################
########################################################################################################################

echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Original model evaluation: saves/eval/${ORIGINAL_TASK_NAME}/TOFU_EVAL.json"
echo "✓ Fine-tuned model (full): saves/finetune/${FULL_TASK_NAME}"
echo "✓ Retain model: saves/finetune/${RETAIN_TASK_NAME}"
echo "✓ Retain evaluation: saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json"
echo "✓ Unlearned model: saves/unlearn/${UNLEARN_TASK_NAME}"
echo "✓ Unlearned evaluation: saves/unlearn/${UNLEARN_TASK_NAME}/evals/TOFU_EVAL.json"
echo "=========================================="
echo ""
echo "Pipeline completed successfully! 🎉"

