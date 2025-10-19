#!/bin/bash

# LoRA Integration Experiments for Open-Unlearning
# This script demonstrates how to run fine-tuning and unlearning experiments with LoRA

set -e

echo "🚀 Starting LoRA Integration Experiments"
echo "========================================"

# Set default values
MODEL="Qwen2.5-3B-Instruct-lora"
EXPERIMENT_TYPE="finetune"
DATASET="tofu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--model MODEL] [--type TYPE] [--dataset DATASET]"
            echo ""
            echo "Options:"
            echo "  --model MODEL     LoRA model to use (default: Qwen2.5-3B-Instruct-lora)"
            echo "                    Available: Qwen2.5-3B-Instruct-lora, Llama-2-7b-hf-lora, Llama-2-7b-chat-hf-lora"
            echo "  --type TYPE       Experiment type: finetune or unlearn (default: finetune)"
            echo "  --dataset DATASET Dataset to use: tofu, muse, or wmdp (default: tofu)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Fine-tune Qwen2.5-3B-Instruct with LoRA on TOFU"
            echo "  $0 --type unlearn                      # Unlearn with Qwen2.5-3B-Instruct LoRA on TOFU"
            echo "  $0 --model Llama-2-7b-hf-lora         # Fine-tune Llama-2-7b-hf with LoRA"
            echo "  $0 --dataset muse --type unlearn      # Unlearn with Qwen2.5-3B-Instruct LoRA on MUSE"
            echo "  $0 --dataset wmdp --type unlearn      # Unlearn with Qwen2.5-3B-Instruct LoRA on WMDP"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Type: $EXPERIMENT_TYPE"
echo "  Dataset: $DATASET"
echo ""

# Validate inputs
if [[ "$EXPERIMENT_TYPE" != "finetune" && "$EXPERIMENT_TYPE" != "unlearn" ]]; then
    echo "❌ Error: Experiment type must be 'finetune' or 'unlearn'"
    exit 1
fi

if [[ "$DATASET" != "tofu" && "$DATASET" != "muse" && "$DATASET" != "wmdp" ]]; then
    echo "❌ Error: Dataset must be 'tofu', 'muse', or 'wmdp'"
    exit 1
fi

# Check if model configuration exists
MODEL_CONFIG="configs/model/${MODEL}.yaml"
if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "❌ Error: Model configuration not found: $MODEL_CONFIG"
    echo "Available LoRA models:"
    ls configs/model/*-lora.yaml 2>/dev/null | sed 's/configs\/model\///g' | sed 's/\.yaml//g' | sed 's/^/  - /'
    exit 1
fi

# Check if experiment configuration exists
if [[ "$DATASET" == "wmdp" && "$EXPERIMENT_TYPE" == "finetune" ]]; then
    echo "❌ Error: WMDP dataset only supports unlearning, not fine-tuning"
    echo "Use --type unlearn for WMDP dataset"
    exit 1
fi

EXPERIMENT_CONFIG="configs/experiment/${EXPERIMENT_TYPE}/${DATASET}/lora.yaml"
if [[ ! -f "$EXPERIMENT_CONFIG" ]]; then
    echo "❌ Error: Experiment configuration not found: $EXPERIMENT_CONFIG"
    echo "Available experiment configurations:"
    find configs/experiment -name "lora.yaml" | sed 's/^/  - /'
    exit 1
fi

echo "✅ All configurations found"
echo ""

# Set up experiment command
if [[ "$EXPERIMENT_TYPE" == "finetune" ]]; then
    TRAIN_CONFIG="train"
else
    TRAIN_CONFIG="unlearn"
fi

# Build the command
CMD="python src/train.py --config-name=${TRAIN_CONFIG} @experiment=${EXPERIMENT_TYPE}/${DATASET}/lora model=${MODEL}"

echo "Running command:"
echo "  $CMD"
echo ""

# Run the experiment
echo "🏃 Starting experiment..."
eval $CMD

echo ""
echo "✅ Experiment completed!"
echo ""
echo "Results should be saved in the output directory."
echo "Check the logs for detailed information about the training process."
