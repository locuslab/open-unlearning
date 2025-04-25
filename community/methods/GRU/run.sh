#!/bin/bash

# GRU with GradAscent
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  trainer=GRU \
  task_name=gru_ga_forget10 \
  trainer.method_args.forget_loss_type=GradAscent \
  trainer.args.gradient_accumulation_steps=16 \
  trainer.args.per_device_train_batch_size=4

# Evaluation for GRU with GradAscent
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=forget10 \
  model=Llama-3.2-1B-Instruct \
  task_name=gru_ga_forget10 \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/gru_ga_forget10 \
  paths.output_dir=saves/unlearn/gru_ga_forget10/evals \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json

# GRU with NPO
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  trainer=GRU \
  task_name=gru_npo_forget10 \
  trainer.method_args.forget_loss_type=NPO \
  trainer.args.gradient_accumulation_steps=16 \
  trainer.args.per_device_train_batch_size=4

# Evaluation for GRU with NPO
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=forget10 \
  model=Llama-3.2-1B-Instruct \
  task_name=gru_npo_forget10 \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/gru_npo_forget10 \
  paths.output_dir=saves/unlearn/gru_npo_forget10/evals \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
