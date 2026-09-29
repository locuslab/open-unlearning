#!/bin/bash
# =============================================================================
# Demo 3: 用 LoRA 微调模型
# -----------------------------------------------------------------------------
# 注意: 原框架默认不支持 LoRA。本 demo 配套做了一处最小、非破坏性的扩展:
#   - src/model/__init__.py: get_model() 在加载完基座模型后, 若 model 配置里有
#     `peft_args` 块, 就调用 get_peft_lora_model() 用 peft 包套一层 LoRA adapter。
#   - configs/model/Llama-3.2-1B-Instruct-LoRA.yaml: 在原模型配置上加了 peft_args。
#
# 前置依赖: pip install peft
#
# 训练流程其余部分和 Demo 1 完全一样 (同样走 FinetuneTrainer)，只是把
# model 换成带 peft_args 的 LoRA 配置。HF Trainer 会自动只保存 adapter 权重。
# 输出: saves/finetune/demo_lora_finetune
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# 共享集群必看: 只暴露一张【空闲】GPU, 否则 HF Trainer 会在所有可见卡上启用
# DataParallel, 往被别人占满的卡复制模型而 CUDA OOM。先 `nvidia-smi` 选空闲卡,
# 或运行时 `CUDA_VISIBLE_DEVICES=3 bash demos/3_lora_finetune.sh` 覆盖。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  model=Llama-3.2-1B-Instruct-LoRA \
  task_name=demo_lora_finetune \
  trainer.args.num_train_epochs=5 \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.learning_rate=1e-4    # LoRA 通常用比全参微调更大的学习率

# 也可以完全在命令行里临时指定 LoRA 超参 (无需改 yaml), 例如:
#   model=Llama-3.2-1B-Instruct \
#   +model.peft_args.r=16 +model.peft_args.lora_alpha=32 \
#   +model.peft_args.task_type=CAUSAL_LM \
#   '+model.peft_args.target_modules=[q_proj,v_proj]'
#
# ---- 评测训练好的 LoRA adapter ----
# 把 base 模型路径 + adapter 路径 (peft_args.path) 一起传给 eval:
#   python src/eval.py experiment=eval/tofu/default \
#     model=Llama-3.2-1B-Instruct-LoRA \
#     +model.peft_args.path=saves/finetune/demo_lora_finetune \
#     task_name=demo_lora_eval
