#!/bin/bash
# =============================================================================
# Demo 5: 运行【自定义】遗忘算法 BoundedGradDiff
# -----------------------------------------------------------------------------
# 把自己的算法接入本项目, 只需 3 步 (本 demo 已帮你做好):
#   1. 写 trainer 类:  src/trainer/unlearn/bounded_grad_diff.py
#                      - 继承 GradDiff (或更底层的 UnlearnTrainer)
#                      - 只重写 compute_loss(model, inputs, ...)
#   2. 注册:           src/trainer/__init__.py 里 _register_trainer(BoundedGradDiff)
#   3. 写配置:         configs/trainer/BoundedGradDiff.yaml (handler + method_args)
#
# 之后用法和内置方法完全一样, 只是 trainer=BoundedGradDiff:
# 输出: saves/unlearn/demo_custom_unlearn
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# 共享集群必看: 只暴露一张【空闲】GPU, 否则 HF Trainer 会在所有可见卡上启用
# DataParallel, 往被别人占满的卡复制模型而 CUDA OOM。先 `nvidia-smi` 选一张空闲卡,
# 改下面的默认 0, 或运行时 `CUDA_VISIBLE_DEVICES=3 bash demos/5_custom_unlearn.sh` 覆盖。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=Llama-3.2-1B-Instruct

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model=${MODEL} \
  trainer=BoundedGradDiff \
  trainer.method_args.forget_loss_bound=4.0 \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.alpha=1.0 \
  forget_split=forget10 \
  retain_split=retain90 \
  holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json \
  task_name=demo_custom_unlearn

# 调用链 (谁调用了你的 compute_loss):
#   src/train.py -> load_trainer() 按 handler 从 TRAINER_REGISTRY 取出 BoundedGradDiff,
#   用 trainer.args(=TrainingArguments) 和 **method_args 实例化 -> trainer.train()
#   -> HuggingFace Trainer 训练循环每个 step 调用 compute_loss(model, inputs)。
