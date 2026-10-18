#!/bin/bash
# =============================================================================
# Demo 4: 用一个【已有的】遗忘方法做 unlearning (这里用 GradDiff)
# -----------------------------------------------------------------------------
# 入口: src/train.py  (mode=unlearn)
# experiment=unlearn/tofu/default -> configs/experiment/unlearn/tofu/default.yaml
#   - model:   待遗忘的目标模型 (默认 open-unlearning/tofu_Llama-3.2-1B-Instruct_full)
#   - data:    unlearn -> 同时加载 forget 和 retain 两个数据集
#              collator 把每个 batch 组织成 {"forget": {...}, "retain": {...}}
#   - trainer: 由命令行 trainer=GradDiff 指定 (handler=GradDiff)
#   - eval:    tofu (遗忘过程中/结束后评测)
#
# 遗忘方法的核心只有一个函数: trainer 的 compute_loss(model, inputs)。
#   - GradAscent: loss = -forget_loss
#   - GradDiff:   loss = gamma*(-forget_loss) + alpha*retain_loss
#   - NPO/SimNPO/DPO/RMU/...: 各自不同的 compute_loss
#
# 输出: saves/unlearn/demo_unlearn_graddiff
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# 共享集群必看: 只暴露一张【空闲】GPU, 否则 HF Trainer 会在所有可见卡上启用
# DataParallel, 往被别人占满的卡复制模型而 CUDA OOM。先 `nvidia-smi` 选一张空闲卡,
# 改下面的默认 0, 或运行时 `CUDA_VISIBLE_DEVICES=3 bash demos/4_unlearn.sh` 覆盖。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=Llama-3.2-1B-Instruct

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model=${MODEL} \
  trainer=GradDiff \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.alpha=1.0 \
  trainer.method_args.retain_loss_type=NLL \
  forget_split=forget10 \
  retain_split=retain90 \
  holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json \
  task_name=demo_unlearn_graddiff

# 换方法只需改 trainer= : GradAscent / NPO / SimNPO / DPO / RMU / UNDIAL / WGA / CEU ...
# 对应方法的额外超参在 trainer.method_args.* 下覆盖, 例如 NPO:
#   trainer=NPO trainer.method_args.beta=0.1 trainer.method_args.gamma=1.0
