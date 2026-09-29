#!/bin/bash
# =============================================================================
# Demo 1: 直接（全参数）微调一个模型
# -----------------------------------------------------------------------------
# 入口: src/train.py  (mode=train)
# 通过 Hydra 把以下配置组合在一起:
#   experiment=finetune/tofu/default
#     -> configs/experiment/finetune/tofu/default.yaml
#        - model:   Llama-3.2-1B-Instruct
#        - trainer: finetune  (handler=FinetuneTrainer, configs/trainer/finetune.yaml)
#        - data:    TOFU_QA_full (locuslab/TOFU 的 "full" split)
#        - eval:    tofu  (训练过程中按 epoch 评测)
#
# 训练完的权重会保存到 paths.output_dir，默认 = saves/<mode>/<task_name>
# 即: saves/finetune/demo_finetune_full
# =============================================================================
set -e
cd "$(dirname "$0")/.."   # 切到仓库根目录

# 共享集群必看: 只暴露一张【空闲】GPU, 否则 HF Trainer 会在所有可见卡上启用
# DataParallel, 往被别人占满的卡复制模型而 CUDA OOM。先 `nvidia-smi` 选空闲卡,
# 或运行时 `CUDA_VISIBLE_DEVICES=3 bash demos/1_finetune.sh` 覆盖。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=Llama-3.2-1B-Instruct

python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  model=${MODEL} \
  task_name=demo_finetune_full \
  trainer.args.num_train_epochs=5 \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.learning_rate=1e-5

# 命令行上任意 key 都能覆盖 yaml，比如:
#   trainer.args.num_train_epochs=10
#   data/datasets@data.train=TOFU_QA_retain \
#   data.train.TOFU_QA_retain.args.hf_args.name=retain90   # 改成训练 retain 模型
#
# 多卡训练把 `python src/train.py` 换成:
#   accelerate launch --config_file configs/accelerate/default_config.yaml src/train.py ...


python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN