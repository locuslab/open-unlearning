#!/bin/bash
# =============================================================================
# Demo 2: 评测一个模型 (TOFU benchmark)
# -----------------------------------------------------------------------------
# 入口: src/eval.py  (mode=eval)
#   1) get_model()      根据 model 配置加载模型+tokenizer
#   2) get_evaluators() 根据 eval=tofu 构造 TOFUEvaluator
#   3) evaluator.evaluate() 逐个跑 configs/eval/tofu.yaml 里 default 列出的指标
#
# 结果文件 (写入 paths.output_dir = saves/eval/<task_name>):
#   TOFU_EVAL.json     每条样本的细粒度分数 (value_by_index)
#   TOFU_SUMMARY.json  每个指标的聚合值 (agg_value)
#
# retain_logs_path: 指向 "retain 参照模型" 的 EVAL.json。
#   forget_quality / privleak 等指标需要它来和参照模型做对比。
#   需要先 `python setup_data.py --eval` 下载官方参照日志，或自己评一个 retain 模型。
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# 共享集群必看: 只暴露一张【空闲】GPU 评测。先 `nvidia-smi` 选空闲卡,
# 或运行时 `CUDA_VISIBLE_DEVICES=3 bash demos/2_eval.sh` 覆盖。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=Llama-3.2-1B-Instruct

python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${MODEL}_full \
  forget_split=forget10 \
  holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json \
  task_name=demo_eval

# 想评测 Demo 1 自己微调出来的模型, 把上面这行换成本地路径:
#   model.model_args.pretrained_model_name_or_path=saves/finetune/demo_finetune_full
#
# 没有 retain 参照日志时, 去掉 retain_logs_path 即可 (forget_quality 会是 None)。
