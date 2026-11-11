#!/usr/bin/env bash
set -euo pipefail

# Run one example from the Open-Unlearning repository root; append Hydra overrides.
MODEL="${MODEL:-Llama-2-7b-chat-hf}"
PRETRAINED_PATH="${PRETRAINED_PATH:-open-unlearning/tofu_${MODEL}_full}"
KL_DIRECTION="${KL_DIRECTION:-reverse}"
TASK_NAME="palu_${MODEL}_forget01_${KL_DIRECTION}"

python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=PALU \
    data/datasets@data.forget=TOFU_QA_forget_palu \
    '~data.forget.TOFU_QA_forget' \
    model="${MODEL}" \
    model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
    forget_split=forget01 \
    retain_split=retain99 \
    holdout_split=holdout01 \
    task_name="${TASK_NAME}" \
    paths.output_dir=".cache/palu/runs/${TASK_NAME}" \
    trainer.method_args.first_n=3 \
    trainer.method_args.top_k=1000 \
    trainer.method_args.kl_direction="${KL_DIRECTION}" \
    trainer.args.report_to=none \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=false \
    trainer.args.do_eval=false \
    '~eval.tofu' \
    "$@"
