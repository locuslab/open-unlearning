#!/usr/bin/env bash
# K-FORGE on TOFU forget10 (Llama-3.2-1B-Instruct), paper configuration.
#
# Stage 1 computes the one-shot K-FORGE edit (wiener_v2) and saves the edited
# checkpoint under saves/unlearn/tofu_kforge_edit. Stage 2 runs downstream
# preference-based unlearning (NPO by default, SimNPO via DOWNSTREAM_TRAINER)
# initialized from the edited checkpoint. For a one-shot-only run, stop after
# stage 1.
set -euo pipefail

DOWNSTREAM_TRAINER="${DOWNSTREAM_TRAINER:-NPO}"
EDIT_TASK="${EDIT_TASK:-tofu_kforge_edit}"

# ---------------------------------------------------------------------------
# Stage 1: one-shot K-FORGE edit (wiener_v2, rank 2, alpha = 0.45, lambda = 0.01)
# ---------------------------------------------------------------------------
python src/train.py \
  --config-name=unlearn \
  experiment=unlearn/tofu/default \
  trainer=KFORGE \
  trainer.method_args.target_modules_regex='.*mlp\.down_proj$' \
  trainer.method_args.rank=2 \
  trainer.method_args.strength=0.45 \
  trainer.method_args.edit_variant=wiener_v2 \
  trainer.method_args.lambda_tradeoff=0.01 \
  task_name="${EDIT_TASK}"

# ---------------------------------------------------------------------------
# Stage 2: downstream unlearning initialized from the edited checkpoint
# ---------------------------------------------------------------------------
python src/train.py \
  --config-name=unlearn \
  experiment=unlearn/tofu/default \
  trainer="${DOWNSTREAM_TRAINER}" \
  model.model_args.pretrained_model_name_or_path="saves/unlearn/${EDIT_TASK}" \
  task_name="tofu_$(echo "${DOWNSTREAM_TRAINER}" | tr '[:upper:]' '[:lower:]')_from_kforge"
