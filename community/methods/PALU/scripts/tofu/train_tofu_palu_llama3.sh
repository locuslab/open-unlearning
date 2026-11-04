#!/usr/bin/env bash
set -euo pipefail

DATE=$(date "+%m%d")
TIME=$(date "+%H%M%S")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="Llama-3.1-8B-Instruct"

REPORTTO="${REPORTTO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-palu}"
KL_DIRECTION="${KL_DIRECTION:-reverse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.cache/palu/runs}"
RETAIN_LOGS_ROOT="${RETAIN_LOGS_ROOT:-.cache/palu/retain_logs}"

TRAINER="PALU"
PRETRAINED_PATH="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"

splits=(
    "forget05 holdout05 retain95"
    "forget01 holdout01 retain99"
    "forget10 holdout10 retain90"
)
# lr, batchsize, grad_acc, epochs
lr_set=("1e-5" "2e-5" "3e-5" "4e-5" "5e-5")
bz_set=("8 4")
target_mode_set=("mean")
alpha_set=(1 2 5 0.5 0.2)
topk_set=(1 1000 5000 10000)
first_n_set=(1 2 3 4 5 10)
epoch_set=(10)

for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for target_mode in "${target_mode_set[@]}"; do
                for epochs in "${epoch_set[@]}"; do
                    for alpha in "${alpha_set[@]}"; do
                        for topk in "${topk_set[@]}"; do
                            for first_n in "${first_n_set[@]}"; do
                                # Args ========================================
                                read -r forget_split holdout_split retain_split <<< "$split"

                                read -r bsz grad_acc <<< "$bz"

                                # learning_rate, batchsize, grad_acc, epochs
                                SUFFIX="kl${KL_DIRECTION}_target_mode${target_mode}_first_n${first_n}_lr${lr}_b${bsz}_ga${grad_acc}_a${alpha}_topk${topk}_e${epochs}_day${DATE}_time${TIME}"
                                TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
                                OUTPUT_DIR="${OUTPUT_ROOT}/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

                                # TRAIN COMMAND =================================
                                export WANDB_PROJECT
                                python src/train.py --config-name=unlearn.yaml \
                                    experiment=unlearn/tofu/palu \
                                    trainer="${TRAINER}" \
                                    model="${MODEL}" \
                                    model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                                    model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                                    forget_split="${forget_split}" \
                                    holdout_split="${holdout_split}" \
                                    retain_split="${retain_split}" \
                                    task_name="${TASK_NAME}" \
                                    paths.output_dir="${OUTPUT_DIR}" \
                                    eval.tofu.retain_logs_path="${RETAIN_LOGS_ROOT}/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json" \
                                    trainer.args.ddp_find_unused_parameters=true \
                                    trainer.args.gradient_checkpointing=true \
                                    trainer.args.report_to="${REPORTTO}" \
                                    +trainer.args.run_name="${TASK_NAME}" \
                                    trainer.args.logging_steps=1 \
                                    trainer.args.learning_rate="$lr" \
                                    trainer.args.per_device_train_batch_size="$bsz" \
                                    trainer.args.gradient_accumulation_steps="$grad_acc" \
                                    trainer.args.num_train_epochs="$epochs" \
                                    trainer.args.eval_strategy=epoch \
                                    trainer.args.eval_on_start=True \
                                    trainer.method_args.gamma=1.0 \
                                    trainer.method_args.alpha="$alpha" \
                                    trainer.method_args.retain_loss_type=NLL \
                                    trainer.method_args.top_k="$topk" \
                                    trainer.method_args.target_mode="${target_mode}" \
                                    trainer.method_args.first_n="${first_n}" \
                                    trainer.method_args.kl_direction="${KL_DIRECTION}" \
                                    "$@"
                            done
                        done
                    done
                done
            done
        done
    done
done
