#!/usr/bin/env bash

DATE=$(date "+%m%d")
TIME=$(date "+%H%M%S")


export CUDA_VISIBLE_DEVICES=0

REPORTTO="wandb"
WANDB_PROJECT="BalDRO"

MODEL="Llama-2-7b-hf"
TRAINER="DrSimNPO"
splits=(
    "News"
    "Books"
)

# lr, batchsize, grad_acc, epochs
lr_set=("1e-5" "2e-5" "5e-5")
bz_set=("4 8")
beta_set=(3.5 4.5)
delta_set=(0 1)
gamma_set=(0.125 0.25)
epoch_set=(10)
beta_dv_set=(0.5 1.0 2.0 5.0 10.0)


for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for epochs in "${epoch_set[@]}"; do
                for beta in "${beta_set[@]}"; do
                    for delta in "${delta_set[@]}"; do
                        for gamma in "${gamma_set[@]}"; do
                            for beta_dv_forget in "${beta_dv_set[@]}"; do
                                # Args ========================================
                                PRETRAINED_PATH="muse-bench/MUSE-${split}_target"
                                TOKENIZER_PRETRAINED="meta-llama/Llama-2-7b-chat-hf"

                                bsz=$(echo $bz | cut -d' ' -f1)
                                grad_acc=$(echo $bz | cut -d' ' -f2)

                                # learning_rate, batchsize, grad_acc, epochs
                                SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_a1.0_b${beta}_d${delta}_g${gamma}_e${epochs}_betaDV${beta_dv_forget}_day${DATE}_time${TIME}"
                                TASK_NAME="unlearn_muse_${split}_${MODEL}_${TRAINER}_${SUFFIX}"
                                OUTPUT_DIR="./saves/unlearn/muse/${split}/${MODEL}/${TRAINER}/${SUFFIX}"

                                # TRAIN COMMAND =================================
                                export WANDB_PROJECT=${WANDB_PROJECT}
                                python src/train.py --config-name=unlearn.yaml \
                                    experiment=unlearn/muse/default.yaml  \
                                    trainer=BalDRO/${TRAINER} \
                                    model=${MODEL} \
                                    model.model_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
                                    model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PRETRAINED} \
                                    data_split=${split} \
                                    task_name=${TASK_NAME} \
                                    paths.output_dir="${OUTPUT_DIR}" \
                                    eval.muse.retain_logs_path=./saves/eval/muse_${MODEL}_${split}_retrain/MUSE_EVAL.json \
                                    trainer.args.ddp_find_unused_parameters=true \
                                    trainer.args.gradient_checkpointing=true \
                                    trainer.args.report_to=${REPORTTO} \
                                    +trainer.args.run_name=${TASK_NAME} \
                                    trainer.args.logging_steps=1 \
                                    trainer.args.learning_rate=$lr \
                                    trainer.args.per_device_train_batch_size=$bsz \
                                    trainer.args.gradient_accumulation_steps=$grad_acc \
                                    trainer.args.num_train_epochs=$epochs \
                                    trainer.args.eval_strategy=epoch \
                                    trainer.args.eval_on_start=False \
                                    trainer.method_args.gamma=$gamma \
                                    trainer.method_args.alpha=1.0 \
                                    trainer.method_args.retain_dro=False \
                                    trainer.method_args.retain_loss_type=NLL \
                                    trainer.method_args.beta=$beta \
                                    trainer.method_args.delta=$delta \
                                    trainer.method_args.beta_dv_forget=${beta_dv_forget}
                            done
                        done
                    done
                done
            done
        done
    done
done
