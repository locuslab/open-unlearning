#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
)
cls=(
    "none"
    "superloss"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)



per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer in "${trainers[@]}"; do
            for cl in "${cls[@]}"; do
            
                task_name=tofu_${model}_${forget_split}_${trainer}
                model_path=open-unlearning/tofu_${model}_full

                if [ ! -f saves/unlearn/"${task_name}"/evals/TOFU_SUMMARY.json ]; then
                    if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                        echo "${task_name}" "Model Not Found"
                        
                        # Unlearn
                        CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                        src/train.py --config-name=unlearn.yaml \
                        experiment=unlearn/tofu/default.yaml \
                        trainer.cl=${cl} \
                        trainer=${trainer} \
                        task_name=${task_name} \
                        model=${model} \
                        forget_split=${forget_split} \
                        retain_split=${retain_split} \
                        model.model_args.pretrained_model_name_or_path=${model_path} \
                        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
                        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                        trainer.args.ddp_find_unused_parameters=true \
                        trainer.args.gradient_checkpointing=true trainer.args.eval_strategy=no
                    fi
                fi

                if [ ! -f saves/unlearn/"${task_name}"/evals/TOFU_SUMMARY.json ]; then
                    echo "${task_name}" "Eval Not Found"
                    if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] || [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                        # Eval  
                        CUDA_VISIBLE_DEVICES=4 python src/eval.py \
                        experiment=eval/tofu/default.yaml \
                        forget_split=${forget_split} \
                        holdout_split=${holdout_split} \
                        model=${model} \
                        task_name=${task_name} \
                        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                        paths.output_dir=saves/unlearn/${task_name}/evals \
                        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
                    fi
                fi
            done
        done
    done
done