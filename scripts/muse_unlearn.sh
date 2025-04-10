#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=4
gradient_accumulation_steps=8


model=Llama-2-7b-hf

data_splits=(
    "News"
    "Books"
)

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
)

cls=(
    "default"
    "superloss"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do
        for cl in "${cls[@]}"; do

            task_name=muse_${model}_${data_split}_${trainer}_${cl}

            if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                echo "${task_name}" "Model Not Found"

                CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/muse/${cl}.yaml \
                model=${model} \
                data_split=${data_split} \
                trainer=${trainer} \
                task_name=${task_name} \
                retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true
            fi

            if [ ! -f saves/unlearn/"${task_name}"/evals/MUSE_SUMMARY.json ]; then
                echo "${task_name}" "Eval Not Found"

                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split=${data_split} \ 
                task_name=${task_name} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                paths.output_dir=saves/unlearn/${trainer}/evals \
                retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
            fi
        done
    done
done



# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do
        for scal in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
            task_name=muse_${model}_${data_split}_${trainer}_scal_${scal} \
            
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/scalability.yaml \
            model=${model} \
            data_split=${data_split} \
            forget_split=${scal} \
            trainer=${trainer} \
            task_name=${task_name} \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true

            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${data_split} \ 
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${trainer}/evals \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        done
    done
done



#########################################################
########### MUSE News Unlearning sustainability #########
#########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do
        model_path=muse-bench/MUSE-${data_split}_target
        for sust in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
            task_name=muse_${model}_${data_split}_${trainer}_sust_${sust}

            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/sustainabilty.yaml \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            data_split=${data_split} \
            trainer=${trainer} \
            task_name=${task_name} \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true

            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split=${data_split} \ 
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${trainer}/evals \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

            model_path=saves/unlearn/${task_name}
        done
    done
done