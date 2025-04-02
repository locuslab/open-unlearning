#!/bin/bash


models=(
    "Llama-3.2-1B-Instruct"
)
trainers_experiments=(
    "UNDIAL unlearn/tofu/default.yaml"
)
forget_retain_splits=(
    "forget05 retain95"
    "forget10 retain90"
    "forget01 retain99"
)

per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################


for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            
            task_name=tofu_${model}_${forget_split}_${trainer} 
            model_path=open-unlearning/tofu_${model}_full
            echo ${task_name}: Unlearning ${model_path} using ${trainer}

            # Unlearn
            python src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
        done
    done
done

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


model=Llama-3.2-1B-Instruct

data_splits=(
    "News"
    "Books"
)

trainers=(
    "UNDIAL"
)

for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do

        task_name=muse_${model}_${data_split}_${trainer}

        python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \

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
