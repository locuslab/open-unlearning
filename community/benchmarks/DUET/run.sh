#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

########################################################################################################################
########################################### Unlearn DUET models ########################################################
########################################################################################################################

# "<model config> <subfolder in SwetieePawsss/DUET_ft_models>"
models=(
    "Llama-3.1-8B-Instruct llama-3.1-8b-instruct-tripunlamb-ft"
    "Qwen2.5-7B-Instruct qwen2.5-7b-instruct-tripunlamb-ft"
    "gemma-7b-it gemma-7b-it-tripunlamb-ft"
)
trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "WGA"
)
forget_retain_splits=(
    "city_forget_rare_5 city_fast_retain_500"
    "city_forget_popular_5 city_fast_retain_500"
)

per_device_train_batch_size=4
gradient_accumulation_steps=8
lr=1e-5

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model_entry in "${models[@]}"; do
        model=$(echo $model_entry | cut -d' ' -f1)
        subfolder=$(echo $model_entry | cut -d' ' -f2)
        # gemma's chat template has no system role, so drop the DUET system prompt for it
        template_args=()
        if [[ $model == gemma* ]]; then
            template_args+=("model.template_args.system_prompt=null")
        fi
        for trainer in "${trainers[@]}"; do
            task_name=duet_${model}_${forget_split}_${trainer}_lr${lr}
            echo ${task_name}: Unlearning ${subfolder} using ${trainer}

            # Unlearn
            CUDA_VISIBLE_DEVICES=0 \
            python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/duet/default.yaml \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            model_subfolder=${subfolder} \
            "${template_args[@]}" \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.eval_strategy=no \
            trainer.args.eval_on_start=False \
            trainer.args.learning_rate=$lr

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/duet/default.yaml \
            model=${model} \
            model_subfolder="" \
            "${template_args[@]}" \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals
        done
    done
done
