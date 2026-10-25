#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

per_device_train_batch_size=4
gradient_accumulation_steps=8
lrs=(1e-5)

########################################################################################################################
################################ TOFU: no popularity column (use_popularity=false) #####################################
########################################################################################################################
# Every fact gets the default popularity (100, i.e. beta ~= 1.5).

model=Llama-3.2-1B-Instruct
forget_retain_splits=(
    "forget10 retain90"
    "forget05 retain95"
    "forget01 retain99"
)

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for lr in "${lrs[@]}"; do
        task_name=tofu_${model}_${forget_split}_AdaPop_nopop_lr${lr}
        model_path=open-unlearning/tofu_${model}_full
        echo ${task_name}: Unlearning ${model_path} using AdaPop

        # Unlearn
        CUDA_VISIBLE_DEVICES=0 \
        python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=AdaPop \
        task_name=${task_name} \
        model=${model} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        trainer.method_args.use_popularity=false \
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=False \
        trainer.args.learning_rate=$lr

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

########################################################################################################################
################################ DUET: Wikidata popularity column (use_popularity=true) ################################
########################################################################################################################
# Needs the DUET benchmark configs (configs/experiment/unlearn/duet, configs/eval/duet.yaml).
# The forget data's `pop_sum` column (see wikidata_popularity.py) sets the per-fact beta.

model=Llama-3.1-8B-Instruct
subfolder=llama-3.1-8b-instruct-tripunlamb-ft # in SwetieePawsss/DUET_ft_models
forget_retain_splits=(
    "city_forget_rare_5 city_fast_retain_500"
    "city_forget_popular_5 city_fast_retain_500"
)

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for lr in "${lrs[@]}"; do
        task_name=duet_${model}_${forget_split}_AdaPop_lr${lr}
        echo ${task_name}: Unlearning ${subfolder} using AdaPop

        # Unlearn
        CUDA_VISIBLE_DEVICES=0 \
        python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/duet/default.yaml \
        trainer=AdaPop \
        task_name=${task_name} \
        model=${model} \
        model_subfolder=${subfolder} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        trainer.method_args.use_popularity=true \
        +data.forget.DUET_QA_forget.args.popularity_key=pop_sum \
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
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        paths.output_dir=saves/unlearn/${task_name}/evals
    done
done
