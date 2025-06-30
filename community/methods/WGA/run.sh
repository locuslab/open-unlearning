#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################

models=(
    "Llama-3.2-1B-Instruct"
)
trainers_experiments=(
    "WGA unlearn/tofu/default.yaml"
)
forget_retain_splits=(
    "forget10 retain90"
    "forget05 retain95"
    "forget01 retain99"
)

per_device_train_batch_size=16
gradient_accumulation_steps=2


lrs=(1e-5)
alphas=(1.0, 0.1, 10.0)
betas= (1.0)

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            for lr in "${lrs[@]}"; do
                for beta1 in "${betas[@]}"; do 
                    for alpha in "${alphas[@]}"; do          
                        task_name=tofu_${model}_${forget_split}_${trainer}_lr${lr}_beta${beta1}_alpha${alpha}
                        model_path=open-unlearning/tofu_${model}_full
                        echo ${task_name}: Unlearning ${model_path} using ${trainer}

                        # Unlearn
                        CUDA_VISIBLE_DEVICES=0 \
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
                        trainer.args.eval_strategy=no \
                        trainer.args.eval_on_start=False \
                        trainer.args.learning_rate=$lr \
                        trainer.method_args.beta1=$beta1 \
                        trainer.method_args.beta2=$beta2 \
                        trainer.method_args.alpha=$alpha

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
        done
    done
done
