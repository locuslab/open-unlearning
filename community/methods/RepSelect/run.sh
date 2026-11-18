#!/bin/bash
# RepSelect on TOFU (Llama-3.2-1B-Instruct): unlearn, then evaluate.
#
# strength is the size of RepSelect's single update step. Several values are run, to
# trace the forgetting / utility trade-off.

model="Llama-3.2-1B-Instruct"
forget_retain_splits=(
    "forget10 retain90"
)
strengths=(0.05 0.1 0.2)

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for strength in "${strengths[@]}"; do
        task_name=tofu_${model}_${forget_split}_RepSelect_strength${strength}
        model_path=open-unlearning/tofu_${model}_full

        # Unlearn
        CUDA_VISIBLE_DEVICES=0 \
        python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=RepSelect \
        task_name=${task_name} \
        model=${model} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        trainer.method_args.strength=${strength} \
        trainer.method_args.n_pcs=512 \
        trainer.args.do_eval=False \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=False

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
