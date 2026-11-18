#!/bin/bash
# RepSelect and NPO on TOFU (Llama-3.2-1B-Instruct) under a relearning attack, evaluated
# on forget samples that the attacker does not have (as in arxiv.org/abs/2410.08827).
#
# For each run:
#   1. unlearn the whole forget split
#   2. evaluate on the second half of the forget split            (pre-attack)
#   3. attack: fine-tune for one epoch on the first half of the forget split
#   4. evaluate on the second half again                          (post-attack)
#
# forget10 is 20 authors x 20 questions, in order, so the two halves are disjoint
# sets of 10 authors. The attacker never sees the evaluated questions: what comes
# back on them was still in the weights.
#
# The last two runs are references under the same attack, without any unlearning:
# the target model, and the retain model, which never saw the forget split. The
# retain model shows how much the attack teaches on its own.
#
# Results: saves/unlearn/<task>/evals_heldout/TOFU_SUMMARY.json (pre-attack) and
#          saves/finetune/<task>_relearned/evals_heldout/TOFU_SUMMARY.json (post-attack);
#          for the references, the pre-attack one is in saves/finetune/<task>/evals_heldout/

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
relearn_subset='"train[:50%]"'  # the attacker's half of the forget split
heldout_subset='"train[50%:]"'  # the evaluated half
relearn_lr=1e-5  # of the grid 1e-6 ... 3e-4, the one that recovers the most on RepSelect

target_model=open-unlearning/tofu_${model}_full

evaluate() {  # <model path> <output dir>
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    experiment=eval/tofu/default.yaml \
    eval=tofu_forget_subset \
    "eval.tofu.forget_subset=${heldout_subset}" \
    forget_split=${forget_split} \
    model=${model} \
    task_name=heldout_eval \
    model.model_args.pretrained_model_name_or_path=$1 \
    paths.output_dir=$2
}

relearn() {  # <model path> <task name>; saves to saves/finetune/<task name>
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default.yaml \
    task_name=$2 \
    model=${model} \
    model.model_args.pretrained_model_name_or_path=$1 \
    data/datasets@data.train=TOFU_QA_forget \
    data.train.TOFU_QA_forget.args.hf_args.name=${forget_split} \
    "data.train.TOFU_QA_forget.args.hf_args.split=${relearn_subset}" \
    trainer.args.learning_rate=${relearn_lr} \
    trainer.args.num_train_epochs=1 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.warmup_epochs=0 \
    +trainer.args.lr_scheduler_type=constant \
    trainer.args.do_eval=False \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False
}

unlearn_attack_eval() {  # <task name> <trainer overrides ...>
    task_name=$1
    shift

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    task_name=${task_name} \
    model=${model} \
    forget_split=${forget_split} \
    retain_split=${retain_split} \
    model.model_args.pretrained_model_name_or_path=${target_model} \
    trainer.args.do_eval=False \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    "$@"

    evaluate saves/unlearn/${task_name} saves/unlearn/${task_name}/evals_heldout
    relearn saves/unlearn/${task_name} ${task_name}_relearned
    evaluate saves/finetune/${task_name}_relearned saves/finetune/${task_name}_relearned/evals_heldout
}

for strength in 0.05 0.1 0.2; do
    unlearn_attack_eval tofu_${model}_${forget_split}_RepSelect_strength${strength} \
        trainer=RepSelect \
        trainer.method_args.strength=${strength} \
        trainer.method_args.n_pcs=512
done

# NPO with the repository's default hyperparameters, for 250 steps
# (batch 4 x 8 accumulation steps: the default effective batch of 32, fits in 16 GB)
unlearn_attack_eval tofu_${model}_${forget_split}_NPO \
    trainer=NPO \
    +trainer.args.max_steps=250 \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=8

# references under the same attack, without any unlearning:
# the target model, and the retain model, which never saw the forget split
for reference in full ${retain_split}; do
    task_name=tofu_${model}_${reference}_reference
    reference_model=open-unlearning/tofu_${model}_${reference}
    evaluate ${reference_model} saves/finetune/${task_name}/evals_heldout
    relearn ${reference_model} ${task_name}_relearned
    evaluate saves/finetune/${task_name}_relearned saves/finetune/${task_name}_relearned/evals_heldout
done
