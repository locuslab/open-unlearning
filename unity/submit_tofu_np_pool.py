import os
import argparse

from sbatch import (
    create_submit_sbatch,
    add_argument_to_cmd
)


def add_hparams_to_cmd(cmd, hparams):
    for k, v in hparams.items():
        argument = f"{k}={v}"
        cmd = add_argument_to_cmd(cmd=cmd, argument=argument)
    return cmd

def train_tofu(train_hparams, eval_hparams, task_name, output_dir, model="Llama-3.2-1B-Instruct", dataset="TOFU_QA_retain", forget_split="forget10", retain_split="retain90", holdout_split="holdout10"):
    output_dir = os.path.join(output_dir, task_name)
    error_file = os.path.join(output_dir,"unity","err.txt")
    logs_file = os.path.join(output_dir,"unity","logs.txt")
    
    train_command = f"""CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=finetune/tofu/default.yaml \
    task_name={task_name} \
    model={model} \
    data/datasets@data.train={dataset} \
    data.train.{dataset}.args.hf_args.name={retain_split} \
    paths.output_dir={output_dir}
    """
    
    train_command = add_hparams_to_cmd(train_command, train_hparams)

    eval_command = f"""model_dirs=($(find {output_dir} -type f -name "model.safetensors" -exec dirname {{}} \; | sort -u))
    
echo "Checkpoints found in: ${{model_dirs[@]}}"

for dir in "${{model_dirs[@]}}"; do
    echo "Evaluating : $dir"
    CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split={forget_split} \
    holdout_split={holdout_split} \
    task_name={task_name} \
    model={model} \
    model.model_args.pretrained_model_name_or_path=$dir \
    eval.tofu.question_key="question" \
    paths.output_dir=$dir/eval_Q
    
    CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split={forget_split} \
    holdout_split={holdout_split} \
    task_name={task_name} \
    model={model} \
    model.model_args.pretrained_model_name_or_path=$dir \
    eval.tofu.question_key="paraphrased_question" \
    paths.output_dir=$dir/eval_Q_PARA
done
"""

    command = f"""{train_command}
    
{eval_command}"""

    constraint = "a100"
    time = "3:00:0"
    template_params_list = []
    template_params = {
            "job_name": task_name,
            "constraint": constraint,
            "error_file": error_file,
            "logs_file": logs_file,
            "num_gpu": 1,
            "num_cpu": 1,
            "time": time,
            "command": command,
        }
    template_params_list.append(template_params)
    return template_params_list

def main(args):
    print_only = args.print_only
    template_params_list = []
    
    if args.exp=="np_neg":
        # output_dir = "saves/pool/neg"
        # dataset = "TOFU_QA_retain"
        # model = "Llama-3.2-1B-Instruct"
        # forget_split="forget10"
        # retain_split="retain90"
        # holdout_split="holdout10"
        # num_train_epochs=10
        
        # lrs = [1e-5, 5e-5, 1e-4]
        # for lr in lrs:
        #     train_hparams = {
        #         "trainer.args.learning_rate": lr,
        #         "trainer.args.num_train_epochs": num_train_epochs,
        #         "trainer.args.save_strategy": "steps",
        #         "+trainer.args.save_steps": 0.5,
        #         "trainer.args.eval_strategy": "steps",
        #         "+trainer.args.eval_steps":1/num_train_epochs
        #     }
        #     eval_hparams = {}
        #     task_name = f"{model}_{retain_split}_lr{lr}"
        #     cmd_params_list = train_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, dataset=dataset, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
        #     template_params_list += cmd_params_list
        
        output_dir = "saves/pool/neg"
        dataset = "TOFU_QA_full_para"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90_forget10_pert"
        holdout_split="holdout10"
        num_train_epochs=10
        
        # lrs = [1e-5, 5e-5, 1e-4]
        lrs = [5e-5]
        for lr in lrs:
            train_hparams = {
                "trainer.args.learning_rate": lr,
                "trainer.args.num_train_epochs": num_train_epochs,
                "trainer.args.save_strategy": "steps",
                "+trainer.args.save_steps": 0.5,
                "trainer.args.eval_strategy": "steps",
                "+trainer.args.eval_steps":1/num_train_epochs,
                f"data.train.{dataset}.args.answer_key": "answer"
            }
            eval_hparams = {}
            task_name = f"{model}_{retain_split}_lr{lr}"
            cmd_params_list = train_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, dataset=dataset, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
            template_params_list += cmd_params_list
    
    elif args.exp=="np_pos":
        
        # output_dir = "saves/pool/pos"
        # dataset = "TOFU_QA_full"
        # model = "Llama-3.2-1B-Instruct"
        # forget_split="forget10"
        # retain_split="full"
        # holdout_split="holdout10"
        # num_train_epochs=10
        
        # lrs = [1e-5, 5e-5, 1e-4]
        # for lr in lrs:
        #     train_hparams = {
        #         "trainer.args.learning_rate": lr,
        #         "trainer.args.num_train_epochs": num_train_epochs,
        #         "trainer.args.save_strategy": "steps",
        #         "+trainer.args.save_steps": 0.5,
        #         "trainer.args.eval_strategy": "steps",
        #         "+trainer.args.eval_steps":1/num_train_epochs
        #     }
        #     eval_hparams = {}
        #     task_name = f"{model}_{retain_split}_lr{lr}"
        #     cmd_params_list = train_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, dataset=dataset, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
        #     template_params_list += cmd_params_list
            
        
        output_dir = "saves/pool/pos"
        dataset = "TOFU_QA_full_para"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="full_paraphrased"
        holdout_split="holdout10"
        num_train_epochs=10
        
        # lrs = [1e-5, 5e-5, 1e-4]
        lrs = [1e-4]
        for lr in lrs:
            train_hparams = {
                "trainer.args.learning_rate": lr,
                "trainer.args.num_train_epochs": num_train_epochs,
                "trainer.args.save_strategy": "steps",
                "+trainer.args.save_steps": 0.5,
                "trainer.args.eval_strategy": "steps",
                "+trainer.args.eval_steps":1/num_train_epochs,
                f"data.train.{dataset}.args.answer_key": "paraphrased_answer"
            }
            eval_hparams = {}
            task_name = f"{model}_{retain_split}_lr{lr}"
            cmd_params_list = train_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, dataset=dataset, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
            template_params_list += cmd_params_list
    
    
    create_submit_sbatch(template_params_list, print_only=print_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to submit sbatch jobs with optional print-only mode.")
    parser.add_argument(
        "--print_only", 
        action="store_true", 
        help="If set, will only print the sbatch parameters without submitting."
    )
    parser.add_argument(
        "--exp", 
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args)