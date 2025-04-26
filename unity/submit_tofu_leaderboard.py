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


def unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, experiment="unlearn/tofu/default.yaml", model="Llama-3.2-1B-Instruct", forget_split="forget10", retain_split="retain90", holdout_split="holdout10"):
    output_dir = os.path.join(output_dir, task_name)
    error_file = os.path.join(output_dir,"unity","err.txt")
    logs_file = os.path.join(output_dir,"unity","logs.txt")
    
    retain_logs_path = f"saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json"
    reference_model = f"open-unlearning/tofu_{model}_{retain_split}"
    train_command = f"""CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment={experiment} \
    task_name={task_name} \
    model={model} \
    forget_split={forget_split} \
    retain_split={retain_split} \
    paths.output_dir={output_dir} \
    retain_logs_path={retain_logs_path} \
    eval.tofu.metrics.mia_reference.reference_model_path={reference_model} \
    """
    
    train_command = add_hparams_to_cmd(train_command, train_hparams)

    command = f"""{train_command}

ls {output_dir}
rm {output_dir}/model.safetensors"""

    constraint = "a100"
    time = "2:30:0"
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
    output_dir = "saves/leaderboard"
    if "gradasc" in args.exp:
        trainer = "GradAscent"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        
        
        # lrs = [1e-5, 2e-5, 5e-5]
        lrs = [1e-5, 2e-5,]
        
        for lr in lrs:
            train_hparams = {
                "trainer": trainer,
                "trainer.args.learning_rate": lr,
                "trainer.args.num_train_epochs": num_train_epochs,
            }
            
            eval_hparams = {}
            task_name = f"{model}_{forget_split}_{trainer}_lr{lr}"
            cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
            template_params_list += cmd_params_list
    
    if "graddiff" in args.exp:
        trainer = "GradDiff"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 2e-5, 5e-5]
        alphas = [1, 2, 5]
        
        for lr in lrs:
            for alpha in alphas:
                train_hparams = {
                    "trainer": trainer,
                    "trainer.args.learning_rate": lr,
                    "trainer.args.num_train_epochs": num_train_epochs,
                    "trainer.method_args.alpha": alpha
                }
                
                eval_hparams = {}
                task_name = f"{model}_{forget_split}_{trainer}_lr{lr}_alpha{alpha}"
                cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
                template_params_list += cmd_params_list
    
    if "idkdpo" in args.exp:
        trainer = "DPO"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        experiment="unlearn/tofu/idk.yaml"
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        alphas = [1, 2, 5]
        betas = [0.01, 0.03, 0.05, 0.1]    
        
        for lr in lrs:
            for alpha in alphas:
                for beta in betas:
                    train_hparams = {
                        "trainer": trainer,
                        "trainer.args.learning_rate": lr,
                        "trainer.args.num_train_epochs": num_train_epochs,
                        "trainer.method_args.alpha": alpha,
                        "trainer.method_args.beta": beta
                    }
                    
                    eval_hparams = {}
                    task_name = f"{model}_{forget_split}_{trainer}_lr{lr}_alpha{alpha}_beta{beta}"
                    cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, experiment=experiment, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
                    template_params_list += cmd_params_list
    
    if "npo" in args.exp:
        trainer = "NPO"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        experiment="unlearn/tofu/default.yaml"
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        alphas = [1, 2, 5]
        betas = [0.01, 0.03, 0.05, 0.1]    
        
        for lr in lrs:
            for alpha in alphas:
                for beta in betas:
                    train_hparams = {
                        "trainer": trainer,
                        "trainer.args.learning_rate": lr,
                        "trainer.args.num_train_epochs": num_train_epochs,
                        "trainer.method_args.alpha": alpha,
                        "trainer.method_args.beta": beta
                    }
                    
                    eval_hparams = {}
                    task_name = f"{model}_{forget_split}_{trainer}_lr{lr}_alpha{alpha}_beta{beta}"
                    cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, experiment=experiment, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
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
        required=True,
        nargs='+',  # Accept one or more values as a list
        help="List of experiments to run"
    )
    args = parser.parse_args()
    main(args)