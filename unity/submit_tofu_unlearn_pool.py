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
    
    train_command = f"""CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment={experiment} \
    task_name={task_name} \
    model={model} \
    forget_split={forget_split} \
    retain_split={retain_split} \
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
    
    CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split={forget_split} \
    holdout_split={holdout_split} \
    task_name={task_name} \
    model={model}-JB \
    model.model_args.pretrained_model_name_or_path=$dir \
    eval.tofu.question_key="paraphrased_question" \
    paths.output_dir=$dir/eval_Q_PARA_JB
    
    CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split={forget_split} \
    holdout_split={holdout_split} \
    task_name={task_name} \
    model={model} \
    +model.model_args.load_in_4bit=True \
    model.model_args.pretrained_model_name_or_path=$dir \
    eval.tofu.question_key="paraphrased_question" \
    paths.output_dir=$dir/eval_Q_PARA_quant4bit
done
"""
    command = f"""{train_command}
    
{eval_command}"""

    constraint = "a100"
    time = "4:00:0"
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
    output_dir = "saves/pool/unlearn"
    if "gradasc" in args.exp:
        trainer = "GradAscent"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        save_total_limit=5
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        
        for lr in lrs:
            train_hparams = {
                "trainer": trainer,
                "trainer.args.learning_rate": lr,
                "trainer.args.num_train_epochs": num_train_epochs,
                "trainer.args.eval_strategy": "no",
                "trainer.args.save_strategy": "steps",
                "trainer.args.eval_on_start": False,
                "+trainer.args.save_steps": 0.1,
                "+trainer.args.save_total_limit":save_total_limit,
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
        save_total_limit=5
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        
        for lr in lrs:
            train_hparams = {
                "trainer": trainer,
                "trainer.args.learning_rate": lr,
                "trainer.args.num_train_epochs": num_train_epochs,
                "trainer.args.eval_strategy": "no",
                "trainer.args.save_strategy": "steps",
                "trainer.args.eval_on_start": False,
                "+trainer.args.save_steps": 0.1,
                "+trainer.args.save_total_limit":save_total_limit,
            }
            
            eval_hparams = {}
            task_name = f"{model}_{forget_split}_{trainer}_lr{lr}"
            cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
            template_params_list += cmd_params_list
    
    if "idkdpo" in args.exp:
        trainer = "DPO"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        save_total_limit=5
        experiment="unlearn/tofu/idk.yaml"
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        betas = [0.05, 0.1, 0.5]
        
        for lr in lrs:
            for beta in betas:
                train_hparams = {
                    "trainer": trainer,
                    "trainer.args.learning_rate": lr,
                    "trainer.method_args.beta": beta,
                    "trainer.args.num_train_epochs": num_train_epochs,
                    "trainer.args.eval_strategy": "no",
                    "trainer.args.save_strategy": "steps",
                    "trainer.args.eval_on_start": False,
                    "+trainer.args.save_steps": 0.1,
                    "+trainer.args.save_total_limit":save_total_limit,
                }
                
                eval_hparams = {}
                task_name = f"{model}_{forget_split}_{trainer}_lr{lr}_beta{beta}"
                cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, experiment=experiment, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
                template_params_list += cmd_params_list
    
    if "npo" in args.exp:
        trainer = "NPO"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        save_total_limit=5
        experiment="unlearn/tofu/default.yaml"
        
        ## 3*5 = 15 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        betas = [0.05, 0.1, 0.5]
        
        for lr in lrs:
            for beta in betas:
                train_hparams = {
                    "trainer": trainer,
                    "trainer.args.learning_rate": lr,
                    "trainer.method_args.beta": beta,
                    "trainer.args.num_train_epochs": num_train_epochs,
                    "trainer.args.eval_strategy": "no",
                    "trainer.args.save_strategy": "steps",
                    "trainer.args.eval_on_start": False,
                    "+trainer.args.save_steps": 0.1,
                    "+trainer.args.save_total_limit":save_total_limit,
                }
                
                eval_hparams = {}
                task_name = f"{model}_{forget_split}_{trainer}_lr{lr}_beta{beta}"
                cmd_params_list = unlearn_tofu(train_hparams, eval_hparams, task_name, output_dir, experiment=experiment, model=model, forget_split=forget_split, retain_split=retain_split, holdout_split=holdout_split)
                template_params_list += cmd_params_list
    
    if "rmu" in args.exp:
        trainer = "RMU"
        model = "Llama-3.2-1B-Instruct"
        forget_split="forget10"
        retain_split="retain90"
        holdout_split="holdout10"
        num_train_epochs=10
        save_total_limit=5
        experiment="unlearn/tofu/default.yaml"
        
        ## 3*3*3*5 = 135 checkpoints
        
        lrs = [1e-5, 5e-5, 1e-4]
        layers = [5, 10, 15]
        steering_coeffs = [1, 10, 100]
        
        for lr in lrs:
            for layer in layers:
                for scoeff in steering_coeffs:
                    trainable_params_regex=f'\'["model\\.layers\\.({layer-2}|{layer-1}|{layer})\\.mlp\\.down_proj\\.weight"]\''
                    train_hparams = {
                        "trainer": trainer,
                        "trainer.args.learning_rate": lr,
                        "trainer.method_args.module_regex": f"model.layers.{layer}",
                        "trainer.method_args.trainable_params_regex":trainable_params_regex,
                        "trainer.method_args.steering_coeff": scoeff,
                        "trainer.args.num_train_epochs": num_train_epochs,
                        "trainer.args.eval_strategy": "no",
                        "trainer.args.save_strategy": "steps",
                        "trainer.args.eval_on_start": False,
                        "+trainer.args.save_steps": 0.1,
                        "+trainer.args.save_total_limit":save_total_limit,
                    }
                
                    eval_hparams = {}
                    task_name = f"{model}_{forget_split}_{trainer}_lr{lr}_layer{layer}_scoeff{scoeff}"
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