import os
import argparse

from sbatch import (
    create_submit_sbatch,
)    

def eval_muse(model="Llama-2-7b-hf", data_split="News", retain_split="retrain", ds_config_file="configs/accelerate/default_config.yaml", retain_logs_path=None):
    constraint = "l40s"
    time = "5:00:0"
    template_params_list = []

    command = """CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml data_split={data_split} task_name=muse_{model}_{data_split}_{retain_split} model={model} model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-{data_split}_{retain_split}"""
    command = command.format(model=model, data_split=data_split, retain_split=retain_split)
    exp_name = "muse_{model}_{data_split}_{retain_split}".format(model=model, data_split=data_split, retain_split=retain_split)
    if retain_logs_path:
        command += f" retain_logs_path={retain_logs_path}"
    output_dir = os.path.join("saves", "eval", exp_name)
    job_name = exp_name
    error_file = os.path.join(output_dir,"unity","err.txt")
    logs_file = os.path.join(output_dir,"unity","logs.txt")
    template_params = {
        "job_name": job_name,
        "constraint": constraint,
        "error_file": error_file,
        "logs_file": logs_file,
        "time": time,
        "command": command
    }
    template_params_list.append(template_params)
    return template_params_list


def unlearn_muse(model="Llama-2-7b-hf", data_split="News", ds_config_file="configs/accelerate/default_config.yaml", retain_logs_path=None):
    ACCELERATE_TEMPLATE = """export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
    
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--config_file {ds_config_file} \
--main_process_port $MASTER_PORT \
{command} \
trainer.args.per_device_train_batch_size=2 \
trainer.args.gradient_accumulation_steps=8 \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

{eval_command}
"""

    COMMANDS =  [
    # "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=GradAscent task_name={task_name}",
    # "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=GradDiff task_name={task_name}",
    # # "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name={task_name}",
    # "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=NPO  task_name={task_name}",
    # # "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=NPO trainer.method_args.retain_loss_type=KL  task_name={task_name}",
    # "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=NPO  task_name={task_name}",
    "src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml model={model} data_split={data_split} trainer=RMU  task_name={task_name}",
    ]
    eval_command_template = "CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml data_split={data_split} task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name} paths.output_dir=saves/unlearn/{task_name}/evals"
    JOB_NAMES = [
        # "muse_{model}_{data_split}_GradAscent",
        # "muse_{model}_{data_split}_GradDiff",
        # # "muse_{model}_{data_split}_GradDiff_KL",
        # "muse_{model}_{data_split}_NPO",
        # # "muse_{model}_{data_split}_NPO_KL",
        # "muse_{model}_{data_split}_SimNPO",
        "muse_{model}_{data_split}_RMU",
    ]
    # constraint = "l40s"
    constraint = "a100"
    time = "5:00:0"
    mem = "60GB"
    template_params_list = []
    for command, job_name in zip(COMMANDS, JOB_NAMES):
        job_name = job_name.format(model=model, data_split=data_split)
        command = command.format(model=model, data_split=data_split, task_name=job_name)
        eval_command = eval_command_template.format(model=model, data_split=data_split, task_name=job_name)
        if retain_logs_path:
            eval_command += f" retain_logs_path={retain_logs_path}"
            command += f" retain_logs_path={retain_logs_path}"
        accelerate_command = ACCELERATE_TEMPLATE.format(command=command, eval_command=eval_command, ds_config_file=ds_config_file)
        output_dir = os.path.join("saves", "unlearn", job_name)
        error_file = os.path.join(output_dir,"unity","err.txt")
        logs_file = os.path.join(output_dir,"unity","logs.txt")
        template_params = {
            "job_name": job_name,
            "constraint": constraint,
            "error_file": error_file,
            "logs_file": logs_file,
            "num_gpu": 2,
            "num_cpu": 2,
            "time": time,
            "command": accelerate_command,
            "mem": mem
        }
        template_params_list.append(template_params)
    return template_params_list

def main(args):
    ds_config_file="configs/accelerate/default_config.yaml"
    print_only = args.print_only
    template_params_list = []
    
    # template_params_list += eval_muse(model="Llama-2-7b-hf", data_split="News", retain_split="retrain", ds_config_file=ds_config_file, retain_logs_path=None)
    # template_params_list += eval_muse(model="Llama-2-7b-hf", data_split="Books", retain_split="retrain", ds_config_file=ds_config_file, retain_logs_path=None)
    
    # template_params_list += eval_muse(model="Llama-2-7b-hf", data_split="News", retain_split="target", ds_config_file=ds_config_file, retain_logs_path="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json")
    # template_params_list += eval_muse(model="Llama-2-7b-hf", data_split="Books", retain_split="target", ds_config_file=ds_config_file, retain_logs_path="saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json")
    
    template_params_list += unlearn_muse(model="Llama-2-7b-hf", data_split="News", ds_config_file=ds_config_file, retain_logs_path="saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json")
    template_params_list += unlearn_muse(model="Llama-2-7b-hf", data_split="Books", ds_config_file=ds_config_file, retain_logs_path="saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json")
    create_submit_sbatch(template_params_list, print_only=print_only)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to submit sbatch jobs with optional print-only mode.")
    parser.add_argument(
        "--print_only", 
        action="store_true", 
        help="If set, will only print the sbatch parameters without submitting."
    )
    args = parser.parse_args()
    main(args)