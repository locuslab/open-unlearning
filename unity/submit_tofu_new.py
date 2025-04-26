import os
import argparse

from sbatch import (
    create_submit_sbatch,
)    



def train_tofu(model="Llama-2-7b-chat-hf", dataset="TOFU_QA_retain", forget_split="forget10", retain_split="retain90", ds_config_file="configs/accelerate/default_config.yaml", retain_logs_path=None):
    
    ACCELERATE_TEMPLATE = """export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
    
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--config_file {ds_config_file} \
--main_process_port $MASTER_PORT \
{command} \
trainer.args.per_device_train_batch_size=4 \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true
    
{eval_command}
"""

    COMMANDS = [
        "src/train.py experiment=finetune/tofu/default.yaml task_name={task_name} model={model} data/datasets@data.train={dataset} data.train.{dataset}.args.hf_args.name={retain_split}",
    ]
    eval_command_template = "CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml forget_split={forget_split} task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=saves/finetune/{task_name}"
    JOB_NAMES = [
        "tofu_{model}_{retain_split}",
    ]
    
    constraint = "l40s"
    time = "5:00:0"
    template_params_list = []
    
    for command, job_name in zip(COMMANDS, JOB_NAMES):
        job_name = job_name.format(model=model, retain_split=retain_split)
        eval_commands = []
        if dataset == "TOFU_QA_retain":
            eval_command = eval_command_template.format(model=model, forget_split=forget_split, task_name=job_name)
            eval_commands.append(eval_command)
        elif dataset == "TOFU_QA_full":
            for f_split, r_split in [("forget10", "retain90"), ("forget05", "retain95"), ("forget01", "retain99")]:
                eval_command = eval_command_template.format(model=model, forget_split=f_split, task_name=job_name)
                retain_logs_path=f"saves/eval/tofu_{model}_{r_split}/TOFU_EVAL.json"
                eval_command += f" retain_logs_path={retain_logs_path}"
                eval_command += f" paths.output_dir=saves/eval/{job_name}/evals_{f_split}"
                eval_commands.append(eval_command)
        command = command.format(model=model, retain_split=retain_split, task_name=job_name, dataset=dataset)
        accelerate_command = ACCELERATE_TEMPLATE.format(command=command, eval_command="\n\n".join(eval_commands), task_name=job_name, ds_config_file=ds_config_file)
        # accelerate_command = ACCELERATE_TEMPLATE.format(eval_command="\n\n".join(eval_commands))
        output_dir = os.path.join("saves", "finetune", job_name)
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
            # "exclusive": exclusive,
        }
        template_params_list.append(template_params)
    return template_params_list


def unlearn_tofu(model="Llama-2-7b-chat-hf", forget_split="forget10", retain_split="retain90", ds_config_file="configs/accelerate/default_config.yaml", retain_logs_path=None):
    ACCELERATE_TEMPLATE = """export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
    
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--config_file configs/accelerate/default_config.yaml \
--main_process_port $MASTER_PORT \
{command} \
trainer.args.per_device_train_batch_size=4 \
trainer.args.gradient_accumulation_steps=4 \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

{eval_command}
"""
    # ACCELERATE_TEMPLATE = """{eval_command}"""

    COMMANDS = [
        # "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default.yaml trainer=GradAscent forget_split={forget_split} retain_split={retain_split} task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
        # "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default.yaml trainer=GradDiff forget_split={forget_split} retain_split={retain_split} task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
        # # "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default.yaml trainer=GradDiff forget_split={forget_split} retain_split={retain_split} trainer.method_args.retain_loss_type=KL task_name={task_name} model={model model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full}",
        # "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default.yaml trainer=NPO forget_split={forget_split} retain_split={retain_split} task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
        # # "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default.yaml trainer=NPO forget_split={forget_split} retain_split={retain_split} trainer.method_args.retain_loss_type=KL model={model} task_name={task_name} model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
        # "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/idk.yaml forget_split={forget_split} retain_split={retain_split} trainer=DPO task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
        "src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default.yaml forget_split={forget_split} retain_split={retain_split} trainer=RMU task_name={task_name} model={model} model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
    ]
    
    eval_command_template = "CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml forget_split={forget_split} model={model} task_name={task_name} model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name} paths.output_dir=saves/unlearn/{task_name}/evals"
    JOB_NAMES = [
        # "tofu_{model}_{forget_split}_GradAscent",
        # "tofu_{model}_{forget_split}_GradDiff",
        # # "tofu_{model}_{forget_split}_GradDiff_KL",
        # "tofu_{model}_{forget_split}_NPO",
        # # "tofu_{model}_{forget_split}_NPO_KL",
        # "tofu_{model}_{forget_split}_IdkDPO",
        # "tofu_{model}_{forget_split}_SimNPO",
        "tofu_{model}_{forget_split}_RMU",
    ]
    
    # constraint = "l40s"
    constraint = "a100"
    time = "5:00:0"
    template_params_list = []
    for command, job_name in zip(COMMANDS, JOB_NAMES):
        job_name = job_name.format(model=model, forget_split=forget_split)
        command = command.format(model=model, forget_split=forget_split, retain_split=retain_split, task_name=job_name)
        eval_command = eval_command_template.format(model=model, forget_split=forget_split, task_name=job_name)
        if retain_logs_path:
            eval_command += f" retain_logs_path={retain_logs_path}"
            command += f" retain_logs_path={retain_logs_path}"
        accelerate_command = ACCELERATE_TEMPLATE.format(command=command, eval_command=eval_command, ds_config_file=ds_config_file)
        # accelerate_command = ACCELERATE_TEMPLATE.format(eval_command="\n\n".join(eval_commands))
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
            "command": accelerate_command
        }
        template_params_list.append(template_params)
    return template_params_list


def main(args):
    ds_config_file="configs/accelerate/default_config.yaml"
    print_only = args.print_only
    template_params_list = []
    
    # template_params_list += train_tofu(model="Llama-3.2-1B-Instruct", dataset="TOFU_QA_retain", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.1-8B-Instruct", dataset="TOFU_QA_retain", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.2-3B-Instruct", dataset="TOFU_QA_retain", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.2-1B-Instruct", dataset="TOFU_QA_retain", forget_split="forget05", retain_split="retain95", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.1-8B-Instruct", dataset="TOFU_QA_retain", forget_split="forget05", retain_split="retain95", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.2-3B-Instruct", dataset="TOFU_QA_retain", forget_split="forget05", retain_split="retain95", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.2-1B-Instruct", dataset="TOFU_QA_retain", forget_split="forget01", retain_split="retain99", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.1-8B-Instruct", dataset="TOFU_QA_retain", forget_split="forget01", retain_split="retain99", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.2-3B-Instruct", dataset="TOFU_QA_retain", forget_split="forget01", retain_split="retain99", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-2-7b-chat-hf", dataset="TOFU_QA_retain", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-2-7b-chat-hf", dataset="TOFU_QA_retain", forget_split="forget05", retain_split="retain95", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-2-7b-chat-hf", dataset="TOFU_QA_retain", forget_split="forget01", retain_split="retain99", ds_config_file=ds_config_file)
    
    # template_params_list += train_tofu(model="Llama-3.2-1B-Instruct", dataset="TOFU_QA_full", forget_split="forget10", retain_split="full", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.2-3B-Instruct", dataset="TOFU_QA_full", forget_split="forget10", retain_split="full", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-3.1-8B-Instruct", dataset="TOFU_QA_full", forget_split="forget10", retain_split="full", ds_config_file=ds_config_file)
    # template_params_list += train_tofu(model="Llama-2-7b-chat-hf", dataset="TOFU_QA_full", forget_split="forget10", retain_split="full", ds_config_file=ds_config_file)
    
    template_params_list += unlearn_tofu(model="Llama-3.2-1B-Instruct", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json")
    template_params_list += unlearn_tofu(model="Llama-3.2-1B-Instruct", forget_split="forget05", retain_split="retain95", ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json")
    template_params_list += unlearn_tofu(model="Llama-3.2-1B-Instruct", forget_split="forget01", retain_split="retain99", ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json")
    
    template_params_list += unlearn_tofu(model="Llama-2-7b-chat-hf", forget_split="forget10", retain_split="retain90",ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-2-7b-chat-hf_retain90/TOFU_EVAL.json")
    template_params_list += unlearn_tofu(model="Llama-2-7b-chat-hf", forget_split="forget05", retain_split="retain95",ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json")
    template_params_list += unlearn_tofu(model="Llama-2-7b-chat-hf", forget_split="forget01", retain_split="retain99",ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json")
    
    
    # template_params_list += unlearn_tofu(model="Llama-3.2-3B-Instruct", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json")
    # template_params_list += unlearn_tofu(model="Llama-3.1-8B-Instruct", forget_split="forget10", retain_split="retain90", ds_config_file=ds_config_file, retain_logs_path="saves/eval/tofu_Llama-3.1-8B-Instruct_retain90/TOFU_EVAL.json")
    
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