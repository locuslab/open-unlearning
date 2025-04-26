import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Define the SBATCH command template at the top
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p {gpu}
#SBATCH --job-name={job_name}
#SBATCH -N 1
#SBATCH -c {num_cpu}
#SBATCH -G {num_gpu}    
#SBATCH --mem={mem}
#SBATCH --constraint={constraint}
#SBATCH -e {error_file}
#SBATCH --output={logs_file}
#SBATCH --mail-user={user}@umass.edu
#SBATCH --mail-type=ALL
#SBATCH -t {time}
#SBATCH --account={account}
{exclusive_directive}
TZ="America/New_York" date

module load conda/latest
nvidia-smi

conda activate unlearning

{command}
"""


def submit_sbatch_command(sbatch_file_name, command):
    with open(sbatch_file_name, 'w') as f:
        f.write(command)
    exit_code = os.system(f"sbatch {sbatch_file_name}")
    return exit_code

def prepare_template_params(template_params):
    # Runtime parameters
    USER = "vdorna"
    ACCOUNT = "pi_wenlongzhao_umass_edu"
    template_params["gpu"] = template_params.get("gpu", "gpu-preempt")
    template_params["user"] = template_params.get("user", USER)
    template_params["account"] = template_params.get("account", ACCOUNT)
    template_params["num_gpu"] = template_params.get("num_gpu", 1)
    template_params["num_cpu"] = template_params.get("num_cpu", 1)
    template_params["mem"] = template_params.get("mem", "30GB")
    template_params["constraint"] = template_params.get("constraint", "a100")
    template_params["exclusive"] = template_params.get("exclusive", "")
    if template_params["exclusive"] == "exclusive":
        template_params["exclusive_directive"] = "#SBATCH --exclusive"
    else:
        template_params["exclusive_directive"] = ""
    return template_params

def recursive_delete_if_empty(start_dir):
    """
    Recursively delete a directory if it is empty, starting from the specified directory.
    """
    start_dir = Path(start_dir)

    # First, delete the starting directory
    if start_dir.exists():
        try:
            shutil.rmtree(start_dir)  # Delete `start_dir` and its contents
            print(f"Deleted directory: {start_dir}")
        except Exception as e:
            print(f"Error deleting {start_dir}: {e}")
            return

    # Move to the parent directory and recursively check for emptiness
    parent_dir = start_dir.parent
    while parent_dir.exists() and not any(parent_dir.iterdir()):  # Check if empty
        try:
            parent_dir.rmdir()  # Remove the empty parent directory
            print(f"Removed empty parent directory: {parent_dir}")
            parent_dir = parent_dir.parent  # Move up to the next level
        except Exception as e:
            print(f"Error removing {parent_dir}: {e}")
            break
    
def create_submit_sbatch(template_params_list, print_only=True):
    if isinstance(template_params_list, dict):
        template_params_list = [template_params_list]
    for template_params in template_params_list:
        template_params = prepare_template_params(template_params)
        sbatch_command = SBATCH_TEMPLATE.format(**template_params)
        print(template_params.get("command"))
        print("="*30)
        if print_only:
            continue
        unity_dir = os.path.dirname(template_params['logs_file'])
        os.makedirs(unity_dir, exist_ok=True)
        sbatch_file_name = os.path.join(unity_dir, "sbatch")
        exit_code = submit_sbatch_command(sbatch_file_name, sbatch_command)
        # Check the exit code and remove directories if the sbatch command failed
        if exit_code != 0:
            print(f"sbatch command failed with exit code {exit_code}. Removing directories...")
            output_dir = Path(unity_dir)
            if output_dir.exists():
                recursive_delete_if_empty(output_dir)

def add_argument_to_cmd(cmd, argument):
    cmd = cmd.rstrip("\\").rstrip()
    argument = argument.strip("\\").strip()
    cmd += "     " + argument
    return cmd
