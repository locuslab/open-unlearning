<div align="center">

# Configuring and running experiments

</div>


## Overview

The large number of component variants supported in this repository creates the need for configuring many components and their parameters before running a specific experiment. We rely on features provided by Hydra to make this process easier.

At the core, three main Hydra configs—`train.yaml` (generic training), `eval.yaml` (running evaluation), and `unlearn.yaml` (unlearning training)—provide the base configuration for the main types of experiments. These are then extended by experiment-specific configs and command-line overrides. We set up experiment configs for common usecases like LLaMA-2 unlearning on TOFU, LLaMA-2 evaluation on MUSE etc. which set the required datasets, models, and base train and eval configs to make things easier.

Experiment output directories are constructed based on the task mode (`train` / `eval` / `unlearn`) and the task name (provided by the user) as `./saves/${mode}/${task_name}`. The experiment logging will display where the model checkpoints, logs and evaluation dumps are stored.

---

### Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Example Commands](#example-commands)
- [Commonly Overridden Arguments](#commonly-overridden-arguments)
  - [Model Settings](#model-settings)
  - [Trainer Settings](#trainer-settings)
  - [Data Settings](#data-settings)
  - [Experiment Settings](#experiment-settings)
- [Simple Finetuning](#simple-finetuning)
- [Distributed Training](#distributed-training)

---

## Example Commands

```bash
## runs a finetuning using experiment details from configs/finetune/tofu/default.yaml
python src/train.py --config-name=train.yaml experiment=finetune/tofu/default task_name=SAMPLE_TRAIN

## runs an unlearning training using experiment details from configs/unlearn/tofu/default.yaml
# output directory will be constructed as: saves/unlearn/SAMPLE_UNLEARN
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default task_name=SAMPLE_TRAIN


## runs an evaluation using experiment details from configs/eval/muse/default.yaml
python src/eval.py --config-name=eval.yaml experiment=eval/muse/default task_name=SAMPLE_EVAL
## Note: eval.yaml is the default config set in src/eval.py, so this argument can be omitted

## an extensively filled out configuration for an unlearning experiment
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default data_split=News \
trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_books_NPO_KL \
retain_logs_path=saves/eval/muse_books_retain/MUSE_EVAL.json

## an even more extensively filled out configuration for an unlearning experiment
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default.yaml \
task_name=NPO_unlearn_tofu_llama_8 \
model=Llama-3.1-8B-Instruct \
model.model_args.pretrained_model_name_or_path=saves/finetune/path_model_llama \
trainer=NPO trainer.args.per_device_train_batch_size=4 \
forget_split=forget05 retain_split=retain95 \
retain_logs_path=saves/eval/tofu_retain95/TOFU_EVAL.json \
paths.output_dir=saves/unlearn/NPO/evals
```


> [!NOTE]
The unlearning experiments support evaluation during the unlearning finetuning. But this is supported only when a single accelerator process is used, checkpoints must be stored and evaluated after training.

---

## Commonly Overridden Arguments

To understand the structure of an evaluation config and the kind of available parameters for overriding, refer to: [`configs/experiment/examples/tofu_eval.yaml`](../configs/experiment/examples/tofu_eval.yaml).

To understand the structure of an unlearning config and the kind of available parameters for overriding, refer to: [`configs/experiment/examples/muse_unlearn.yaml`](../configs/experiment/examples/muse_unlearn.yaml).

The following tables list the most commonly used arguments while running experiments.

### <h3>Model Settings</h3>
<table>
  <colgroup>
    <col class="argument">
    <col class="description">
  </colgroup>
  <tr>
    <th>Argument</th>
    <th>Description and examples</th>
  </tr>
  <tr>
    <td><code>model</code></td>
    <td>Selecting the model. Example: <code>model=Llama-2-7b-hf</code></td>
  </tr>
  <tr>
    <td><code>model.model_args.pretrained_model_name_or_path</code></td>
    <td>Specifies the model checkpoint or HuggingFace ID.</td>
  </tr>
  <tr>
    <td><code>model.tokenizer_args.pretrained_model_name_or_path</code></td>
    <td>Specifies the tokenizer location. Make sure this matches the model from above by providing model path as needed..</td>
  </tr>
  <tr>
    <td><code>model.template_args</code></td>
    <td>Optional chat templating parameters (e.g., start/end tags). Example: <code>apply_chat_template: false, user_start_tag: "[INST] "</code></td>
  </tr>
</table>

### <h3>Trainer Settings</h3>
<table>
  <colgroup>
    <col class="argument">
    <col class="description">
  </colgroup>
  <tr>
    <th>Argument</th>
    <th>Description and examples</th>
  </tr>
  <tr>
    <td><code>trainer</code></td>
    <td>Overall trainer or unlearning method selection, decides the finetuning algorithm. Example: <code>trainer=NPO</code> or <code>trainer=finetune</code></td>
  </tr>
  <tr>
    <td><code>trainer.args</code></td>
    <td>Main training hyperparameters like <code>per_device_train_batch_size</code>, <code>per_device_eval_batch_size</code>, <code>gradient_accumulation_steps</code>, <code>learning_rate</code>, <code>num_train_epochs</code>, <code>optim</code> and other HuggingFace TrainingArguments.
    </td>
  </tr>
    <td><code>trainer.method_args</code></td>
    <td>Method-specific parameters for unlearning trainers. Example: <code>retain_loss_type</code>, NPO hyperparams like <code>gamma, alpha, beta</code> etc.</td>
  </tr>
</table>

### <h3>Data Settings</h3>
<table>
  <colgroup>
    <col class="argument">
    <col class="description">
  </colgroup>
  <tr>
    <th>Argument</th>
    <th>Description and examples</th>
  </tr>
  <tr>
    <td><code>data</code></td>
    <td>Overall data configuration/format. Example: <code>data=unlearn</code>, <code>data=finetune</code>.</td>
  </tr>
  <tr>
    <td><code>data.forget, data.retain, data.anchor</code> etc.</td>
    <td>Set sub-datasets in the overall dataset using <code>data.forget=MUSE_forget data.retain=MUSE_retain</code>, set which sub-dataset to index over (others are randomly sampled) using <code>data.anchor=forget</code></td>
  </tr>
  <tr>
    <td><code>data_split/forget_split/retain_split</code></td>
    <td>These arguments are custom to specific datasets and are used to populate dataset paths.
    <br>
    <code>data_split</code> specifies the overall dataset split or type. Example: <code>data_split=News</code> or <code>data_split=Books</code>
    <br>
    <code>forget_split/retain_split</code> splits are used to use various sub-parts of the dataset. Example: <code>forget_split=forget01 retain_split=retain99</code></td>
  </tr>
</table>

### <h3>Experiment Settings</h3>
<table>
  <colgroup>
    <col class="argument">
    <col class="description">
  </colgroup>
  <tr>
    <th>Argument</th>
    <th>Description and examples</th>
  </tr>
  <tr>
    <td><code>task_name</code></td>
    <td>
      Experiment identifier used to generate custom output paths. 
      Example: <code>task_name=llama2_books_NPO_KL</code>.
    </td>
  </tr>
  <tr>
    <td><code>eval</code></td>
    <td>
      Overall evaluation benchmark configuration selection.
      Example: <code>eval=muse</code>.
    </td>
  </tr>
  <tr>
    <td><code>retain_logs_path</code></td>
    <td>
      Path to load eval logs of retain models used some evaluation metrics
      Example: <code>retain_logs_path=saves/eval/muse_books_retain/MUSE_EVAL.json</code>.
    </td>
  </tr>
  <tr>
    <td><code>paths</code></td>
    <td>
      Contains attributes used to decide path configuration like <code>paths.output_dir=$LOCAL_PATH</code>.
    </td>
  </tr>
</table>


---


## Simple Finetuning

In addition to running unlearning based finetuning, we also support simple finetuning training with a given dataset. 

These use [`src/train.py`](../src/train.py) with the [`train.yaml`](../train.yaml) config to set up a standard supervised training environment. Parameters such as learning rate, batch size, and optimizer settings can be adjusted via experiment-specific configs or command-line overrides.

Example:

```bash
python src/train.py --config-name=train.yaml experiment=finetune/tofu/default \
  trainer.args.learning_rate=5e-5 task_name=llama3.2-1B_finetune_example
```

## Distributed Training

Distributed training configurations enable scaling experiments across multiple devices or nodes. In most cases, default distributed settings from [`configs/accelerate/default_config.yaml`](../configs/accelerate/default_config.yaml) are sufficient. You can run distributed training with the below command that uses DeepSpeed for distributed training (which is our default setup):

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml task_name=DISTRIBUTED_TRAIN
```

You may also simply run `CUDA_VISIBLE_DEVICES=0,1,.. python ...` to leverage Accelerate's DDP or model parallel. For model parallel you can use `device_map="auto"` in the `model_args` while loading the model.

> [!CAUTION]
> Train runs using multiple accelerate processes will not be able to run evaluations during training. To achieve this, you may want to use DDP/model parallel (see #94) or use a single GPU to run the evaluation code directly on a saved model checkpoint like below

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml task_name=SAMPLE_EVAL \
model.model_args.pretrained_model_name_or_path=saves/unlearn/muse_unlearn_exp \
```
