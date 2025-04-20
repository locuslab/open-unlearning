## Hydra Features

The below are some important Hydra features we use for flexible composition while writing configurations to our YAML files.

We use this config file for illustration, from [`configs/experiment/unlearn/muse/default.yaml`](../configs/experiment/unlearn/muse/default.yaml):

```yaml
# @package _global_
# ^ not a comment, sets the path of this config to be the the config root directory
defaults:
- override /model: Llama-2-7b-hf # loads from model/Llama-2-7b-hf.yaml into the model attribute
- override /trainer: GradAscent # loads from trainer/GradAscent.yaml into the trainer attribute
- override /data: unlearn # loads from data/unlearn.yaml into the "data" attribute,, setting up data structures for loading datasets during unlearning
- override /eval: muse # loads MUSE evaluation suite from eval/muse.yaml into the eval attribute 

# define variables
data_split: News
forget_split: forget
retain_split: retain1
retain_logs_path: null

model:
model_args:
    pretrained_model_name_or_path: muse-bench/MUSE-${data_split}_target
tokenizer_args:
    pretrained_model_name_or_path: muse-bench/MUSE-${data_split}_target
data:
    anchor: forget
    forget:
        MUSE_forget: 
        args:
            hf_args:
            split: ${forget_split}
    retain:
        MUSE_retain:
        args:
            hf_args:
            split: ${retain_split}

eval:
    muse:
        data_split: ${data_split}
        retain_logs_path: ${retain_logs_path}

trainer:
    args:
        per_device_train_batch_size: 4
        gradient_accumulation_steps: 8
        learning_rate: 1e-5
        num_train_epochs: 10
        lr_scheduler_type: constant
        # save_strategy: steps
        # save_steps: 0.5
        # optim: paged_adamw_32bit
        # optim: adamw_torch

task_name: ??? # ??? raises and error if this attribute is not set
```

- **Structure & Attribute Access:** Configs are written in YAML and structured hierarchically like a dictionary. Attributes are accessed using dot notation: In code `cfg.model.args.learning_rate`, in command-line: `model.args.learning_rate=1e-5`.

- **Defaults & Overrides:**  Configs are files are included in one another using `defaults` and `override` commands. 

- **Command-Line Overrides:**  Any parameter can be overridden directly from the command line. For instance:
```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default \
trainer.args.num_train_epochs=50 data_split=Books trainer=SimNPO trainer.method_args.beta=3 \
task_name=unlearn_muse_simnpo
```

- **Package Directives:**  The `# @package` directive organizes configurations into namespaces for cleaner composition and specifies the configuration path. At the head of a YAML file, you might see directives like `# @package _global_` or more specific ones such as `# @package eval.muse.metrics.forget_knowmem_ROUGE` which inform Hydra exactly where the configuration parameters should be placed within the final composed config.

    For example, refer [`configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml`](../configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml) 

- **Variable Substitution:**  Variables are defined once and reused using the `${}` syntax.

- **Adding New Attributes with `+`:** Use the `+` prefix to add attributes that are not already in the config. For example, to add a new argument to the trainer:
```bash
python src/train.py experiment=unlearn/muse/default +trainer.args.my_new_arg=10
```

- **Attribute Removal with `~`:** You can remove an attribute from the config at runtime using the tilde `~`. For example, to remove flash attention setting:
```bash
python src/train.py experiment=unlearn/muse/default ~model.model_args.attn_implementation
```
> [!NOTE]
> In `zsh`, you must **quote** or **escape** the `~` to avoid it being misinterpreted as a home directory: e.g.:
```bash
python src/train.py \~model.model_args.attn_implementation
python src/train.py "~model.model_args.attn_implementation"
```
> [!NOTE]
> Hydra uses PyYAML to handle yaml files and transform inputs while giving config inputs. This handles cases like converting `true` to `True`

Refer to the following for config structures and overridable parameters:
- Evaluation: [`configs/experiment/examples/tofu_eval.yaml`](../configs/experiment/examples/tofu_eval.yaml)
- Unlearning: [`configs/experiment/examples/muse_unlearn.yaml`](../configs/experiment/examples/muse_unlearn.yaml)