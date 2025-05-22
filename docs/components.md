# Components

The OpenUnlearning framework requires a structured approach for adding new components in the unlearning pipeline.

This process involves three main steps:
1. __Implementing a handler__: Define the core logic for the component (usually a python class or function). A single handler can be reused across multiple components. For example, a handler that computes the ROUGE score can support various evaluation metrics across multiple datasets.
2. __Registering the handler__: Add the handler to a registry that links it to a key, allowing access during execution through the config files.
3. __Adding a config file__: Set up a configuration using Hydra that specifies the handler and relevant parameters. These configurations can then be passed directly as arguments when running Python scripts.

---

## Documentation on adding each type of component

1. [Trainer](#trainer) - Algorithm used in LLM training or unlearning
2. [Dataset](#dataset) - Dataset class for preprocessing raw data
3. [Evaluation Metric](#evaluation-metric) - Metric class implementing model evaluation
4. [Benchmark](#benchmark) - Suite combining multiple evaluation metrics
5. [Model](#model) - LLM used in unlearning
6. [Collator](#collator) - Handles data collation logic
7. [Experiment](#experiment) - Combines components into a final experiment config

> [!NOTE]
> Adding each component requires Hydra config management features, which are documented in [`docs/hydra.md`](../docs/hydra.md). 

---

## Trainer

To add a new **Trainer**:

### Implement a handler
We extend HuggingFace's [`Trainer`](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) for custom training algorithms. Trainer handlers are written in [`src/trainer`](../src/trainer/).

Example: defining a gradient-difference based unlearning trainer.

```python
class GradDiff(UnlearnTrainer):
    def __init__(self, gamma, alpha, ...):
        ...
      
    def compute_loss(self, model, inputs, return_outputs=False):
        ...
```

### Register the trainer handler
Register the handler to link the class to the configs via the class name in [`TRAINER_REGISTRY`](../src/trainer/__init__.py).

Example: Registering a fine-tuning trainer and `GradDiff` unlearning trainer 

```python
from transformers import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradDiff
_register_trainer(FinetuneTrainer) # class defined in src/trainer/base.py
_register_trainer(GradDiff) # class defined in src/trainer/unlearn/grad_diff.py
```

### Add a trainer to configs

Add a config that uses the new trainer and set parameters. Trainer configurations are in [`configs/trainer`](../configs/trainer/). Each config contains a handler that points to the defined trainer class and the arguments used to initialise the trainer.

Example: Config file ([`configs/trainer/GradDiff.yaml`](../configs/trainer/GradDiff.yaml)) for GradDiff.
```yaml
handler: GradDiff # corresponds to the class defined in src/trainer/unlearn/grad_diff.py
args: # HuggingFace TrainingArguments
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  num_train_epochs: 10
method_args: # Your own method-specific arguments
  gamma: 1.0
  alpha: 1.0
  retain_loss_type: NLL
```

---

## Dataset

To add a new dataset, we create a generic preprocessing handler and then configure it to create a dataset:

### Implement a handler
Extend `torch.utils.data.Dataset` to to create dataset handlers for loading and preprocessing data. These are written in [`src/data`](../src/data/). A new dataset would then instantiated by providing its parameters (dataset column, length etc) to an existing dataset handler.

Example: defining a `PretrainingDataset` dataset handler to load texts for pre-training style next token prediction.

```python
class PretrainingDataset(Dataset):
    def __init__(self, hf_args, text_key, max_length, ...):
        ...

    def __getitem__(self, idx):
        ...
        return item
```

### Register the dataset handler
Register the handler to link the class to the configs via the class name in [`DATASET_REGISTRY`](../src/data/__init__.py).

Example: Registering `PretrainingDataset`

```python
from data.pretraining import PretrainingDataset
_register_data(PretrainingDataset)
```

### Add a dataset to configs
Add a specific instance of dataset class that uses the `PretrainingDataset` class format. Dataset configurations go in [`configs/data/datasets`](../configs/data/datasets/). Each config contains a handler that points to the defined dataset class and the arguments used to create the dataset.

Example: add a config file for the `MUSE_forget` and `MUSE_forget_sust` datasets using the `PretrainingDataset` handler
```yaml
MUSE_forget: # the name of a particular dataset instance
  handler: PretrainingDataset # name of the dataset class
  args:
    hf_args:
      path: "muse-bench/MUSE-News"
      name: "raw"
      split: "forget"
    text_key: "text"
    max_length: 2048

MUSE_forget_sust: # another dataset
  handler: PretrainingDataset # name of the dataset class
  args:
    hf_args:
      path: "muse-bench/MUSE-Books"
      name: "sust"
      split: "forget_1"
    text_key: "text"
    max_length: 2048
```
---

## Evaluation Metric

To add a new evaluation metric, we create a handler with the metric computation logic and then configure it. More documentation on adding metrics is in [`docs/evaluation.md#metrics`](../docs/evaluation.md#metrics).


## Benchmark

A benchmark, aggregates various evaluation metrics into a suite, e.g. TOFU, MUSE etc. To add a new benchmark, we create a handler ([example](../src/evals/muse.py)) with the metric aggregation logic, benchmark name etc. and then create a config. More documentation on adding metrics is in [`docs/evaluation.md#benchmarks`](../docs/evaluation.md#benchmarks).


## Model

To add a new model architecture:

### Implement a handler

Most tokenizers can be loaded directly using `AutoTokenizer`, so there's no need to register them explicitly.

We support Hugging Face's `AutoModelForCausalLM`, which is commonly used to load and serve a wide range of language models. For custom extension—such as[`ProbedLlamaForCausalLM`](../src/model/probe.py), implement a class extending `AutoModelForCausalLM` in [`src/model`](../src/model).

```python
class ProbedLlamaForCausalLM(LlamaForCausalLM):
  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      head_pretrained_model_name_or_path: str = None,
      n_layers: int = 100,
      **kwargs,
  ):
    ...
```

### Register a handler
Register your new model handler in the [`MODEL_REGISTRY`](../src/model/__init__.py).

```python
from model.probe import ProbedLlamaForCausalLM
_register_model(ProbedLlamaForCausalLM)
```

> [!NOTE]
Currently, we do not support loading models modified with LoRA and related variants. If you wish use such features, please create define and register model handlers for this logic in [`src/model`](../src/model) and provide the config info as discussed next.

### Add to configs
Model configurations contain details required to load the model+tokenizer such as paths, chat templating arguments, LoRA parameters etc. in [`configs/models`](../configs/models/).

Example: LLaMA-3.1 model config in [`configs/model/Llama-3.1-8B-Instruct.yaml`](../configs/model/Llama-3.1-8B-Instruct.yaml).

```yaml
model_handler: AutoModelForCausalLM
model_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16
tokenizer_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
template_args:
  apply_chat_template: True
  system_prompt: You are a helpful assistant.
```

---

## Collator

Different dataset formats might have different data collation logic to pad and organize sequences in a batch. We do not expect most users to require new collators, but we provide the option to extend this component if needed.

### Implement a handler
Collators implementing batch collation are implemented in [`src/collators`](../src/collators/), imported in [`src/collators/__init__.py`](../src/collators/__init__.py).

```python
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer, padding_side, index):
      ...
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
      ...
```

### Register Collator handler
Register the collator to link the class to the configs via the class name in [`COLLATOR_REGISTRY`](../src/collators/__init__.py).

Example: Registering `DataCollatorForSupervisedDataset` 

```python
from collators.base import DataCollatorForSupervisedDataset
_register_collator(DataCollatorForSupervisedDataset)
```

### Add to configs
Collator configurations are in [`configs/collator`](../configs/collator/).

```yaml
DataCollatorForSupervisedDataset:
  handler: DataCollatorForSupervisedDataset
  args:
    padding_side: right
```

---

## Experiment


Experiment configs helps interface with various benchmarks and setups using certain default configs.
They reduce the need to manually set and override the many components and attributes. There is no handler or registration required here, as this is done completely in Hydra.

These configs are found in [`configs/experiment`](../configs/experiment/).

More details on how to run and organise experiments are in [`docs/experiment.md`](experiment.md).

### Add to configs
Experiment configurations specify the model, dataset, trainer, and evaluation components.

Example: a TOFU unlearning experiment configuration (from [`configs/experiment/unlearn/tofu/default.yaml`](../configs/experiment/unlearn/tofu/default.yaml)) involves setting the model, the trainer, the dataset, the evaluation benchmark and the various attributes involves in them.

```yaml
# @package _global_

defaults: # load pre-defined configs for model, trainer, data format, datasets etc.
  - override /model: Llama-2-7b-chat-hf # from configs/model/Llama-2-7b-chat-hf.yaml
  - override /trainer: GradAscent # from configs/trainer/GradAscent.yaml
  - override /data: unlearn # ...
  - override /data/datasets@data.forget: TOFU_QA_forget
  - override /data/datasets@data.retain: TOFU_QA_retain
  - override /eval: tofu

# Now, we have to further modify specific arguments from the defaults imported above
# This enables easily running multiple experiments varying hyper paramters, data splits, models etc

model:
  model_args: # use our finetuned target models for the TOFU benchmark task
    pretrained_model_name_or_path: open-unlearning/tofu_Llama-3.2-1B-Instruct_full

forget_split: forget10 
retain_split: retain90
retain_logs_path: null

eval:
  tofu:
    forget_split: ${forget_split}
    retain_logs_path: ${retain_logs_path}
    
data:
  anchor: forget
  forget:
    TOFU_QA_forget: 
      args:
        hf_args:
          name: ${forget_split}
  retain:
    TOFU_QA_retain:
      args:
        hf_args:
          name: ${retain_split}

trainer:
  args:
    warmup_epochs: 1.0
    learning_rate: 2e-5
    weight_decay: 0.01
    num_train_epochs: 10

override task_name: llama2_unlearn
```