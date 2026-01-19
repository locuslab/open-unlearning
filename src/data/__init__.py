import copy
from typing import Dict, Any, Union
import os
import json
from omegaconf import DictConfig

from data.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset, QAEdgeDataset
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.pretraining import PretrainingDataset, CompletionDataset

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    # Convert DictConfig to dict to avoid unexpected fields from merging
    dataset_args = dict(dataset_cfg.get("args", {}))
    
    # Check if we need to load from JSON file (for QAEdgeDataset with data_path)
    if dataset_args.get("format") == "json" and "data_path" in dataset_args:
        data_path = dataset_args.pop("data_path")
        edge = dataset_args.pop("edge", None)
        edge_filter = dataset_args.pop("edge_filter", None)
        
        # Load and filter JSON data
        with open(data_path, "r") as f:
            json_data = json.load(f)
        
        edge_key = dataset_args.get("edge_key", "edge")
        
        if edge is not None:
            filtered_data = [d for d in json_data if d.get(edge_key, "").startswith(edge)]
        elif edge_filter is not None:
            filtered_data = [d for d in json_data if not d.get(edge_key, "").startswith(edge_filter)]
        else:
            # No filtering - use all data
            filtered_data = json_data
        
        # Set the dataset parameter for QAEdgeDataset
        dataset_args["dataset"] = filtered_data
    
    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return mapping to multiple datasets
    return dataset

def load_dataset_json(dataset_name):
    data_path = os.path.join("dataset/unlearning", f"{dataset_name}.json")
    print(f"Loading dataset from {data_path}")
    with open(data_path, "r") as f:
        dataset_full = json.load(f)
    return dataset_full


def split_raw_dataset_for_forget(
    cfg,
    data_path,
    model_base,
    forget_edge,
    instructions_only=True,
    torch_reformat=False,
):
    def torch_reformat(cfg, input_QA_list, tokenizer):
        max_length = 500
        torch_format_dataset = QuestionsDataset(
            input_QA_list=input_QA_list,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=max_length,
            split="train",
        )
        torch_format_dataset = DataLoader(
            torch_format_dataset,
            batch_size=1,
            shuffle=False,
        )

        return torch_format_dataset

    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Split into 'forget' and 'retain' based on the 'edge' key
    forget_dataset = [
        d for d in dataset if d["edge"] in forget_edge
    ]  # because we want to unlearn edge one by one
    if cfg.use_different_retain_dataset:
        with open(cfg.different_retain_set_path, "r") as f:
            dataset = json.load(f)
        retain_dataset = [d for d in dataset] # use the whole dataset as retain dataset
    else:
        retain_dataset = [d for d in dataset if d["edge"] not in cfg.forget_edge]
    if instructions_only:
        forget_dataset = [d["question"] for d in forget_dataset]
        retain_dataset = [d["question"] for d in retain_dataset]
    else:
        if torch_reformat:
            forget_dataset = torch_reformat(cfg, forget_dataset, model_base.tokenizer)
            retain_dataset = torch_reformat(cfg, retain_dataset, model_base.tokenizer)

    return forget_dataset, retain_dataset


def _load_json_dataset(data_path, dataset_cfg, edge=None, edge_filter=None, **kwargs):
    handler_name = dataset_cfg.get("handler", "QAEdgeDataset")
    handler = DATASET_REGISTRY.get(handler_name)
    if handler is None:
        raise NotImplementedError(f"{handler_name} not registered")

    with open(data_path, "r") as f:
        dataset = json.load(f)

    if edge is not None:
        dataset_filtered = [
            d for d in dataset if d.get("edge", "").startswith(edge)
        ]
    elif edge_filter is not None:
        dataset_filtered = [
            d for d in dataset if not d.get("edge", "").startswith(edge_filter)
        ]
    else:
        raise ValueError("Either edge or edge_filter must be provided")
    return handler(dataset_filtered, **kwargs)


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    
    # Check if top-level format is 'json'
    format_type = data_cfg.get("format", None)
    data = {}
    for split, dataset_cfgs in data_cfg.items():
        # Skip format/data_path/edge/edge_filter if they're at top level
        if split in ("format", "data_path", "edge", "edge_filter"):
            continue
        
        # Convert to dict if DictConfig
        if isinstance(dataset_cfgs, DictConfig):
            dataset_cfgs = dict(dataset_cfgs)
        
        # Check if this split should load from JSON
        edge = None
        edge_filter = None
        # Check top-level format/data_path/edge first
        # import IPython; IPython.embed()
        if format_type == "json":
            data_path = dataset_cfgs.get("data_path", None)
            edge = dataset_cfgs.get("edge", None)
            edge_filter = dataset_cfgs["args"].get("edge_filter", None)
        
        
        # If format is 'json' and data_path is provided, load from JSON file
        if format_type == "json":
            # Handle multiple datasets in a split
            data[split] = _load_json_dataset(data_path, dataset_cfgs, edge=edge, edge_filter=edge_filter, **kwargs)
        else:
            # Normal dataset loading
            data[split] = get_datasets(dataset_cfgs, **kwargs)
    
    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        # Keep retain dataset accessible separately for trainers that need it
        retain_dataset = unlearn_splits.get("retain", None)
        # import IPython; IPython.embed()
        if "retain" in unlearn_splits:
            unlearn_dataset = ForgetRetainDataset(forget=unlearn_splits["forget"], retain=retain_dataset, anchor=anchor)
        else:
            unlearn_dataset = ForgetRetainDataset(forget=unlearn_splits["forget"], anchor=anchor)
        data["train"] = unlearn_dataset
        # Keep retain in data dict for separate access if needed
        if retain_dataset is not None:
            data["retain"] = retain_dataset
        # Remove forget from data dict (it's now in ForgetRetainDataset)
        if "forget" in data:
            data.pop("forget")
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(QAwithAlternateDataset)
_register_data(QAEdgeDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
