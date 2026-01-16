"""
Lunar is a unlearning trainer that uses the Lunar algorithm to unlearn the model.
https://github.com/facebookresearch/LUNAR/blob/main/run_lunar.py
"""

import copy
import json
import os
import contextlib
import functools
import torch
import torch.nn.functional as F
from itertools import chain
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Tuple, Callable
import torch.optim as optim
from tqdm import tqdm

from trainer.unlearn.grad_diff import GradDiff


# ============================================================================
# Lunar Utility Classes and Functions
# ============================================================================

# Copyright (c) Meta Platforms, Inc. and affiliates.


class EstimatedNet(torch.nn.Module):
    def __init__(
        self, in_features, out_features, bias, original_down_proj_weight, if_mask=False
    ):
        super(EstimatedNet, self).__init__()
        # Define the layers with the same dimensions as the original MLP
        self.down_proj = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.original_down_proj_weight = original_down_proj_weight
        self.if_mask = if_mask
        # Initialize the weights randomly (default initialization in PyTorch)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():  # Disable gradient tracking while setting weights
            self.down_proj.weight.copy_(self.original_down_proj_weight)

    def forward(self, x, mask=None):
        # Forward pass based on the given architecture
        output = self.down_proj(x)
        if self.if_mask:
            output = output * mask
        return output


class LUNAR_LoRA_net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, rank, pretrained_weight=None):
        super(LUNAR_LoRA_net, self).__init__()

        # Define the original linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

        # Initialize the linear layer's weight with the pretrained weight if provided
        if pretrained_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_weight)

        # Freeze the original linear layer's weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Define LoRA parameters (low-rank adaptation matrices)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(output_dim, rank) * 0.01)

    def forward(self, x):
        # Original forward pass
        base_output = self.linear(x)

        # LoRA adaptation
        lora_output = x @ self.lora_A.T @ self.lora_B.T

        return base_output + lora_output

    def merge_weights(self):
        # Merge the LoRA weights into the linear layer's weight
        with torch.no_grad():
            merged_weight = self.linear.weight + (self.lora_B @ self.lora_A)
            self.linear.weight.copy_(merged_weight)

        # After merging, LoRA parameters can optionally be deleted or frozen
        del self.lora_A
        del self.lora_B


class ActivationDataset_multiple_layers(Dataset):
    def __init__(self, inputs_list, targets_list):
        self.inputs_list = inputs_list
        self.targets_list = targets_list

    def __len__(self):
        return self.inputs_list[0].size(0)

    def __getitem__(self, idx):
        return [inputs[idx] for inputs in self.inputs_list], [
            targets[idx] for targets in self.targets_list
        ]


def train_multiple_layers(
    model_list, train_loader, optimizer, scheduler, device, num_epochs=100
):
    """
    Train multiple estimated networks on activations.
    """
    for model in model_list:
        model.train()

    criterion = torch.nn.MSELoss()
    print(f"Running optimizer for all models together......")
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(
            total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ) as pbar:
            # Loop through the batches of the training data
            for inputs_list, targets_list in train_loader:
                inputs_list = [input.to(device) for input in inputs_list]
                targets_list = [target.to(device) for target in targets_list]

                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs_list = [
                    model(inputs) for model, inputs in zip(model_list, inputs_list)
                ]
                loss = sum(
                    [
                        criterion(outputs, target)
                        for outputs, target in zip(outputs_list, targets_list)
                    ]
                )

                loss.backward()
                optimizer.step()

                # Keep track of the running loss for the current epoch
                running_loss += loss.item()

                # Update tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            if scheduler is not None:
                scheduler.step()

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print("Training completed!")
    return model_list


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs,
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activations_fwd_hook(cache):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activation = output.clone().detach()
        cache.append(activation)

    return hook_fn


def get_activations_pre_hook(cache):
    """Hook function to capture input activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input):
        if isinstance(input, tuple):
            input = input[0]
        activation = input.clone().detach()
        cache.append(activation)

    return hook_fn


def get_post_block_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size, device
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    fwd_hooks = [
        (model.model.layers[layer_idx], get_activations_fwd_hook(cache=activations))
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )
    return activations


def get_pre_down_proj_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size, device
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    pre_hooks = [
        (
            model.model.layers[layer_idx].mlp.down_proj,
            get_activations_pre_hook(cache=activations),
        )
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )

    return activations


def get_pre_post_attention_layernorm_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size, device
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    pre_hooks = [
        (
            model.model.layers[layer_idx].post_attention_layernorm,
            get_activations_pre_hook(cache=activations),
        )
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])
        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )

    return activations


def get_activations(
    model,
    tokenizer,
    tokenize_instructions_fn,
    layer_idx,
    forget_dataset,
    retain_dataset,
    device,
    batch_size_forget=1,
    batch_size_remain=1,
):
    post_block_activation_forget = get_post_block_activation(
        model=model,
        input_data=forget_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
        device=device,
    )

    post_block_activation_remain = get_post_block_activation(
        model=model,
        input_data=retain_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
        device=device,
    )

    # get pre post attention layernorm activations
    pre_post_attention_layernorm_activation_forget = (
        get_pre_post_attention_layernorm_activation(
            model=model,
            input_data=forget_dataset,
            tokenize_instructions_fn=tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size_forget,
            device=device,
        )
    )

    pre_post_attention_layernorm_activation_remain = (
        get_pre_post_attention_layernorm_activation(
            model=model,
            input_data=retain_dataset,
            tokenize_instructions_fn=tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size_remain,
            device=device,
        )
    )

    pre_down_proj_activation_forget = get_pre_down_proj_activation(
        model=model,
        input_data=forget_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
        device=device,
    )

    pre_down_proj_activation_remain = get_pre_down_proj_activation(
        model=model,
        input_data=retain_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
        device=device,
    )

    return (
        post_block_activation_forget,
        post_block_activation_remain,
        pre_post_attention_layernorm_activation_forget,
        pre_post_attention_layernorm_activation_remain,
        pre_down_proj_activation_forget,
        pre_down_proj_activation_remain,
    )


def perturb_post_block_activations_forget(
    post_block_activation_forget, direction, coeff=+2.0
):
    """perturb the output_activations of forget data. but only perturb the last token"""

    ### post_block_activation_forget: list of n_sample tensors [1, seq_length, d_model]
    for i in range(len(post_block_activation_forget)):
        post_block_activation_forget[i] += coeff * direction

    return post_block_activation_forget


# ============================================================================
# Lunar Dataset and Training Utilities
# ============================================================================


def load_dataset_to_get_direction(
    cfg, data_path, instructions_only=True, use_harmful=True, use_unverified=False
):
    # Load the forget dataset as harmless
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Change the key 'question' to 'instruction'
    for d in dataset:
        d["instruction"] = d.pop("question")

    # Split into 'forget' and 'retain' based on the 'edge' key
    forget_dataset = [d for d in dataset if d["edge"] in cfg.forget_edge]

    # Load the harmful dataset
    if use_harmful:
        harmful_file_path = os.path.join("dataset/splits", "harmful.json")
        print(f'loading harmful dataset from {harmful_file_path}')
        with open(harmful_file_path, "r") as f:
            harmful_dataset = json.load(f)
    elif use_unverified:
        unverified_file_path = os.path.join("dataset/splits", "unverified.json")
        print(f'loading unverified dataset from {unverified_file_path}')
        with open(unverified_file_path, "r") as f:
            harmful_dataset = json.load(f)

    if instructions_only:
        forget_train = [d["instruction"] for d in forget_dataset]
        harmful_train = [d["instruction"] for d in harmful_dataset]

    return harmful_train, forget_train


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
    """
    Split dataset into forget and retain datasets.
    Note: torch_reformat functionality is not fully implemented as it requires QuestionsDataset
    which may not be available in this codebase.
    """
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Split into 'forget' and 'retain' based on the 'edge' key
    forget_dataset = [
        d for d in dataset if d["edge"] in forget_edge
    ]  # because we want to unlearn edge one by one
    if cfg.use_different_retain_dataset:
        with open(cfg.different_retain_set_path, "r") as f:
            dataset = json.load(f)
        retain_dataset = [d for d in dataset]  # use the whole dataset as retain dataset
    else:
        retain_dataset = [d for d in dataset if d["edge"] not in cfg.forget_edge]
    if instructions_only:
        forget_dataset = [d["question"] for d in forget_dataset]
        retain_dataset = [d["question"] for d in retain_dataset]
    else:
        if torch_reformat:
            raise NotImplementedError(
                "torch_reformat requires QuestionsDataset which is not available in this codebase"
            )

    return forget_dataset, retain_dataset


def prepare_trainset_raw(
    pre_down_proj_activation_forget,
    post_block_activation_forget,
    pre_post_attention_layernorm_activation_forget,
    pre_down_proj_activation_remain,
    post_block_activation_remain,
    pre_post_attention_layernorm_activation_remain,
):
    # prepare the forget data
    inputs_forget = [item.detach() for item in pre_down_proj_activation_forget]
    post_mlp_activation_forget = [
        x - y
        for x, y in zip(
            post_block_activation_forget, pre_post_attention_layernorm_activation_forget
        )
    ]
    targets_forget = [item.detach() for item in post_mlp_activation_forget]

    # prepare the remain data
    pre_down_proj_activation_remain = [
        item.view(-1, pre_down_proj_activation_remain[0].size()[-1])
        for item in pre_down_proj_activation_remain
    ]
    post_block_activation_remain = [
        item.view(-1, post_block_activation_remain[0].size()[-1])
        for item in post_block_activation_remain
    ]
    pre_post_attention_layernorm_activation_remain = [
        item.view(-1, pre_post_attention_layernorm_activation_remain[0].size()[-1])
        for item in pre_post_attention_layernorm_activation_remain
    ]
    inputs_remain = torch.cat(pre_down_proj_activation_remain, dim=0)
    targets_remain = torch.cat(post_block_activation_remain, dim=0) - torch.cat(
        pre_post_attention_layernorm_activation_remain, dim=0
    )

    concat_forget_input = torch.cat(
        [activation.squeeze(0) for activation in inputs_forget], dim=0
    )
    concat_forget_target = torch.cat(
        [activation.squeeze(0) for activation in targets_forget], dim=0
    )

    return concat_forget_input, concat_forget_target, inputs_remain, targets_remain


def prepare_trainset(
    layer_idx_list,
    model,
    tokenizer,
    tokenize_instructions_fn,
    forget_dataset,
    retain_dataset,
    direction,
    coeff_list,
    device,
):
    # this is to prepare the data for training the estimated net
    # we need both forget dataset and remain dataset
    estimated_net_list = []
    forget_input_list = []
    forget_target_list = []
    remain_input_list = []
    remain_target_list = []

    # loop to get the input and target for each layer
    for i, layer_idx in enumerate(layer_idx_list):
        (
            post_block_activation_forget,
            post_block_activation_remain,
            pre_post_attention_layernorm_activation_forget,
            pre_post_attention_layernorm_activation_remain,
            pre_down_proj_activation_forget,
            pre_down_proj_activation_remain,
        ) = get_activations(
            model=model,
            tokenizer=tokenizer,
            tokenize_instructions_fn=tokenize_instructions_fn,
            layer_idx=layer_idx,
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
            device=device,
        )

        # perturb the post block activations for forget data
        post_block_activation_forget = perturb_post_block_activations_forget(
            post_block_activation_forget,
            direction[i],
            coeff=coeff_list[i],
        )

        # prepare the data for training estimated net
        concat_forget_input, concat_forget_target, input_remain, target_remain = (
            prepare_trainset_raw(
                pre_down_proj_activation_forget,
                post_block_activation_forget,
                pre_post_attention_layernorm_activation_forget,
                pre_down_proj_activation_remain,
                post_block_activation_remain,
                pre_post_attention_layernorm_activation_remain,
            )
        )

        forget_input_list.append(concat_forget_input)
        forget_target_list.append(concat_forget_target)
        remain_input_list.append(input_remain)
        remain_target_list.append(target_remain)

    return (
        forget_input_list,
        forget_target_list,
        remain_input_list,
        remain_target_list,
        estimated_net_list,
    )


def prepare_estimated_net_list(
    device, layer_idx_list, model, init_model_list=None
):
    estimated_net_list = []
    init_weight_list = []
    if init_model_list is None:
        print("initialize the estimated net list with the model base MLP weight")
        for layer_idx in layer_idx_list:
            init_weight_list.append(
                model.model.layers[layer_idx].mlp.down_proj.weight.clone()
            )
    else:
        print(
            "initialize the estimated net list with the previous unlearning MLP weight"
        )
        for estimante_model in init_model_list:
            init_weight_list.append(estimante_model.down_proj.weight.clone())

    for weight_parameter in init_weight_list:
        down_proj_in_features = weight_parameter.shape[1]
        down_proj_out_features = weight_parameter.shape[0]
        print(f"down_proj_in_features: {down_proj_in_features}")
        print(f"down_proj_out_features: {down_proj_out_features}")
        estimated_down_proj = EstimatedNet(
            in_features=down_proj_in_features,
            out_features=down_proj_out_features,
            bias=False,
            original_down_proj_weight=weight_parameter,
        ).to(device, dtype=torch.bfloat16)

        estimated_net_list.append(estimated_down_proj)
    return estimated_net_list


def prepare_estimated_net_lora_list(
    device, layer_idx_list, model, init_model_list=None
):
    estimated_net_list = []
    init_weight_list = []
    if init_model_list is None:
        print("initialize the estimated net list with the model base MLP weight")
        for layer_idx in layer_idx_list:
            init_weight_list.append(
                model.model.layers[layer_idx].mlp.down_proj.weight.clone()
            )
    else:
        print(
            "initialize the estimated net list with the previous unlearning MLP weight"
        )
        for estimante_model in init_model_list:
            init_weight_list.append(estimante_model.down_proj.weight.clone())

    for weight_parameter in init_weight_list:
        down_proj_in_features = weight_parameter.shape[1]
        down_proj_out_features = weight_parameter.shape[0]
        print(f"down_proj_in_features: {down_proj_in_features}")  # 11008
        print(f"down_proj_out_features: {down_proj_out_features}")  # 4096
        estimated_down_proj = LUNAR_LoRA_net(
            input_dim=down_proj_in_features,
            output_dim=down_proj_out_features,
            rank=8,
            pretrained_weight=weight_parameter,
        ).to(device, dtype=torch.bfloat16)

        estimated_net_list.append(estimated_down_proj)

    return estimated_net_list

class Lunar(GradDiff):
    """
    Lunar unlearning trainer that uses estimated networks to modify MLP layers.
    
    Note: This implementation requires additional configuration parameters:
    - layer_idx_list: List of layer indices to modify
    - direction: List of direction vectors for perturbation
    - coeff_list: List of coefficients for perturbation
    - lr: Learning rate for training estimated networks
    - num_epochs: Number of epochs for training estimated networks
    
    The Lunar algorithm works by:
    1. Extracting activations from forget/retain datasets
    2. Training estimated networks to approximate perturbed MLP layers
    3. Updating the model weights with trained estimated networks
    """

    def __init__(
        self,
        layer_idx_list=None,
        direction=None,
        coeff_list=None,
        lr=0.001,
        num_epochs=100,
        batch_size=64,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layer_idx_list = layer_idx_list or []
        self.direction = direction or []
        self.coeff_list = coeff_list or [2.0] * len(self.layer_idx_list)
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Ensure coeff_list matches layer_idx_list length
        if len(self.coeff_list) < len(self.layer_idx_list):
            self.coeff_list.extend(
                [2.0] * (len(self.layer_idx_list) - len(self.coeff_list))
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        For Lunar, the actual unlearning happens during training.
        This is a placeholder loss function for compatibility.
        """
        # Extract forget and retain inputs
        forget_inputs = inputs.get("forget", {})
        retain_inputs = inputs.get("retain", {})

        # Compute a simple loss for compatibility
        if forget_inputs and "labels" in forget_inputs:
            forget_outputs = model(**{k: forget_inputs[k] for k in ["input_ids", "attention_mask", "labels"]})
            loss = -forget_outputs.loss  # Negative loss for unlearning
        else:
            # Fallback dummy loss
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        return (loss, forget_outputs) if return_outputs and forget_inputs else loss

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        Override train to implement Lunar's custom training procedure.
        """
        # If layer_idx_list is empty, fall back to standard training
        if not self.layer_idx_list:
            return super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs,
            )

        # Get device
        device = self.accelerator.device

        # Get datasets
        if hasattr(self.train_dataset, "forget") and hasattr(self.train_dataset, "retain"):
            # Extract forget and retain datasets
            # Note: This assumes the dataset is structured appropriately
            forget_dataset = self._extract_forget_dataset()
            retain_dataset = self._extract_retain_dataset()
        else:
            # Fall back to standard training if datasets aren't structured correctly
            return super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs,
            )

        print(f"forget_dataset: {len(forget_dataset)}")
        print(f"retain_dataset: {len(retain_dataset)}")

        # Create tokenize function
        def tokenize_instructions_fn(instructions):
            return self.tokenizer(
                instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

        # Prepare training data
        (
            forget_input_list,
            forget_target_list,
            remain_input_list,
            remain_target_list,
            _,
        ) = prepare_trainset(
            layer_idx_list=self.layer_idx_list,
            model=self.model,
            tokenizer=self.tokenizer,
            tokenize_instructions_fn=tokenize_instructions_fn,
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
            direction=self.direction,
            coeff_list=self.coeff_list,
            device=device,
        )

        # Initialize estimated nets
        estimated_net_list = prepare_estimated_net_list(
            device=device,
            layer_idx_list=self.layer_idx_list,
            model=self.model,
            init_model_list=None,
        )

        # Prepare training dataset
        train_dataset_forget = ActivationDataset_multiple_layers(
            forget_input_list, forget_target_list
        )
        train_dataset_remain = ActivationDataset_multiple_layers(
            remain_input_list, remain_target_list
        )
        combined_dataset = ConcatDataset([train_dataset_forget, train_dataset_remain])
        train_loader = DataLoader(
            combined_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training
        optimizer = optim.AdamW(
            chain(*[model.parameters() for model in estimated_net_list]), lr=self.lr
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        updated_estimated_net_list = train_multiple_layers(
            estimated_net_list,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            num_epochs=self.num_epochs,
        )

        # Update model weights
        with torch.no_grad():
            for i, layer_idx in enumerate(self.layer_idx_list):
                self.model.model.layers[layer_idx].mlp.down_proj.weight.data = (
                    updated_estimated_net_list[i].down_proj.weight.data
                )

        # Save model if needed
        if hasattr(self.args, "output_dir") and self.args.output_dir:
            self.save_model(self.args.output_dir)

    def _extract_forget_dataset(self):
        """Extract forget dataset from train_dataset."""
        # This is a placeholder - actual implementation depends on dataset structure
        # For now, return empty list if dataset structure is not compatible
        if hasattr(self.train_dataset, "forget"):
            return list(self.train_dataset.forget)
        return []

    def _extract_retain_dataset(self):
        """Extract retain dataset from train_dataset."""
        # This is a placeholder - actual implementation depends on dataset structure
        if hasattr(self.train_dataset, "retain"):
            return list(self.train_dataset.retain)
        return []