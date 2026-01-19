"""
Lunar is a unlearning trainer that uses the Lunar algorithm to unlearn the model.
https://github.com/facebookresearch/LUNAR/blob/main/run_lunar.py

Multi-GPU Support:
    This implementation supports multi-GPU training with DeepSpeed and DDP.
    Note: The activation extraction and estimated network training currently run
    on the main process only, as these operations are lightweight. The main model
    inference during activation extraction uses the distributed model.
"""

import copy
import json
import os
import warnings
import logging
import contextlib
import functools
from evals.train_eval import TrainDatasetEvaluator
import torch
import torch.nn.functional as F
from itertools import chain
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Tuple, Callable
import torch.optim as optim
from tqdm import tqdm
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from trainer.unlearn.grad_diff import GradDiff
from accelerate.utils import is_deepspeed_available

logger = logging.getLogger(__name__)

if is_deepspeed_available():
    import deepspeed

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


def _extract_text_from_item(item, tokenizer=None):
    """Extract text string from a dataset item.
    
    Args:
        item: Can be a string, dict with 'input_ids', or dict with text keys
        tokenizer: Tokenizer used to decode input_ids if needed
        
    Returns:
        str: The text string
    """
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        # Try common text keys first
        for key in ['instruction', 'question', 'text', 'prompt']:
            if key in item and isinstance(item[key], str):
                return item[key]
        # If no text key found, try to decode input_ids
        if 'input_ids' in item and tokenizer is not None:
            input_ids = item['input_ids']
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            return tokenizer.decode(input_ids, skip_special_tokens=False)
        # Fallback: try to find any string value
        for value in item.values():
            if isinstance(value, str):
                return value
    # If all else fails, convert to string
    return str(item)


def get_post_block_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size, device, tokenizer=None
):
    """
    Collect post-block activations. Uses single GPU only (GPU 0) for activation extraction.
    
    Args:
        model: Model to extract activations from (can be unwrapped)
        input_data: List of input items
        tokenize_instructions_fn: Function to tokenize instructions
        layer_idx: Index of the layer to extract activations from
        batch_size: Batch size for processing
        device: Device parameter (ignored, uses GPU 0 instead)
        tokenizer: Tokenizer for text extraction
    """
    # Use single GPU (GPU 0) for activation extraction
    if torch.cuda.is_available():
        single_gpu_device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        single_gpu_device = torch.device("cpu")
    
    instructions = input_data

    activations = []
    
    # Access the layer module for hook registration
    # For DeepSpeed/DDP, model might be wrapped, but we can still access via model.model
    layer_module = model.model.layers[layer_idx]
    fwd_hooks = [
        (layer_module, get_activations_fwd_hook(cache=activations))
    ]

    # Get tokenizer if not provided
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None) or getattr(model, 'get_tokenizer', lambda: None)()

    # Check if model is already on the desired device
    # For Flash Attention 2.0, moving models after initialization triggers warnings
    # Only move if necessary to avoid the warning
    if next(model.parameters()).device != single_gpu_device:
        model_on_single_gpu = model.to(single_gpu_device)
    else:
        model_on_single_gpu = model
    
    # Set model to eval mode and disable gradients for inference
    model_on_single_gpu.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size), desc="Extracting post-block activations"):
            batch_items = instructions[i : i + batch_size]
            # Extract text strings from batch items
            batch_texts = [_extract_text_from_item(item, tokenizer) for item in batch_items]
            inputs = tokenize_instructions_fn(instructions=batch_texts)

            # Move inputs to single GPU device
            input_ids = inputs.input_ids.to(single_gpu_device)
            attention_mask = inputs.attention_mask.to(single_gpu_device)

            with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
                # Use model on single GPU for inference
                model_on_single_gpu(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
    return activations


def get_pre_down_proj_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size, device, tokenizer=None
):
    """
    Collect pre-down-proj activations. Uses single GPU only (GPU 0) for activation extraction.
    
    Args:
        model: Model to extract activations from (can be unwrapped)
        input_data: List of input items
        tokenize_instructions_fn: Function to tokenize instructions
        layer_idx: Index of the layer to extract activations from
        batch_size: Batch size for processing
        device: Device parameter (ignored, uses GPU 0 instead)
        tokenizer: Tokenizer for text extraction
    """
    # Use single GPU (GPU 0) for activation extraction
    if torch.cuda.is_available():
        single_gpu_device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        single_gpu_device = torch.device("cpu")
    
    instructions = input_data

    activations = []
    
    # Access the module for hook registration
    down_proj_module = model.model.layers[layer_idx].mlp.down_proj
    pre_hooks = [
        (down_proj_module, get_activations_pre_hook(cache=activations))
    ]

    # Get tokenizer if not provided
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None) or getattr(model, 'get_tokenizer', lambda: None)()

    # Check if model is already on the desired device
    # For Flash Attention 2.0, moving models after initialization triggers warnings
    # Only move if necessary to avoid the warning
    if next(model.parameters()).device != single_gpu_device:
        model_on_single_gpu = model.to(single_gpu_device)
    else:
        model_on_single_gpu = model
    
    # Set model to eval mode and disable gradients for inference
    model_on_single_gpu.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size), desc="Extracting pre-down-proj activations"):
            batch_items = instructions[i : i + batch_size]
            # Extract text strings from batch items
            batch_texts = [_extract_text_from_item(item, tokenizer) for item in batch_items]
            inputs = tokenize_instructions_fn(instructions=batch_texts)

            # Move inputs to single GPU device
            input_ids = inputs.input_ids.to(single_gpu_device)
            attention_mask = inputs.attention_mask.to(single_gpu_device)

            with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
                # Use model on single GPU for inference
                model_on_single_gpu(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

    return activations


def get_pre_post_attention_layernorm_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size, device, tokenizer=None
):
    """
    Collect pre-post-attention-layernorm activations. Uses single GPU only (GPU 0) for activation extraction.
    
    Args:
        model: Model to extract activations from (can be unwrapped)
        input_data: List of input items
        tokenize_instructions_fn: Function to tokenize instructions
        layer_idx: Index of the layer to extract activations from
        batch_size: Batch size for processing
        device: Device parameter (ignored, uses GPU 0 instead)
        tokenizer: Tokenizer for text extraction
    """
    # Use single GPU (GPU 0) for activation extraction
    if torch.cuda.is_available():
        single_gpu_device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        single_gpu_device = torch.device("cpu")
    
    instructions = input_data

    activations = []
    
    # Access the module for hook registration
    layernorm_module = model.model.layers[layer_idx].post_attention_layernorm
    pre_hooks = [
        (layernorm_module, get_activations_pre_hook(cache=activations))
    ]

    # Get tokenizer if not provided
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None) or getattr(model, 'get_tokenizer', lambda: None)()

    # Check if model is already on the desired device
    # For Flash Attention 2.0, moving models after initialization triggers warnings
    # Only move if necessary to avoid the warning
    if next(model.parameters()).device != single_gpu_device:
        model_on_single_gpu = model.to(single_gpu_device)
    else:
        model_on_single_gpu = model
    
    # Set model to eval mode and disable gradients for inference
    model_on_single_gpu.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size), desc="Extracting pre-post-attention-layernorm activations"):
            batch_items = instructions[i : i + batch_size]
            # Extract text strings from batch items
            batch_texts = [_extract_text_from_item(item, tokenizer) for item in batch_items]
            inputs = tokenize_instructions_fn(instructions=batch_texts)
            
            # Move inputs to single GPU device
            input_ids = inputs.input_ids.to(single_gpu_device)
            attention_mask = inputs.attention_mask.to(single_gpu_device)
            
            with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
                # Use model on single GPU for inference
                model_on_single_gpu(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
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
        tokenizer=tokenizer,
    )

    post_block_activation_remain = get_post_block_activation(
        model=model,
        input_data=retain_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
        device=device,
        tokenizer=tokenizer,
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
            tokenizer=tokenizer,
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
            tokenizer=tokenizer,
        )
    )

    pre_down_proj_activation_forget = get_pre_down_proj_activation(
        model=model,
        input_data=forget_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
        device=device,
        tokenizer=tokenizer,
    )

    pre_down_proj_activation_remain = get_pre_down_proj_activation(
        model=model,
        input_data=retain_dataset,
        tokenize_instructions_fn=tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
        device=device,
        tokenizer=tokenizer,
    )

    return (
        post_block_activation_forget,
        post_block_activation_remain,
        pre_post_attention_layernorm_activation_forget,
        pre_post_attention_layernorm_activation_remain,
        pre_down_proj_activation_forget,
        pre_down_proj_activation_remain,
    )


def compute_direction_from_activations(
    harmful_activations, harmless_activations
):
    """
    Compute direction vector from harmful and harmless activations.
    
    The direction is computed as the difference between mean harmful and harmless
    activations. This direction vector is used to guide unlearning by perturbing
    activations away from the harmful direction.
    
    Args:
        harmful_activations: List of activation tensors from harmful/forget data
        harmless_activations: List of activation tensors from harmless/retain data
        
    Returns:
        torch.Tensor: Direction vector with shape matching activations
    """
    if not harmful_activations or not harmless_activations:
        raise ValueError("Both harmful and harmless activations must be provided")
    
    # Concatenate all activations
    harmful_cat = torch.cat([act.squeeze(0) for act in harmful_activations], dim=0)
    harmless_cat = torch.cat([act.squeeze(0) for act in harmless_activations], dim=0)
    
    # Compute mean activations (across batch dimension)
    mean_harmful = harmful_cat.mean(dim=0, keepdim=True)  # [1, seq_len, hidden_size]
    mean_harmless = harmless_cat.mean(dim=0, keepdim=True)  # [1, seq_len, hidden_size]
    
    # Direction: difference between harmful and harmless
    # This points from harmless to harmful
    direction = mean_harmful - mean_harmless
    
    return direction


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

# NOTE: The following functions (load_dataset_to_get_direction, load_dataset_json,
# split_raw_dataset_for_forget) were used in the original LUNAR implementation
# for computing direction vectors. They are now integrated for automatic direction
# computation when direction is not provided.

def get_mean_activations_pre_hook(
    layer, cache: torch.Tensor, n_samples, positions: List[int]  # pyright: ignore[reportUndefinedVariable]
):
    def hook_fn(module, input):
        activation: torch.Tensor = (
            input[0].clone().to(cache.device)
        )
        batch_size, seq_len, d_model = activation.shape
        
        # Extract activations at specified positions for each sample in the batch
        # positions is a list like [-1] or [0, 5, 10] indicating which positions to extract
        for pos_idx, pos in enumerate(positions):
            # Convert negative positions to positive indices relative to sequence length
            if pos < 0:
                # For each sample, get the position relative to its sequence length
                # We need to extract the last token for each sample (handling variable lengths)
                # Since we don't have access to actual sequence lengths here, we use the full seq_len
                # In practice, -1 typically means the last token of the sequence
                actual_pos = seq_len + pos
            else:
                actual_pos = pos
            
            # Extract activations at this position for all samples in batch
            # Shape: (batch_size, d_model)
            activations_at_pos = activation[:, actual_pos, :]
            
            # Sum over batch dimension and accumulate into cache
            # Divide by n_samples to compute mean across all samples
            cache[pos_idx, layer] += (1.0 / n_samples) * activations_at_pos.sum(dim=0)

    return hook_fn


def get_mean_activations(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
):
    # Synchronize and clear cache safely
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore errors if cache is already clear

    # Get device and dtype from model parameters (for multi-GPU and dtype compatibility)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # Use model's dtype instead of float64 to avoid dtype mismatch errors
    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model), dtype=dtype, device=device
    )

    fwd_pre_hooks = [
        (
            block_modules[layer],
            get_mean_activations_pre_hook(
                layer=layer,
                cache=mean_activations,
                n_samples=n_samples,
                positions=positions,
            ),
        )
        for layer in range(n_layers)
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        # Ensure all input tensors are on the same device as the model
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            # Synchronize after each batch to prevent errors
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return mean_activations


def get_mean_diff(
    model,
    tokenizer,
    harmful_instructions,
    harmless_instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
):
    
    # import IPython; IPython.embed()
    mean_activations_harmful = get_mean_activations(
        model,
        tokenizer,
        harmful_instructions,
        tokenize_instructions_fn,
        block_modules,
        batch_size=batch_size,
        positions=positions,
    )
    
    mean_activations_harmless = get_mean_activations(
        model,
        tokenizer,
        harmless_instructions,
        tokenize_instructions_fn,
        block_modules,
        batch_size=batch_size,
        positions=positions,
    )

    mean_diff = (
        mean_activations_harmful - mean_activations_harmless
    )

    return mean_diff


def generate_directions(
    model_base,
    harmful_instructions,
    harmless_instructions,
    artifact_dir=None,
):

    mean_diffs = get_mean_diff(
        model_base.model,
        model_base.tokenizer,
        harmful_instructions,
        harmless_instructions,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        positions=list(range(-len(model_base.eoi_toks), 0)),
    )

    assert mean_diffs.shape == (
        len(model_base.eoi_toks),
        model_base.model.config.num_hidden_layers,
        model_base.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()

    return mean_diffs
    

def generate_candidate_directions(model_base, harmful_train, forget_train):
    """Generate and save candidate directions."""

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        forget_train,
    )

    return mean_diffs

def load_dataset_to_get_direction(forget_dataset, data_path, instructions_only=True, use_harmful=True, use_unverified=False):
    # Load the harmful dataset
    if use_harmful:
        harmful_file_path = os.path.join(f"{data_path}/splits", "harmful.json")
        print(f'loading harmful dataset from {harmful_file_path}')
        with open(harmful_file_path, "r") as f:
            harmful_dataset = json.load(f)
    elif use_unverified:
        unverified_file_path = os.path.join(f"{data_path}/splits", "unverified.json")
        print(f'loading unverified dataset from {unverified_file_path}')
        with open(unverified_file_path, "r") as f:
            harmful_dataset = json.load(f)

    # Extract text from forget_dataset (can be Dataset object or list)
    if instructions_only:
        # Handle different dataset formats
        forget_train = []
        
        # Check if it's a Dataset object with a .data attribute (like QAEdgeDataset)
        if hasattr(forget_dataset, 'data') and hasattr(forget_dataset.data, '__len__'):
            # Access the underlying HuggingFace Dataset
            question_key = getattr(forget_dataset, 'question_key', 'question')
            for i in range(len(forget_dataset.data)):
                item = forget_dataset.data[i]
                # Try to extract text - check for question_key first (QA datasets)
                if isinstance(item, dict):
                    text = item.get(question_key) or item.get("instruction") or item.get("question") or item.get("text")
                    if text:
                        forget_train.append(text)
                    else:
                        # Fallback: use _extract_text_from_item
                        forget_train.append(_extract_text_from_item(item))
                elif isinstance(item, str):
                    forget_train.append(item)
                else:
                    forget_train.append(str(item))
        elif hasattr(forget_dataset, '__len__') and hasattr(forget_dataset, '__getitem__'):
            # If it's a Dataset object without .data, use __getitem__
            question_key = getattr(forget_dataset, 'question_key', 'question')
            for i in range(len(forget_dataset)):
                item = forget_dataset[i]
                # Try to extract text using helper function or check common keys
                if isinstance(item, dict):
                    # Try question_key first (for QA datasets), then instruction
                    text = item.get(question_key) or item.get("instruction") or item.get("question") or item.get("text")
                    if text:
                        forget_train.append(text)
                    else:
                        # Fallback: use _extract_text_from_item
                        forget_train.append(_extract_text_from_item(item))
                elif isinstance(item, str):
                    forget_train.append(item)
                else:
                    forget_train.append(str(item))
        else:
            # If it's already a list
            for d in forget_dataset:
                if isinstance(d, dict):
                    text = d.get("instruction") or d.get("question") or d.get("text")
                    if text:
                        forget_train.append(text)
                    else:
                        forget_train.append(_extract_text_from_item(d))
                elif isinstance(d, str):
                    forget_train.append(d)
                else:
                    forget_train.append(str(d))
        
        # Extract text from harmful_dataset (should be a list from JSON)
        harmful_train = []
        for d in harmful_dataset:
            if isinstance(d, dict):
                text = d.get("instruction") or d.get("question") or d.get("text")
                if text:
                    harmful_train.append(text)
                else:
                    harmful_train.append(_extract_text_from_item(d))
            elif isinstance(d, str):
                harmful_train.append(d)
            else:
                harmful_train.append(str(d))
    else:
        # Return full items if not instructions_only
        forget_train = list(forget_dataset) if hasattr(forget_dataset, '__len__') else forget_dataset
        harmful_train = harmful_dataset

    return harmful_train, forget_train


# def load_dataset_to_get_direction(
#     cfg, data_path, instructions_only=True, use_harmful=True, use_unverified=False
# ):
#     # Load the forget dataset as harmless
#     with open(data_path, "r") as f:
#         dataset = json.load(f)

#     # Change the key 'question' to 'instruction'
#     for d in dataset:
#         d["instruction"] = d.pop("question")

#     # Split into 'forget' and 'retain' based on the 'edge' key
#     forget_dataset = [d for d in dataset if d["edge"] in cfg.forget_edge]

#     # Load the harmful dataset
#     if use_harmful:
#         harmful_file_path = os.path.join("dataset/splits", "harmful.json")
#         print(f'loading harmful dataset from {harmful_file_path}')
#         with open(harmful_file_path, "r") as f:
#             harmful_dataset = json.load(f)
#     elif use_unverified:
#         unverified_file_path = os.path.join("dataset/splits", "unverified.json")
#         print(f'loading unverified dataset from {unverified_file_path}')
#         with open(unverified_file_path, "r") as f:
#             harmful_dataset = json.load(f)

#     if instructions_only:
#         forget_train = [d["instruction"] for d in forget_dataset]
#         harmful_train = [d["instruction"] for d in harmful_dataset]

#     return harmful_train, forget_train


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
    device):
    """
    Prepare training data for estimated networks.
    
    Args:
        compute_direction: If True and direction is empty, compute direction from activations
    """
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
        use_harmful=True,
        use_unverified=False,
        retain_dataset=None,
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
        self.data_path = kwargs.pop("data_path", "/home/judy/code/open-unlearning-dev/saves/dataset")
        self.use_harmful = use_harmful
        self.use_unverified = use_unverified
        self.retain_dataset = retain_dataset
        
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
    
    def compute_direction(self):
        """
        Compute direction vectors from forget and retain dataset activations.
        
        This method extracts activations from both forget and retain datasets,
        then computes direction vectors as the difference between mean forget
        and retain activations for each layer in layer_idx_list.
        
        The computed direction vectors are stored in self.direction and will
        be used during training to perturb forget activations.
        
        Note: This should be called before training if you want to manually
        compute directions. Otherwise, directions are computed automatically
        during prepare_trainset when compute_direction=True (default).
        """
        # Unwrap model to access underlying model (needed for DeepSpeed/DDP)
        unwrapped_model = self._unwrap_model(self.model)
        
        # Check that train_dataset has forget attribute
        if not hasattr(self.train_dataset, 'forget'):
            raise ValueError(
                "train_dataset must have a 'forget' attribute. "
                "Make sure the dataset is structured correctly for unlearning."
            )
        
        # Prepare training dataset
        harmful_train, forget_train = load_dataset_to_get_direction(
            forget_dataset=self.train_dataset.forget,
            data_path=self.data_path,
            instructions_only=True,
            use_harmful=self.use_harmful,
            use_unverified=self.use_unverified,
        )
        
        # Validate that we got data
        if not harmful_train or not forget_train:
            raise ValueError(
                "Failed to load training data. "
                f"harmful_train length: {len(harmful_train) if harmful_train else 0}, "
                f"forget_train length: {len(forget_train) if forget_train else 0}"
            )
        
        # Create tokenize_instructions_fn
        def tokenize_instructions_fn(instructions):
            """Tokenize a list of instruction strings."""
            if isinstance(instructions, str):
                instructions = [instructions]
            # Tokenize with padding and truncation
            inputs = self.tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            return inputs
        
        # Get model block modules (transformer blocks)
        # For most models, this is model.model.layers or model.transformer.h
        if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
            model_block_modules = unwrapped_model.model.layers
        elif hasattr(unwrapped_model, 'transformer') and hasattr(unwrapped_model.transformer, 'h'):
            model_block_modules = unwrapped_model.transformer.h
        elif hasattr(unwrapped_model, 'gpt_neox') and hasattr(unwrapped_model.gpt_neox, 'layers'):
            model_block_modules = unwrapped_model.gpt_neox.layers
        else:
            raise ValueError(f"Could not find transformer blocks in model. Available attributes: {dir(unwrapped_model)}")
        
        # Get end-of-instruction tokens (typically EOS token)
        eoi_toks = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        if not eoi_toks:
            # Fallback: use pad token or a default token
            eoi_toks = [self.tokenizer.pad_token_id] if self.tokenizer.pad_token_id is not None else [0]
        
        # Create a model_base-like object
        class ModelBaseWrapper:
            def __init__(self, model, tokenizer, tokenize_fn, block_modules, eoi_tokens):
                self.model = model
                self.tokenizer = tokenizer
                self.tokenize_instructions_fn = tokenize_fn
                self.model_block_modules = block_modules
                self.eoi_toks = eoi_tokens
        
        model_base = ModelBaseWrapper(
            model=unwrapped_model,
            tokenizer=self.tokenizer,
            tokenize_fn=tokenize_instructions_fn,
            block_modules=model_block_modules,
            eoi_tokens=eoi_toks,
        )
        
        # Generate candidate directions
        candidate_directions = generate_candidate_directions(
            model_base=model_base, harmful_train=harmful_train, forget_train=forget_train
        )  # Shape: [n_pos, n_layer, d_model]

        # Extract directions for specified positions and layers
        # Get positions from config, default to [-1] if not specified
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'positions'):
            positions = self.cfg.positions
        else:
            positions = [-1]  # Default to last position
        
        # Ensure positions is a list
        if not isinstance(positions, list):
            positions = [positions]
        
        direction = []
        n_layers = candidate_directions.shape[1]
        n_positions = candidate_directions.shape[0]
        
        for layer_index in self.layer_idx_list:
            # For each position, extract the direction vector
            # +1 because direction is calculated using pre-hook (original logic)
            layer_idx_in_candidates = layer_index + 1
            
            # Bounds checking for layer index
            if layer_idx_in_candidates >= n_layers:
                warnings.warn(
                    f"Layer index {layer_index}+1={layer_idx_in_candidates} is out of bounds "
                    f"(max: {n_layers-1}). Using last layer {n_layers-1} instead."
                )
                layer_idx_in_candidates = n_layers - 1
            elif layer_idx_in_candidates < 0:
                warnings.warn(
                    f"Layer index {layer_index}+1={layer_idx_in_candidates} is negative. "
                    f"Using layer 0 instead."
                )
                layer_idx_in_candidates = 0
            
            layer_directions = []
            for pos in positions:
                # Find the index of this position in candidate_directions
                # candidate_directions shape: [n_pos, n_layer, d_model]
                # where n_pos corresponds to the positions used in generate_directions
                # (typically positions from -len(eoi_toks) to -1)
                if pos < 0:
                    # Negative position: map to index in candidate_directions
                    # e.g., if eoi_toks has 1 token and pos=-1, it's at index 0
                    pos_idx_in_candidates = len(eoi_toks) + pos
                else:
                    # Positive position: not typically used, but handle it
                    pos_idx_in_candidates = pos
                
                # Ensure index is valid
                if 0 <= pos_idx_in_candidates < n_positions:
                    layer_directions.append(candidate_directions[pos_idx_in_candidates, layer_idx_in_candidates, :])
                else:
                    # Fallback: use the last position if index is out of bounds
                    warnings.warn(
                        f"Position index {pos_idx_in_candidates} (from pos={pos}) is out of bounds "
                        f"(max: {n_positions-1}). Using last position."
                    )
                    layer_directions.append(candidate_directions[-1, layer_idx_in_candidates, :])
            
            # If multiple positions, average them; otherwise use the single direction
            if len(layer_directions) == 1:
                direction.append(layer_directions[0])
            else:
                # Average directions across positions
                direction.append(torch.stack(layer_directions).mean(dim=0))
        
        self.direction = direction

    def _unwrap_model(self, model):
        """Unwrap model from DeepSpeed/DDP wrapper."""
        # Check if DeepSpeed is enabled
        is_deepspeed = (self.accelerator.state.deepspeed_plugin is not None) if hasattr(self.accelerator.state, 'deepspeed_plugin') else False
        
        if is_deepspeed and is_deepspeed_available():
            if isinstance(model, deepspeed.DeepSpeedEngine):
                return model.module
        # Handle DDP wrapped models
        if hasattr(model, 'module'):
            return model.module
        return model

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

        # Get device - use accelerator's device
        device = self.accelerator.device
        
        # Unwrap model to access underlying model (needed for DeepSpeed/DDP)
        unwrapped_model = self._unwrap_model(self.model)

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
        self.compute_direction()
        # Only print on main process to avoid duplicate output
        # if self.accelerator.is_main_process:
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

        # Prepare training data - run activation extraction
        # For multi-GPU, we can either run on main process only or all processes
        # For now, run on main process only to avoid redundancy
        if self.accelerator.is_main_process:
            # For activation extraction:
            # - Use unwrapped_model to access modules for hooks (needed for hook registration)
            # - The unwrapped model can be used for inference on main process
            # - For DeepSpeed ZeRO-3, parameters might be sharded, but on main process
            #   we can still use unwrapped model if we're only running on rank 0
            # Note: In distributed setups, each process might have a shard of parameters
            # but for inference on main process only, unwrapped_model works
            (
                forget_input_list,
                forget_target_list,
                remain_input_list,
                remain_target_list,
                _,
            ) = prepare_trainset(
                layer_idx_list=self.layer_idx_list,
                model=unwrapped_model,  # Use unwrapped model for hook access
                tokenizer=self.tokenizer,
                tokenize_instructions_fn=tokenize_instructions_fn,
                forget_dataset=forget_dataset,
                retain_dataset=retain_dataset,
                direction=self.direction,
                coeff_list=self.coeff_list,
                device=device
            )

            # Initialize estimated nets
            estimated_net_list = prepare_estimated_net_list(
                device=device,
                layer_idx_list=self.layer_idx_list,
                model=unwrapped_model,  # Use unwrapped model
                init_model_list=None,
            )
        else:
            # Other processes create empty lists (will be synced if needed)
            forget_input_list = []
            forget_target_list = []
            remain_input_list = []
            remain_target_list = []
            estimated_net_list = []
        
        # Wait for main process to finish activation extraction
        self.accelerator.wait_for_everyone()
        
        # Note: Lunar's custom training procedure currently runs on main process only
        # The activation extraction and estimated network training are lightweight
        # operations that don't benefit significantly from multi-GPU parallelism.
        # The main model inference during activation extraction uses the distributed
        # model, but the estimated network training runs on CPU/single GPU.
        
        # Only run training on main process
        if self.accelerator.is_main_process:
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

            # Update model weights - need to handle DeepSpeed/DDP wrapped models
            with torch.no_grad():
                for i, layer_idx in enumerate(self.layer_idx_list):
                    # Update the unwrapped model
                    new_weight = updated_estimated_net_list[i].down_proj.weight.data
                    unwrapped_model.model.layers[layer_idx].mlp.down_proj.weight.data.copy_(new_weight)
            
            # Sync model weights across all processes for DeepSpeed/DDP
            if self.accelerator.num_processes > 1:
                is_deepspeed = (self.accelerator.state.deepspeed_plugin is not None) if hasattr(self.accelerator.state, 'deepspeed_plugin') else False
                if is_deepspeed:
                    # DeepSpeed handles synchronization through its engine
                    # No explicit sync needed as DeepSpeed manages distributed state
                    pass
                else:
                    # For DDP, broadcast the updated weights to all processes
                    # Use torch.distributed for broadcasting
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        for i, layer_idx in enumerate(self.layer_idx_list):
                            weight = unwrapped_model.model.layers[layer_idx].mlp.down_proj.weight.data
                            # Broadcast from rank 0 to all other ranks
                            torch.distributed.broadcast(weight, src=0, async_op=False)

        # Wait for main process to finish
        self.accelerator.wait_for_everyone()

        # Save model if needed - only on main process
        # Note: save_model is also overridden below to handle external calls from train.py
        if hasattr(self.args, "output_dir") and self.args.output_dir:
            if self.accelerator.is_main_process:
                self._safe_save_model(self.args.output_dir)
        
        # evaluate on forget dataset and retain dataset separately
        print("Evaluating on forget dataset and retain dataset separately")
        self.evaluate()

    def _safe_save_model(self, output_dir):
        """Helper method to save model with error handling for deepspeed_config AttributeError."""
        try:
            # Try standard save_model (works for most cases)
            super().save_model(output_dir)
        except AttributeError as e:
            # Handle case where accelerator.get_state_dict() tries to access deepspeed_config
            # that doesn't exist (e.g., when using DDP without DeepSpeed)
            if "deepspeed_config" in str(e):
                # Save unwrapped model directly
                unwrapped_model = self._unwrap_model(self.model)
                unwrapped_model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            else:
                raise

    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to handle deepspeed_config AttributeError.
        
        This handles both calls from within train() and external calls from train.py.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        if output_dir:
            if self.accelerator.is_main_process:
                self._safe_save_model(output_dir)

    def save_state(self):
        """Override save_state to handle deepspeed_config AttributeError.
        
        save_state() internally uses accelerator.get_state_dict() which may try
        to access deepspeed_config that doesn't exist in non-DeepSpeed setups.
        """
        try:
            # Try standard save_state (works for most cases)
            super().save_state()
        except AttributeError as e:
            # Handle case where accelerator.get_state_dict() tries to access deepspeed_config
            if "deepspeed_config" in str(e):
                # For Lunar, we don't use standard training state, so skip saving state
                # if there's an accelerator issue (activation extraction doesn't need state saving)
                if self.accelerator.is_main_process:
                    print("Warning: Could not save trainer state due to accelerator configuration. "
                          "This is expected for Lunar trainer with DDP/accelerate without DeepSpeed.")
            else:
                raise
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None):
        # evaluate on forget dataset and retain dataset separately
        # create two evaluators for forget and retain datasets
        
        # Only run evaluation on main process
        if not self.accelerator.is_local_main_process:
            return {}
        
        # Check if we can run evaluation (need single process or evaluators configured)
        if self.accelerator.num_processes > 1:
            logger.warning(
                "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
            )
            return {}
        
        # Get forget and retain datasets
        forget_dataset = self._extract_forget_dataset()
        retain_dataset = self._extract_retain_dataset()
        
        if not forget_dataset and not retain_dataset:
            logger.warning("No forget or retain datasets available for evaluation.")
            return {}
        
        # Get output directory for evaluation results
        run_dir = self._get_output_dir(trial=trial)
        if hasattr(self.state, 'global_step'):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        else:
            checkpoint_folder = "final"
        output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
        os.makedirs(output_dir, exist_ok=True)
        
        eval_metrics = {}
        
        # Check if evaluators are configured
        if self.evaluators and len(self.evaluators) > 0:
            # Use the first evaluator's config as a template
            # We'll create TrainDatasetEvaluator instances for forget and retain
            first_evaluator = list(self.evaluators.values())[0]
            eval_cfg = first_evaluator.eval_cfg
            
            # Create evaluators for forget and retain datasets
            if forget_dataset:
                forget_evaluator = TrainDatasetEvaluator(eval_cfg=eval_cfg)
                forget_evaluator.name = "Custom"
                forget_evaluator.set_dataset(forget_dataset)
                forget_output_dir = os.path.join(output_dir, "forget")
                os.makedirs(forget_output_dir, exist_ok=True)
                
                forget_metrics = forget_evaluator.evaluate(
                    dataset=forget_dataset,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    template_args=self.template_args,
                    output_dir=forget_output_dir,
                )
                # Add prefix to metrics
                for key, value in forget_metrics.items():
                    eval_metrics[f"{metric_key_prefix}_forget_{key}"] = value
            
            if retain_dataset:
                retain_evaluator = TrainDatasetEvaluator(eval_cfg=eval_cfg)
                retain_evaluator.set_dataset(retain_dataset)
                retain_output_dir = os.path.join(output_dir, "retain")
                os.makedirs(retain_output_dir, exist_ok=True)
                
                retain_metrics = retain_evaluator.evaluate(
                    dataset=retain_dataset,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    template_args=self.template_args,
                    output_dir=retain_output_dir,
                )
                # Add prefix to metrics
                for key, value in retain_metrics.items():
                    eval_metrics[f"{metric_key_prefix}_retain_{key}"] = value
        else:
            # If no evaluators configured, log a warning
            logger.warning(
                "No evaluators configured. Skipping evaluation on forget/retain datasets. "
                "Configure evaluators in the training config to enable evaluation."
            )
        
        # Log metrics if any were computed
        if eval_metrics:
            self.log(eval_metrics)
        
        return eval_metrics


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
        else:
            return self.retain_dataset