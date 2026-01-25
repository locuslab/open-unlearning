"""
Trajectory-based metrics for dLLM unlearning evaluation.

This module computes metrics at each diffusion step across three trajectory types
(steps, fixation, ratio), supporting both logit-based and text-based metrics.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    tokenwise_vocab_logprobs,
)
from evals.metrics.trajectory_utils import (
    stack_logits_history,
    compute_trajectories,
    extract_logits_at_step,
    decode_logits_to_text,
)
from evals.metrics.trajectory_adapters import (
    compute_logit_metric_at_step,
    compute_text_metric_at_step,
)
from evals.metrics import METRICS_REGISTRY

logger = logging.getLogger("evaluator")

# IGNORE_INDEX from data.utils
IGNORE_INDEX = -100


def _get_sampler_from_model(model) -> Optional[Any]:
    """Extract sampler from model (handles adapter wrapping)."""
    # Check if model is wrapped with DiffusionModelAdapter
    if hasattr(model, "sampler"):
        return model.sampler
    
    # Check if model has a model attribute (nested wrapping)
    if hasattr(model, "model"):
        if hasattr(model.model, "sampler"):
            return model.model.sampler
        # Check if model.model is the adapter
        if hasattr(model.model, "sampler"):
            return model.model.sampler
    
    return None


def _is_logit_based_metric(metric_name: str) -> bool:
    """Check if a metric is logit-based or text-based."""
    logit_based_metrics = {
        "probability",
        "exact_memorization",
        "extraction_strength",
    }
    return metric_name in logit_based_metrics


def _is_text_based_metric(metric_name: str) -> bool:
    """Check if a metric is text-based."""
    text_based_metrics = {
        "rouge",
        "classifier_prob",
    }
    return metric_name in text_based_metrics


@unlearning_metric(name="trajectory_metrics")
def trajectory_metrics(model, **kwargs):
    """
    Compute metrics along diffusion trajectories.
    
    This function:
    1. Generates text using the model's sampler (with return_logits=True)
    2. Extracts logits_history and fixation_steps from sampler output
    3. Computes three trajectory tensors (steps, fixation, ratio)
    4. For each trajectory type and step, computes specified metrics
    5. Returns results organized by trajectory, step, and metric
    
    Config structure:
    - metrics: list of metric names from registry to compute
    - trajectory_config: config for trajectory computation
      - logits_source: "sampler" (default) or "external"
      - return_logits: true  # Sampler config
      - return_fixation_steps: true  # Sampler config
    - data: dataset to evaluate on
    - collators: collator for batching
    - batch_size: batch size for evaluation
    - tokenizer: tokenizer for text processing
    - generation_args: args for text generation (for text-based metrics)
    """
    # Extract config
    metrics_to_compute = kwargs.get("metrics", [])
    trajectory_config = kwargs.get("trajectory_config", {})
    logits_source = trajectory_config.get("logits_source", "sampler")
    data = kwargs.get("data")
    collator = kwargs.get("collators")
    batch_size = kwargs.get("batch_size", 1)
    tokenizer = kwargs.get("tokenizer")
    generation_args = kwargs.get("generation_args", {})
    
    if not metrics_to_compute:
        raise ValueError("No metrics specified in config")
    
    if not tokenizer:
        raise ValueError("tokenizer is required for trajectory metrics")
    
    # Create dataloader
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    
    # Storage for results
    all_results = {}  # {sample_idx: {trajectories: {...}}}
    trajectory_names = ["steps", "fixation", "ratio"]
    
    # Get sampler from model
    sampler = _get_sampler_from_model(model)
    if sampler is None:
        raise ValueError(
            "Model does not have a sampler. Trajectory metrics require a diffusion model with sampler. "
            "Ensure model is wrapped with DiffusionModelAdapter or has accessible sampler."
        )
    
    # Process each batch
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        indices = batch.get("index", torch.arange(batch_idx * batch_size, 
                                                  (batch_idx + 1) * batch_size))
        
        B = input_ids.shape[0]
        
        # Prepare inputs for sampler (list of token sequences)
        prompts = []
        prompt_lens = []
        for i in range(B):
            # Extract prompt (non-ignored tokens)
            if labels is not None:
                # Find where labels start (first non-IGNORE_INDEX)
                label_mask = labels[i] != IGNORE_INDEX
                if label_mask.any():
                    prompt_end = label_mask.nonzero()[0][0].item()
                else:
                    prompt_end = input_ids.shape[1]
            else:
                prompt_end = input_ids.shape[1]
            
            prompt = input_ids[i, :prompt_end].cpu().tolist()
            prompts.append(prompt)
            prompt_lens.append(len(prompt))
        
        # Generate using sampler with logits tracking
        sampler_output = sampler.sample(
            inputs=prompts,
            config=None,  # Use default config
            return_dict=True,
            return_logits=True,
            **trajectory_config.get("sampler_kwargs", {}),
        )
        
        # Extract logits and fixation steps
        logits_history = sampler_output.logits_history
        fixation_steps = sampler_output.fixation_steps
        
        if logits_history is None or len(logits_history) == 0:
            logger.warning(f"Batch {batch_idx}: No logits_history returned from sampler")
            continue
        
        if fixation_steps is None:
            logger.warning(f"Batch {batch_idx}: No fixation_steps returned from sampler")
            continue
        
        # Stack logits into tensor R [V, L, S]
        R = stack_logits_history(logits_history)  # [V, L, S]
        V, L, S = R.shape
        
        # Extract fixation steps F [L] for each sample
        # fixation_steps is [B, T], we need [L] for generated region
        # For now, use first sample or average
        if fixation_steps.dim() == 2:
            # [B, T] -> take first sample and extract generated region
            F_full = fixation_steps[0]  # [T]
            # Extract only the generated portion (after max prompt length)
            max_prompt_len = max(prompt_lens)
            if F_full.shape[0] > max_prompt_len:
                F = F_full[max_prompt_len:max_prompt_len + L]  # [L]
            else:
                # If fixation_steps doesn't cover full length, pad or truncate
                F = F_full[:L] if F_full.shape[0] >= L else torch.cat([
                    F_full,
                    torch.full((L - F_full.shape[0],), S - 1, dtype=torch.long, device=F_full.device)
                ])
        else:
            raise ValueError(f"Unexpected fixation_steps shape: {fixation_steps.shape}")
        
        # Compute three trajectory tensors
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        trajectories = {
            "steps": T_steps,
            "fixation": T_fixation,
            "ratio": T_ratio,
        }
        
        # Process each sample in batch
        for sample_idx in range(B):
            idx_str = str(indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx])
            
            # For batched case, we'd need per-sample trajectories
            # For now, use shared trajectories (assuming single sample or first sample)
            sample_trajectories = trajectories if sample_idx == 0 else trajectories
            
            sample_results = {
                "trajectories": {
                    "steps": {},
                    "fixation": {},
                    "ratio": {},
                }
            }
            
            # Get ground truth for this sample
            sample_labels = labels[sample_idx] if labels is not None else None
            sample_input_ids = input_ids[sample_idx]
            sample_prompt_len = prompt_lens[sample_idx]
            
            # Create batch template for logit metrics
            batch_template = {
                "input_ids": sample_input_ids.unsqueeze(0),
                "labels": sample_labels.unsqueeze(0) if sample_labels is not None else None,
                "attention_mask": attention_mask[sample_idx].unsqueeze(0) if attention_mask is not None else None,
            }
            
            # Compute metrics for each trajectory type and step
            for traj_name, trajectory in sample_trajectories.items():
                for step in range(S):
                    step_key = f"step_{step}"
                    step_results = {}
                    
                    # Extract logits at this step
                    logits = extract_logits_at_step(trajectory, step)  # [V, L]
                    
                    # Compute each requested metric
                    for metric_name in metrics_to_compute:
                        try:
                            if _is_logit_based_metric(metric_name):
                                # Get metric function from registry or utils
                                if metric_name == "probability":
                                    metric_fn = evaluate_probability
                                elif metric_name == "exact_memorization":
                                    # Use tokenwise_vocab_logprobs helper
                                    def _exact_mem_fn(model, batch, **kwargs):
                                        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
                                            model, batch, grad=False, return_labels=True
                                        )
                                        if len(log_probs_batch) == 0 or len(labels_batch) == 0:
                                            return [{"score": None}]
                                        log_probs = log_probs_batch[0]
                                        labels = labels_batch[0]
                                        if len(labels) == 0:
                                            return [{"score": None}]
                                        preds = torch.argmax(log_probs, dim=-1)
                                        em_score = (preds == labels).sum() / len(labels)
                                        return [{"score": em_score.item()}]
                                    metric_fn = _exact_mem_fn
                                else:
                                    logger.warning(f"Unknown logit-based metric: {metric_name}")
                                    continue
                                
                                result = compute_logit_metric_at_step(
                                    metric_fn, logits, batch_template
                                )
                                
                                # Extract metric value from result
                                if isinstance(result, list) and len(result) > 0:
                                    result_dict = result[0]
                                    # Extract the main metric value
                                    if "prob" in result_dict:
                                        step_results[metric_name] = result_dict["prob"]
                                    elif "score" in result_dict:
                                        step_results[metric_name] = result_dict["score"]
                                    else:
                                        # Use first numeric value
                                        for key, value in result_dict.items():
                                            if isinstance(value, (int, float)):
                                                step_results[metric_name] = value
                                                break
                            
                            elif _is_text_based_metric(metric_name):
                                # Decode logits to text
                                texts = decode_logits_to_text(
                                    logits, tokenizer, sample_input_ids, sample_prompt_len
                                )
                                
                                # Get ground truth text
                                if sample_labels is not None:
                                    # Extract ground truth from labels
                                    gt_tokens = sample_labels[sample_labels != IGNORE_INDEX]
                                    gt_text = tokenizer.decode(gt_tokens.tolist(), skip_special_tokens=True)
                                    ground_truths = [gt_text]
                                else:
                                    ground_truths = [""]  # No ground truth available
                                
                                if metric_name == "rouge":
                                    # Compute ROUGE directly from decoded text
                                    try:
                                        from rouge_score import rouge_scorer
                                    except ImportError:
                                        logger.warning("rouge_score not available, skipping ROUGE metric")
                                        continue
                                    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
                                    
                                    gen_text = texts[0] if texts else ""
                                    gt_text = ground_truths[0] if ground_truths else ""
                                    
                                    rouge_scores = scorer.score(gt_text, gen_text)
                                    
                                    # Extract ROUGE metric (default to rougeL_recall)
                                    rouge_type = kwargs.get("rouge_type", "rougeL_recall")
                                    if rouge_type == "rouge1_recall":
                                        step_results[metric_name] = rouge_scores["rouge1"].recall
                                    elif rouge_type == "rougeL_f1":
                                        step_results[metric_name] = rouge_scores["rougeL"].fmeasure
                                    elif rouge_type == "rougeL_recall":
                                        step_results[metric_name] = rouge_scores["rougeL"].recall
                                    else:
                                        # Default to rougeL_recall
                                        step_results[metric_name] = rouge_scores["rougeL"].recall
                                
                                elif metric_name == "classifier_prob":
                                    # For classifier_prob, we need the text in the expected format
                                    # This metric expects pre_compute with text, so we'll need to handle it differently
                                    # For now, log a warning
                                    logger.warning(
                                        f"classifier_prob metric requires pre_compute setup. "
                                        f"Skipping at step {step} for {traj_name}."
                                    )
                                else:
                                    logger.warning(f"Text-based metric {metric_name} not yet fully implemented")
                            
                            else:
                                logger.warning(f"Unknown metric type: {metric_name}")
                        
                        except Exception as e:
                            logger.warning(f"Error computing {metric_name} at step {step} for {traj_name}: {e}")
                            step_results[metric_name] = None
                    
                    sample_results["trajectories"][traj_name][step_key] = step_results
            
            all_results[idx_str] = sample_results
    
    # Aggregate results
    agg_value = {}
    for traj_name in trajectory_names:
        agg_value[traj_name] = {}
        for metric_name in metrics_to_compute:
            # Collect values per step across all samples
            step_values = {}  # {step: [values across samples]}
            
            for sample_idx, sample_results in all_results.items():
                traj_results = sample_results["trajectories"][traj_name]
                for step_key, step_results in traj_results.items():
                    if metric_name in step_results and step_results[metric_name] is not None:
                        # Extract step number from step_key (e.g., "step_5" -> 5)
                        step_num = int(step_key.split("_")[1])
                        if step_num not in step_values:
                            step_values[step_num] = []
                        step_values[step_num].append(step_results[metric_name])
            
            # Aggregate: mean across samples for each step
            if step_values:
                max_step = max(step_values.keys())
                aggregated = []
                for step in range(max_step + 1):
                    if step in step_values:
                        aggregated.append(np.mean(step_values[step]))
                    else:
                        aggregated.append(np.nan)
                agg_value[traj_name][metric_name] = np.array(aggregated)
            else:
                agg_value[traj_name][metric_name] = np.array([])
    
    return {
        "agg_value": agg_value,
        "value_by_index": all_results,
    }
