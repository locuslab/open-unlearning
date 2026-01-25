#!/usr/bin/env python3
"""
Simple test script for trajectory metrics.

This script can be run locally to test trajectory metrics without K8s.
It uses a small subset of data and reduced steps for quick testing.

Usage:
    cd /workspaces/dllm/open-unlearning
    python scripts/test_trajectory_metrics.py
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import torch
import logging
from omegaconf import OmegaConf

from model import get_model  # noqa: E402
from evals import get_evaluators  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_trajectory_metrics():
    """Test trajectory metrics with a simple setup."""
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create minimal config for testing
    config = OmegaConf.create({
        "model": {
            "model_args": {
                "pretrained_model_name_or_path": "gpt2",  # Small model for testing
                "device_map": device,
            },
            "tokenizer_args": {
                "pretrained_model_name_or_path": "gpt2",
            },
            "template_args": None,
        },
        "eval": {
            "trajectory_test": {
                "handler": "MUSEEvaluator",
                "output_dir": "saves/eval/trajectory_test",
                "metrics": {
                    "trajectory_metrics": {
                        "handler": "trajectory_metrics",
                        "batch_size": 1,
                        "metrics": ["probability"],  # Start with one metric
                        "trajectory_config": {
                            "logits_source": "sampler",
                            "return_logits": True,
                            "return_fixation_steps": True,
                            "sampler_kwargs": {
                                "steps": 16,  # Small for testing
                                "temperature": 0.0,
                                "max_new_tokens": 32,  # Small for testing
                            },
                        },
                    },
                },
                "overwrite": True,
                "data_split": "News",
            },
        },
        "data": {
            "dataset": "muse",
            "data_split": "News",
            "max_samples": 2,  # Just 2 samples for quick test
        },
        "collator": {
            "handler": "default",
        },
    })
    
    logger.info("Loading model...")
    model, tokenizer = get_model(config.model)
    
    # Check if model has sampler (for diffusion models)
    sampler = None
    if hasattr(model, "sampler"):
        sampler = model.sampler
    elif hasattr(model, "model") and hasattr(model.model, "sampler"):
        sampler = model.model.sampler
    
    if sampler is None:
        logger.warning(
            "Model does not have a sampler. Trajectory metrics require a diffusion model. "
            "For testing with a regular model, you would need to wrap it with DiffusionModelAdapter."
        )
        logger.info("Skipping trajectory metrics test (requires diffusion model)")
        return
    
    logger.info("Getting evaluators...")
    evaluators = get_evaluators(config.eval)
    
    logger.info("Running trajectory metrics evaluation...")
    for evaluator_name, evaluator in evaluators.items():
        logger.info(f"Running evaluator: {evaluator_name}")
        eval_args = {
            "model": model,
            "tokenizer": tokenizer,
            "template_args": None,
        }
        try:
            results = evaluator.evaluate(**eval_args)
            logger.info("Evaluation completed!")
            logger.info("Results keys: %s", list(results.keys()) if results else "None")
            if results and "trajectory_metrics" in results:
                traj_results = results["trajectory_metrics"]
                if "agg_value" in traj_results:
                    logger.info(f"Aggregated values available for trajectories: {list(traj_results['agg_value'].keys())}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    test_trajectory_metrics()
