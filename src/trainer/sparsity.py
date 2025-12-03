"""
Sparsity Methods for Large Language Models
===========================================

Multiple sparsity/pruning strategies optimized for transformer models.
These methods are memory-efficient and suitable for models with billions of parameters.

Methods included:
1. Layer-wise magnitude pruning (memory efficient)
2. Random pruning (baseline)
3. Structured pruning (attention heads, FFN neurons)
4. Gradual magnitude pruning (iterative)
5. Movement pruning (data-driven)
6. Magnitude with sampling (fastest)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SparsityManager:
    """Manages different sparsity strategies for LLMs."""

    @staticmethod
    def create_mask(
        model: nn.Module,
        method: str = "layerwise_magnitude",
        sparsity: float = 0.9,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Create sparsity mask using specified method.

        Args:
            model: The model to create mask for
            method: One of ['layerwise_magnitude', 'random', 'structured',
                           'gradual', 'movement', 'magnitude_sampling']
            sparsity: Target sparsity ratio (0.9 = 90% zeros)
            device: Device to place masks on
            **kwargs: Method-specific parameters

        Returns:
            mask_dict: Dictionary mapping parameter names to binary masks
        """
        method_map = {
            'layerwise_magnitude': SparsityManager._layerwise_magnitude,
            'random': SparsityManager._random_pruning,
            'structured': SparsityManager._structured_pruning,
            'gradual': SparsityManager._gradual_magnitude,
            'movement': SparsityManager._movement_pruning,
            'magnitude_sampling': SparsityManager._magnitude_with_sampling,
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")

        logger.info(f"Creating sparsity mask using: {method}")
        logger.info(f"Target sparsity: {sparsity:.1%}")

        mask_dict = method_map[method](model, sparsity, device, **kwargs)

        # Compute statistics
        total_params = sum(m.numel() for m in mask_dict.values())
        total_zero = sum((m == 0).sum().item() for m in mask_dict.values())
        actual_sparsity = total_zero / total_params

        logger.info(f"Mask created: {actual_sparsity:.1%} overall sparsity")

        return mask_dict

    # =========================================================================
    # Method 1: Layer-wise Magnitude Pruning (RECOMMENDED for LLMs)
    # =========================================================================

    @staticmethod
    def _layerwise_magnitude(
        model: nn.Module,
        sparsity: float,
        device: str,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply magnitude pruning independently to each layer.

        Most memory-efficient: processes one layer at a time.
        Good for LLMs because it preserves per-layer capacity.
        """
        mask_dict = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                # Skip frozen parameters
                mask_dict[name] = torch.ones_like(param.data).to(device)
                continue

            # Only prune weight matrices, keep biases
            if 'weight' in name and len(param.shape) >= 2:
                # Compute threshold for this layer
                param_flat = param.data.abs().reshape(-1)

                if param_flat.numel() > 10_000_000:  # >10M elements
                    # Sample for very large layers
                    sample_size = min(1_000_000, param_flat.numel())
                    indices = torch.randperm(param_flat.numel(), device=device)[:sample_size]
                    sampled = param_flat[indices]
                    threshold = torch.quantile(sampled, sparsity)
                else:
                    threshold = torch.quantile(param_flat, sparsity)

                mask = (param.data.abs() >= threshold).float()
            else:
                # Keep biases, LayerNorm, embeddings
                mask = torch.ones_like(param.data)

            mask_dict[name] = mask.to(device)

        return mask_dict

    # =========================================================================
    # Method 2: Magnitude with Sampling (FASTEST for huge models)
    # =========================================================================

    @staticmethod
    def _magnitude_with_sampling(
        model: nn.Module,
        sparsity: float,
        device: str,
        sample_ratio: float = 0.1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Use sampling to estimate global threshold, then apply layer-wise.

        Ultra-fast for billion-parameter models.
        Trade-off: slightly less accurate threshold estimation.
        """
        logger.info(f"Using sampling ratio: {sample_ratio:.1%}")

        # Collect samples from all layers
        samples = []

        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2 and param.requires_grad:
                param_flat = param.data.abs().reshape(-1)
                n_samples = max(1000, int(param_flat.numel() * sample_ratio))
                n_samples = min(n_samples, param_flat.numel())

                indices = torch.randperm(param_flat.numel(), device=device)[:n_samples]
                samples.append(param_flat[indices])

        # Compute global threshold from samples
        if samples:
            all_samples = torch.cat(samples)
            threshold = torch.quantile(all_samples, sparsity)
            logger.info(f"Global threshold: {threshold:.6f}")
        else:
            threshold = 0.0

        # Apply threshold to create masks
        mask_dict = {}
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2 and param.requires_grad:
                mask = (param.data.abs() >= threshold).float()
            else:
                mask = torch.ones_like(param.data)

            mask_dict[name] = mask.to(device)

        return mask_dict

    # =========================================================================
    # Method 3: Random Pruning (BASELINE)
    # =========================================================================

    @staticmethod
    def _random_pruning(
        model: nn.Module,
        sparsity: float,
        device: str,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Random pruning - useful baseline.

        No computation needed, but worse performance than magnitude.
        """
        mask_dict = {}

        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2 and param.requires_grad:
                # Random binary mask
                mask = (torch.rand_like(param.data) > sparsity).float()
            else:
                mask = torch.ones_like(param.data)

            mask_dict[name] = mask.to(device)

        return mask_dict

    # =========================================================================
    # Method 4: Structured Pruning (HEAD/NEURON level)
    # =========================================================================

    @staticmethod
    def _structured_pruning(
        model: nn.Module,
        sparsity: float,
        device: str,
        granularity: str = "neuron",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Structured pruning: prune entire attention heads or FFN neurons.

        Better hardware efficiency than unstructured pruning.
        """
        logger.info(f"Granularity: {granularity}")

        mask_dict = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                mask_dict[name] = torch.ones_like(param.data).to(device)
                continue

            # Apply structured pruning to linear layers
            if 'weight' in name and len(param.shape) == 2:
                if granularity == "neuron":
                    # Prune output neurons (rows)
                    importance = param.data.abs().mean(dim=1)  # [out_features]
                    n_keep = int(len(importance) * (1 - sparsity))
                    _, keep_indices = torch.topk(importance, n_keep)

                    mask = torch.zeros_like(param.data)
                    mask[keep_indices, :] = 1.0

                elif granularity == "row":
                    # Prune entire rows
                    importance = param.data.abs().sum(dim=1)
                    n_keep = int(len(importance) * (1 - sparsity))
                    _, keep_indices = torch.topk(importance, n_keep)

                    mask = torch.zeros_like(param.data)
                    mask[keep_indices, :] = 1.0

                else:
                    # Default: unstructured
                    threshold = torch.quantile(param.data.abs().reshape(-1), sparsity)
                    mask = (param.data.abs() >= threshold).float()
            else:
                mask = torch.ones_like(param.data)

            mask_dict[name] = mask.to(device)

        return mask_dict

    # =========================================================================
    # Method 5: Gradual Magnitude Pruning
    # =========================================================================

    @staticmethod
    def _gradual_magnitude(
        model: nn.Module,
        sparsity: float,
        device: str,
        current_step: int = 0,
        total_steps: int = 100,
        initial_sparsity: float = 0.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Gradually increase sparsity over time.

        More stable than one-shot pruning.
        Good for iterative unlearning.
        """
        # Cubic schedule: s_t = s_f + (s_i - s_f) * (1 - t/T)^3
        t = min(current_step / total_steps, 1.0)
        current_sparsity = sparsity + (initial_sparsity - sparsity) * (1 - t) ** 3

        logger.info(f"Step {current_step}/{total_steps}: {current_sparsity:.1%} sparsity")

        # Use layerwise magnitude with current sparsity
        return SparsityManager._layerwise_magnitude(model, current_sparsity, device)

    # =========================================================================
    # Method 6: Movement Pruning (Data-aware)
    # =========================================================================

    @staticmethod
    def _movement_pruning(
        model: nn.Module,
        sparsity: float,
        device: str,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Movement pruning: prune weights that don't move much during training.

        Requires gradient information.
        Better than magnitude for fine-tuning scenarios.
        """
        if gradients is None:
            logger.warning("No gradients provided, falling back to magnitude")
            return SparsityManager._layerwise_magnitude(model, sparsity, device)

        mask_dict = {}

        for name, param in model.named_parameters():
            if name in gradients and 'weight' in name and len(param.shape) >= 2:
                # Movement score: |weight * gradient|
                movement = (param.data.abs() * gradients[name].abs())
                threshold = torch.quantile(movement.reshape(-1), sparsity)
                mask = (movement >= threshold).float()
            else:
                mask = torch.ones_like(param.data)

            mask_dict[name] = mask.to(device)

        return mask_dict

    # =========================================================================
    # Utility Functions
    # =========================================================================

    @staticmethod
    def apply_mask(model: nn.Module, mask_dict: Dict[str, torch.Tensor]):
        """Apply mask to model parameters (in-place)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask_dict:
                    param.data.mul_(mask_dict[name])

    @staticmethod
    def print_sparsity_stats(mask_dict: Dict[str, torch.Tensor]):
        """Print detailed sparsity statistics."""
        logger.info("\nSparsity Statistics:")
        logger.info("-" * 60)

        for name, mask in mask_dict.items():
            if mask.numel() > 1000:  # Only print for substantial layers
                sparsity = (mask == 0).sum().item() / mask.numel()
                logger.info(f"  {name:40s}: {sparsity:6.1%} sparse")

        total = sum(m.numel() for m in mask_dict.values())
        total_zero = sum((m == 0).sum().item() for m in mask_dict.values())
        logger.info("-" * 60)
        logger.info(f"  Overall: {total_zero/total:6.1%} sparse ({total_zero:,} / {total:,})")
