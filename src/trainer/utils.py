import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import TrainerCallback
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs


# Memory optimization callback for periodic CUDA cache clearing
class CudaCacheCallback(TrainerCallback):
    """
    Callback to periodically clear CUDA cache during training to reduce memory fragmentation.
    This callback synchronizes all processes before clearing the cache to ensure consistency.

    Instead of creating its own Accelerator instance, this callback uses the trainer's
    Accelerator to avoid conflicts in distributed training scenarios.

    Args:
        interval (int): Number of steps between cache clearing operations. Default: 10
    """

    def __init__(self, interval=10):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        if state.global_step > 0 and state.global_step % self.interval == 0:
            # Get the trainer instance from kwargs to access its accelerator
            trainer = kwargs.get("trainer")
            if trainer is not None and hasattr(trainer, "accelerator"):
                # Use trainer's accelerator to synchronize processes
                trainer.accelerator.wait_for_everyone()
                # Clear CUDA cache on all devices
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Synchronize again after clearing
                trainer.accelerator.wait_for_everyone()
            else:
                # Fallback if trainer or accelerator is not available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for unlearning methods"""

    # Time metrics
    unlearning_time_seconds: float = 0.0  # Time for unlearning operation only

    # Memory metrics
    peak_gpu_memory_mb: float = 0.0
    intermediate_storage_mb: float = (
        0.0  # Temporary storage (gradients, checkpoints, etc.)
    )

    # Throughput metrics
    tokens_per_second: float = 0.0
    model_size_mb: float = 0.0
    total_steps: int = 0  # Number of training steps during unlearning

    # Computational metrics
    forward_passes_total: int = 0
    backward_passes_total: int = 0
    flops_estimate: float = 0.0  # Estimated FLOPs (if available)

    # Data requirements
    forget_samples_count: int = 0
    retain_samples_count: int = 0
    requires_retain_set: bool = False

    # GPU utilization (optional, requires pynvml)
    gpu_utilization_mean: float = 0.0
    gpu_utilization_max: float = 0.0

    # Latency metrics
    per_step_latency_mean_ms: float = 0.0
    per_step_latency_std_ms: float = 0.0
    per_step_latency_min_ms: float = 0.0
    per_step_latency_max_ms: float = 0.0

    # I/O metrics
    checkpoint_save_time_seconds: float = 0.0

    # LMCleaner-specific metrics (populated from audit records)
    lmcleaner_total_hvp_calls: int = 0
    lmcleaner_num_forget_steps: int = 0
    lmcleaner_K_configured: int = 0
    lmcleaner_K_used_mean: float = 0.0
    lmcleaner_K_used_max: int = 0
    lmcleaner_correction_norm_mean: float = 0.0
    lmcleaner_correction_norm_max: float = 0.0
    lmcleaner_per_forget_step_time_mean_ms: float = 0.0
    lmcleaner_per_forget_step_time_std_ms: float = 0.0
    lmcleaner_train_logs_size_gb: float = 0.0

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EfficiencyTracker(TrainerCallback):
    """Track computational and storage efficiency metrics during unlearning

    Note: This tracks UNLEARNING time specifically, not the initial training time.
    For unlearning benchmarks, we care about the cost of the unlearning operation itself.
    """

    def __init__(
        self,
        output_dir: str = None,
        storage_dirs: list = None,
        gpu_sampling_interval: int = 10,
        storage_check_interval: int = 10,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        # Normalize storage_dirs to Path objects once at init
        self.storage_dirs = [
            Path(d) if isinstance(d, str) else d for d in (storage_dirs or [])
        ]
        self.gpu_sampling_interval = gpu_sampling_interval
        self.storage_check_interval = storage_check_interval

        # Time tracking
        self.start_time = None
        self.last_step_time = None
        self.step_latencies = []  # Per-step latencies in ms

        # Memory tracking
        self.peak_memory = 0.0
        self.peak_storage = 0.0  # Peak intermediate storage in MB
        self._last_storage_check_step = None
        self._last_storage_value = 0.0

        # Pass counting
        self.forward_count = 0
        self.backward_count = 0
        self._forward_hook = None
        self._backward_hook = None

        # GPU utilization tracking (optional)
        self.gpu_utilizations = []
        self._nvml_initialized = False
        self._nvml_handle = None

        # Sample counts
        self.forget_samples = 0
        self.retain_samples = 0

        # Checkpoint timing
        self.checkpoint_save_times = []
        self._checkpoint_start_time = None

        # Try to initialize NVML for GPU utilization tracking
        self._try_init_nvml()

    def _try_init_nvml(self):
        """Try to initialize NVML for GPU utilization tracking"""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            # Get handle for first GPU (or current device)
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.info("NVML initialized for GPU utilization tracking")
        except (ImportError, Exception) as e:
            logger.debug(f"NVML not available for GPU tracking: {e}")
            self._nvml_initialized = False

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        if not self._nvml_initialized:
            return 0.0
        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    def _forward_hook_fn(self, module, input, output):
        """Hook to count forward passes"""
        self.forward_count += 1

    def _backward_hook_fn(self, module, grad_input, grad_output):
        """Hook to count backward passes"""
        self.backward_count += 1

    def _get_dir_size_mb(self, path: Path) -> float:
        """Get total size of directory in MB"""
        if not path.exists():
            return 0.0
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except (PermissionError, OSError):
                    # Ignore files that can't be accessed, continue scanning others
                    continue
        return total / (1024**2)

    def _get_intermediate_storage(self) -> float:
        """Calculate total intermediate storage across monitored directories"""
        total = 0.0
        for path in self.storage_dirs:
            total += self._get_dir_size_mb(path)
        return total

    def _get_sample_counts(self, trainer):
        """Extract forget and retain sample counts from dataset"""
        if trainer is None or not hasattr(trainer, "train_dataset"):
            return 0, 0

        dataset = trainer.train_dataset
        forget_count = 0
        retain_count = 0

        # Check for ForgetRetainDataset structure
        if hasattr(dataset, "forget") and hasattr(dataset, "retain"):
            forget_count = len(dataset.forget) if dataset.forget else 0
            retain_count = len(dataset.retain) if dataset.retain else 0
        elif hasattr(dataset, "forget_dataset") and hasattr(dataset, "retain_dataset"):
            forget_count = len(dataset.forget_dataset) if dataset.forget_dataset else 0
            retain_count = len(dataset.retain_dataset) if dataset.retain_dataset else 0
        else:
            # Single dataset, assume all are forget samples
            forget_count = len(dataset) if hasattr(dataset, "__len__") else 0

        return forget_count, retain_count

    def _estimate_flops(self, model, num_tokens: int) -> float:
        """Estimate FLOPs for transformer training.

        Uses the approximation: FLOPs ≈ 6 * P * T
        where P = number of parameters, T = number of tokens.

        This accounts for:
        - Forward pass: ~2PT (matrix multiplications)
        - Backward pass: ~4PT (gradients for weights and activations)

        Reference: Kaplan et al. "Scaling Laws for Neural Language Models"
        """
        if model is None or num_tokens <= 0:
            return 0.0

        num_params = sum(p.numel() for p in model.parameters())
        # 6 * P * T approximation for training FLOPs
        return 6.0 * num_params * num_tokens

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of unlearning training"""
        self.start_time = time.time()
        self.last_step_time = self.start_time

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Register hooks for forward/backward counting
        if model is not None:
            # Register on the model itself (counts full forward/backward passes)
            self._forward_hook = model.register_forward_hook(self._forward_hook_fn)
            try:
                self._backward_hook = model.register_full_backward_hook(
                    self._backward_hook_fn
                )
            except Exception:
                # Fallback for older PyTorch versions
                pass

        # Get sample counts
        trainer = kwargs.get("trainer")
        self.forget_samples, self.retain_samples = self._get_sample_counts(trainer)

    def on_step_end(self, args, state, control, **kwargs):
        """Update metrics after each step"""
        current_time = time.time()

        # Track per-step latency
        if self.last_step_time is not None:
            latency_ms = (current_time - self.last_step_time) * 1000
            self.step_latencies.append(latency_ms)
        self.last_step_time = current_time

        # Track peak GPU memory
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_memory = max(self.peak_memory, current_memory)

        # Track peak intermediate storage (throttled to avoid overhead)
        if self.storage_dirs:
            current_step = getattr(state, "global_step", 0)
            should_check = (
                self._last_storage_check_step is None
                or (current_step - self._last_storage_check_step)
                >= self.storage_check_interval
            )
            if should_check:
                self._last_storage_value = self._get_intermediate_storage()
                self._last_storage_check_step = current_step
            self.peak_storage = max(self.peak_storage, self._last_storage_value)

        # Sample GPU utilization periodically
        if self._nvml_initialized:
            current_step = getattr(state, "global_step", 0)
            if current_step % self.gpu_sampling_interval == 0:
                util = self._get_gpu_utilization()
                if util > 0:
                    self.gpu_utilizations.append(util)

    def on_save(self, args, state, control, **kwargs):
        """Track checkpoint save time"""
        if self._checkpoint_start_time is None:
            # Start timing
            self._checkpoint_start_time = time.time()
        else:
            # End timing (called after save completes)
            save_time = time.time() - self._checkpoint_start_time
            self.checkpoint_save_times.append(save_time)
            self._checkpoint_start_time = None

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Compute and save final unlearning efficiency metrics"""
        if not self.start_time:
            logger.warning(
                "Start time was not recorded, timing metrics may be inaccurate"
            )

        unlearning_time = time.time() - self.start_time if self.start_time else 0

        # Remove hooks
        if self._forward_hook is not None:
            self._forward_hook.remove()
        if self._backward_hook is not None:
            self._backward_hook.remove()

        # Shutdown NVML
        if self._nvml_initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception as e:
                logger.debug("Failed to shut down NVML: %s", e)

        # Use state.global_step for accurate step count
        total_steps = state.global_step if hasattr(state, "global_step") else 0

        # Estimate tokens processed during unlearning
        tokens_per_sec = 0
        num_tokens = 0
        if hasattr(state, "num_input_tokens_seen") and state.num_input_tokens_seen > 0:
            num_tokens = state.num_input_tokens_seen
            tokens_per_sec = num_tokens / unlearning_time if unlearning_time > 0 else 0

        # Get model size
        model_size_mb = 0
        if model is not None:
            model_size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024**2)

        # Estimate FLOPs
        flops = self._estimate_flops(model, num_tokens)

        # Determine if method requires retain set (check trainer type)
        requires_retain_set = False
        trainer = kwargs.get("trainer")
        if trainer is not None:
            trainer_name = trainer.__class__.__name__
            # Methods that use retain set for regularization
            retain_set_methods = ["GradDiff", "NPO", "DPO", "SimNPO", "RMU", "UNDIAL"]
            requires_retain_set = any(
                method in trainer_name for method in retain_set_methods
            )

        # Compute latency statistics
        latency_mean = 0.0
        latency_std = 0.0
        latency_min = 0.0
        latency_max = 0.0
        if self.step_latencies:
            latency_mean = np.mean(self.step_latencies)
            latency_std = np.std(self.step_latencies)
            latency_min = np.min(self.step_latencies)
            latency_max = np.max(self.step_latencies)

        # Compute GPU utilization statistics
        gpu_util_mean = 0.0
        gpu_util_max = 0.0
        if self.gpu_utilizations:
            gpu_util_mean = np.mean(self.gpu_utilizations)
            gpu_util_max = np.max(self.gpu_utilizations)

        # Compute checkpoint save time
        checkpoint_time = sum(self.checkpoint_save_times)

        metrics = EfficiencyMetrics(
            unlearning_time_seconds=unlearning_time,
            peak_gpu_memory_mb=self.peak_memory,
            intermediate_storage_mb=self.peak_storage,
            tokens_per_second=tokens_per_sec,
            model_size_mb=model_size_mb,
            total_steps=total_steps,
            forward_passes_total=self.forward_count,
            backward_passes_total=self.backward_count,
            flops_estimate=flops,
            forget_samples_count=self.forget_samples,
            retain_samples_count=self.retain_samples,
            requires_retain_set=requires_retain_set,
            gpu_utilization_mean=gpu_util_mean,
            gpu_utilization_max=gpu_util_max,
            per_step_latency_mean_ms=latency_mean,
            per_step_latency_std_ms=latency_std,
            per_step_latency_min_ms=latency_min,
            per_step_latency_max_ms=latency_max,
            checkpoint_save_time_seconds=checkpoint_time,
        )

        # Populate LMCleaner-specific metrics from audit records
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self._populate_lmcleaner_metrics(metrics, trainer)

        # Save metrics
        if self.output_dir:
            metrics.save(self.output_dir / "efficiency_metrics.json")

        # Format FLOPs for display
        if flops >= 1e15:
            flops_str = f"{flops / 1e15:.2f} PFLOPs"
        elif flops >= 1e12:
            flops_str = f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            flops_str = f"{flops / 1e9:.2f} GFLOPs"
        else:
            flops_str = f"{flops:.2e}"

        # Log efficiency metrics via project logger
        logger.info(
            "\n%s\n"
            "Unlearning Efficiency Metrics:\n"
            "  Unlearning Time: %.2fs\n"
            "  Total Steps: %d\n"
            "  Peak GPU Memory: %.2f MB\n"
            "  Intermediate Storage: %.2f MB\n"
            "  Tokens/Second: %.2f\n"
            "  Model Size: %.2f MB\n"
            "  Estimated FLOPs: %s\n"
            "  Forward Passes: %d\n"
            "  Backward Passes: %d\n"
            "  Forget Samples: %d\n"
            "  Retain Samples: %d\n"
            "  Requires Retain Set: %s\n"
            "  GPU Utilization (mean/max): %.1f%% / %.1f%%\n"
            "  Step Latency (mean/std): %.2f / %.2f ms\n"
            "  Checkpoint Save Time: %.2fs\n"
            "%s",
            "=" * 50,
            unlearning_time,
            total_steps,
            self.peak_memory,
            self.peak_storage,
            tokens_per_sec,
            model_size_mb,
            flops_str,
            self.forward_count,
            self.backward_count,
            self.forget_samples,
            self.retain_samples,
            requires_retain_set,
            gpu_util_mean,
            gpu_util_max,
            latency_mean,
            latency_std,
            checkpoint_time,
            "=" * 50,
        )


    def _populate_lmcleaner_metrics(self, metrics: EfficiencyMetrics, trainer):
        """Populate LMCleaner-specific fields from trainer's audit records."""
        audit_records = getattr(trainer, "audit_records", None)
        if not audit_records:
            return

        hvp_calls = []
        k_used = []
        v_norms = []
        step_times = []

        for rec in audit_records:
            r = dict(rec) if not isinstance(rec, dict) else rec
            hvp_calls.append(r.get("hvp_calls", 0))
            k_used.append(r.get("K_used", 0))
            v_norms.append(r.get("v_norm", 0.0))
            if "wall_time_ms" in r:
                step_times.append(r["wall_time_ms"])

        metrics.lmcleaner_total_hvp_calls = sum(hvp_calls)
        metrics.lmcleaner_num_forget_steps = len(audit_records)
        metrics.lmcleaner_K_configured = getattr(trainer, "K", 0)

        if k_used:
            metrics.lmcleaner_K_used_mean = float(np.mean(k_used))
            metrics.lmcleaner_K_used_max = int(np.max(k_used))
        if v_norms:
            metrics.lmcleaner_correction_norm_mean = float(np.mean(v_norms))
            metrics.lmcleaner_correction_norm_max = float(np.max(v_norms))
        if step_times:
            metrics.lmcleaner_per_forget_step_time_mean_ms = float(np.mean(step_times))
            metrics.lmcleaner_per_forget_step_time_std_ms = float(np.std(step_times))

        # Measure train_logs directory size
        train_log_dir = getattr(trainer, "train_log_dir", None)
        if train_log_dir is None:
            train_log_dir = getattr(trainer, "log_dir", None)
        if train_log_dir and Path(train_log_dir).exists():
            total_bytes = sum(
                f.stat().st_size for f in Path(train_log_dir).rglob("*") if f.is_file()
            )
            metrics.lmcleaner_train_logs_size_gb = total_bytes / (1024**3)


def compute_hvp_for_batch(
    model: nn.Module,
    batch: dict,
    v: torch.Tensor,
    params: list = None,
) -> torch.Tensor:
    """
    Compute Hessian-vector product using GGN approximation.

    H ≈ J^T J where J is the Jacobian of log-probabilities.
    HVP is computed as: J^T (J v)

    Args:
        model: The model
        batch: Input batch dict
        v: Vector to multiply with Hessian
        params: List of parameters (if None, uses all trainable params)

    Returns:
        Hessian-vector product
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()

    # Filter out non-model keys (e.g., "index" from DataCollatorWithIndex)
    model_keys = {"input_ids", "attention_mask", "labels", "token_type_ids", "position_ids"}
    model_batch = {k: v for k, v in batch.items() if k in model_keys}
    labels = batch.get("labels", batch.get("input_ids"))

    # Forward pass
    outputs = model(**model_batch)
    logits = outputs.logits

    # Compute log-softmax probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Flatten for Jacobian computation
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    labels_flat = labels.view(-1)

    # Select relevant log-probs (ignore padding)
    mask = labels_flat != -100
    if mask.sum() == 0:
        return torch.zeros_like(v)

    selected_log_probs = log_probs_flat[mask]
    selected_labels = labels_flat[mask]

    # Get log-prob of correct tokens
    correct_log_probs = selected_log_probs.gather(1, selected_labels.unsqueeze(1)).squeeze()

    # First backward: compute Jv
    # We need gradients of log_probs w.r.t. params, then dot with v
    param_shapes = [p.shape for p in params]
    param_numels = [p.numel() for p in params]

    # Reshape v to match params
    v_params = []
    offset = 0
    for numel, shape in zip(param_numels, param_shapes):
        v_params.append(v[offset:offset + numel].view(shape))
        offset += numel

    # Compute Jv via forward-mode AD approximation
    # We use the trick: Jv ≈ (f(θ + εv) - f(θ)) / ε for small ε
    eps = 1e-4

    # Save original params
    original_params = [p.data.clone() for p in params]

    try:
        # Perturb params: θ + εv
        with torch.no_grad():
            for p, vp in zip(params, v_params):
                p.data.add_(vp, alpha=eps)

        # Forward with perturbed params
        outputs_pert = model(**model_batch)
        log_probs_pert = F.log_softmax(outputs_pert.logits, dim=-1)
        log_probs_pert_flat = log_probs_pert.view(-1, log_probs_pert.size(-1))
        selected_log_probs_pert = log_probs_pert_flat[mask]
        correct_log_probs_pert = selected_log_probs_pert.gather(1, selected_labels.unsqueeze(1)).squeeze()

        # Jv ≈ (f(θ+εv) - f(θ)) / ε
        Jv = (correct_log_probs_pert - correct_log_probs) / eps

    finally:
        # Always restore original params
        with torch.no_grad():
            for p, orig in zip(params, original_params):
                p.data.copy_(orig)

    # Second step: compute J^T (Jv) via backward
    model.zero_grad()

    # Recompute forward
    outputs = model(**model_batch)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    selected_log_probs = log_probs_flat[mask]
    correct_log_probs = selected_log_probs.gather(1, selected_labels.unsqueeze(1)).squeeze()

    # Backward with Jv as grad_output
    correct_log_probs.backward(Jv.detach())

    # Collect gradients
    hvp_parts = []
    for p in params:
        if p.grad is not None:
            hvp_parts.append(p.grad.view(-1).clone())
        else:
            hvp_parts.append(torch.zeros(p.numel(), device=v.device, dtype=v.dtype))

    hvp = torch.cat(hvp_parts)
    model.zero_grad()

    return hvp


def estimate_spectral_norm_power_iter(
    model: nn.Module,
    batch: dict,
    eta: float,
    params: list = None,
    num_iters: int = 20,
    device: str = "cuda",
) -> tuple:
    """
    Estimate ||I - η*H||_2 using power iteration.

    Args:
        model: The model
        batch: Input batch
        eta: Learning rate
        params: List of parameters
        num_iters: Number of power iterations
        device: Device for computation

    Returns:
        (spectral_norm, lambda_max_estimate)
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in params)

    # Initialize random vector
    v = torch.randn(num_params, device=device, dtype=dtype)
    v = v / v.norm()

    model.eval()

    try:
        for _ in range(num_iters):
            # Compute Hv
            Hv = compute_hvp_for_batch(model, batch, v, params)

            # P = I - η*H, so Pv = v - η*Hv
            Pv = v - eta * Hv

            # Normalize
            Pv_norm = Pv.norm()
            if Pv_norm > 1e-10:
                v = Pv / Pv_norm

        # Final estimate
        Hv = compute_hvp_for_batch(model, batch, v, params)
        Pv = v - eta * Hv
        spectral_norm = Pv.norm().item()

        # Estimate λ_max of H
        lambda_est = Hv.norm().item()

        return spectral_norm, lambda_est

    except Exception as e:
        logger.warning(f"Error in spectral norm estimation: {e}")
        return None, None

    finally:
        model.train()


class SpectralNormCallback(TrainerCallback):
    """
    Callback to compute spectral norms of the propagation operator during training.

    Computes ||P^[t]||_2 = ||I - η*H^[t]||_2 periodically to verify contraction property.

    Args:
        interval: Number of steps between spectral norm computations
        num_power_iters: Number of power iterations for estimation
        output_dir: Directory to save spectral norm data
        enabled: Whether the callback is enabled
    """

    def __init__(
        self,
        interval: int = 100,
        num_power_iters: int = 20,
        output_dir: str = None,
        enabled: bool = True,
    ):
        self.interval = interval
        self.num_power_iters = num_power_iters
        self.output_dir = Path(output_dir) if output_dir else None
        self.enabled = enabled

        # Results storage
        self.spectral_norms = {}  # step -> spectral_norm
        self.lambda_maxs = {}  # step -> lambda_max estimate
        self.etas = {}  # step -> learning rate

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, train_dataloader=None, optimizer=None, **kwargs):
        """Compute spectral norm at specified intervals"""
        if not self.enabled:
            return

        if state.global_step == 0 or state.global_step % self.interval != 0:
            return

        if model is None:
            return

        # Get a batch from train_dataloader
        if train_dataloader is None:
            logger.debug("No train_dataloader available for spectral norm computation")
            return

        # Get learning rate from optimizer
        if optimizer is None:
            logger.debug("No optimizer available for spectral norm computation")
            return

        eta = optimizer.param_groups[0]["lr"]
        self.etas[state.global_step] = eta

        # Get trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]

        try:
            device = next(model.parameters()).device

            # Sample a random batch from the dataset directly to avoid
            # interfering with training's dataloader iteration order
            dataset = train_dataloader.dataset
            batch_size = train_dataloader.batch_size or 4

            # Random sample indices
            indices = torch.randint(0, len(dataset), (batch_size,)).tolist()

            # Collate samples into a batch
            samples = [dataset[i] for i in indices]
            if hasattr(train_dataloader, "collate_fn") and train_dataloader.collate_fn is not None:
                batch = train_dataloader.collate_fn(samples)
            else:
                # Default collation
                batch = {
                    k: torch.stack([s[k] for s in samples])
                    for k in samples[0].keys()
                    if torch.is_tensor(samples[0][k])
                }

            # Move batch to correct device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Compute spectral norm
            spectral_norm, lambda_max = estimate_spectral_norm_power_iter(
                model, batch, eta, params, self.num_power_iters, device
            )

            if spectral_norm is not None:
                self.spectral_norms[state.global_step] = spectral_norm
                self.lambda_maxs[state.global_step] = lambda_max

                # Log periodically
                logger.info(
                    f"Step {state.global_step}: ||P||_2 = {spectral_norm:.6f}, "
                    f"λ_max ≈ {lambda_max:.4f}, η = {eta:.2e}"
                )

                # Check contraction
                if spectral_norm >= 1.0:
                    logger.warning(
                        f"Non-contractive at step {state.global_step}: ||P||_2 = {spectral_norm:.6f}"
                    )

        except StopIteration:
            logger.warning(f"DataLoader exhausted at step {state.global_step}")
        except Exception as e:
            logger.warning(f"Failed to compute spectral norm at step {state.global_step}: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """Save results at end of training"""
        if not self.enabled or not self.spectral_norms:
            return

        results = {
            "spectral_norms": self.spectral_norms,
            "lambda_maxs": self.lambda_maxs,
            "etas": self.etas,
            "metadata": {
                "interval": self.interval,
                "num_power_iters": self.num_power_iters,
                "total_measurements": len(self.spectral_norms),
            }
        }

        # Compute statistics
        norms = list(self.spectral_norms.values())
        if norms:
            results["statistics"] = {
                "mean": sum(norms) / len(norms),
                "min": min(norms),
                "max": max(norms),
                "all_contractive": all(n < 1.0 for n in norms),
                "contraction_rate": sum(1 for n in norms if n < 1.0) / len(norms),
            }
            logger.info(
                f"Spectral norm statistics: mean={results['statistics']['mean']:.4f}, "
                f"range=[{results['statistics']['min']:.4f}, {results['statistics']['max']:.4f}], "
                f"all_contractive={results['statistics']['all_contractive']}"
            )

        # Save to file
        if self.output_dir:
            output_file = self.output_dir / "spectral_norms.json"
            with open(output_file, "w") as f:
                # Convert int keys to strings for JSON
                json_results = {
                    "spectral_norms": {str(k): v for k, v in self.spectral_norms.items()},
                    "lambda_maxs": {str(k): v for k, v in self.lambda_maxs.items()},
                    "etas": {str(k): v for k, v in self.etas.items()},
                    "metadata": results["metadata"],
                    "statistics": results.get("statistics", {}),
                }
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved spectral norm data to {output_file}")
