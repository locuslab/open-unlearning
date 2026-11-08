import logging
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers.trainer_utils import TrainOutput

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


@dataclass
class _KronStats:
    a: torch.Tensor
    b: torch.Tensor
    count: int = 0


class KFORGE(UnlearnTrainer):
    """One-shot Kronecker-Fisher edits for selected linear layers.

    K-FORGE estimates empirical Kronecker factors A = E[x x^T] and
    B = E[g g^T] on forget and retain batches. `legacy_v1` preserves the
    original doubly-whitened SVD heuristic; `wiener_v2` applies the
    retain-aware Wiener relaxation in the generalized-eigen basis.
    """

    def __init__(
        self,
        target_modules_regex: str = r".*mlp\.down_proj$",
        rank: int = 8,
        strength: float = 1.0,
        damping: float = 1e-4,
        damping_mode: str = "fixed",
        adaptive_damping_coeff: float = 0.1,
        damping_floor: float = 1e-12,
        max_calibration_batches: int = 8,
        max_samples: Optional[int] = None,
        normalize_factors: bool = True,
        skip_modules_larger_than: Optional[int] = None,
        max_target_modules: Optional[int] = None,
        factor_mode: str = "kron",
        use_retain_fisher: bool = True,
        spectrum_output_path: Optional[str] = None,
        spectrum_top_k: int = 64,
        skip_edit: bool = False,
        edit_variant: str = "wiener_v2",
        lambda_tradeoff: float = 0.01,
        edit_weight_dtype: str = "float32",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if rank < 1:
            raise ValueError("KFORGE rank must be >= 1")
        if strength < 0:
            raise ValueError("KFORGE strength must be non-negative")
        if damping <= 0:
            raise ValueError("KFORGE damping must be positive")
        if adaptive_damping_coeff <= 0:
            raise ValueError("KFORGE adaptive_damping_coeff must be positive")
        if damping_floor <= 0:
            raise ValueError("KFORGE damping_floor must be positive")
        if damping_mode not in {"fixed", "adaptive"}:
            raise ValueError("KFORGE damping_mode must be 'fixed' or 'adaptive'")
        if factor_mode not in {"kron", "diagonal"}:
            raise ValueError("KFORGE factor_mode must be 'kron' or 'diagonal'")
        if edit_variant not in {"legacy_v1", "wiener_v2"}:
            raise ValueError("KFORGE edit_variant must be 'legacy_v1' or 'wiener_v2'")
        if lambda_tradeoff < 0:
            raise ValueError("KFORGE lambda_tradeoff must be non-negative")
        if edit_weight_dtype not in {"preserve", "float32"}:
            raise ValueError("KFORGE edit_weight_dtype must be 'preserve' or 'float32'")

        self.target_modules_regex = target_modules_regex
        self.rank = rank
        self.strength = strength
        self.damping = damping
        self.damping_mode = damping_mode
        self.adaptive_damping_coeff = adaptive_damping_coeff
        self.damping_floor = damping_floor
        self.max_calibration_batches = max_calibration_batches
        self.max_samples = max_samples
        self.normalize_factors = normalize_factors
        self.skip_modules_larger_than = skip_modules_larger_than
        self.max_target_modules = max_target_modules
        self.factor_mode = factor_mode
        self.use_retain_fisher = use_retain_fisher
        self.spectrum_output_path = spectrum_output_path
        self.spectrum_top_k = spectrum_top_k
        self.skip_edit = skip_edit
        self.edit_variant = edit_variant
        self.lambda_tradeoff = lambda_tradeoff
        self.edit_weight_dtype = edit_weight_dtype

    def train(self, *args, **kwargs):
        if self.accelerator.num_processes != 1:
            raise RuntimeError("KFORGE currently supports single-process calibration.")

        start = time.time()
        modules = self._matching_linear_modules()
        if not modules:
            raise ValueError(
                f"No nn.Linear modules matched target_modules_regex={self.target_modules_regex!r}"
            )

        logger.info("KFORGE calibrating %d linear modules", len(modules))
        was_training = self.model.training
        self.model.eval()
        try:
            forget_stats = self._calibrate("forget", modules)
            retain_stats = self._calibrate("retain", modules)
        finally:
            self.model.train(was_training)
        self._log_factor_scales(modules, forget_stats, retain_stats)
        spectrum_metrics = self._write_spectrum(modules, forget_stats, retain_stats)
        if self.skip_edit:
            edit_metrics = {
                "kforge_edited_modules": 0.0,
                "kforge_skipped_modules": 0.0,
                "kforge_rank": float(self.rank),
                "kforge_strength": float(self.strength),
                "kforge_factor_mode_diagonal": float(self.factor_mode == "diagonal"),
                "kforge_use_retain_fisher": float(self.use_retain_fisher),
                "kforge_damping_mode_adaptive": float(self.damping_mode == "adaptive"),
                "kforge_adaptive_damping_coeff": float(self.adaptive_damping_coeff),
                "kforge_damping_floor": float(self.damping_floor),
                "kforge_top_sigma_sum": 0.0,
                "kforge_edit_variant_wiener_v2": float(
                    self.edit_variant == "wiener_v2"
                ),
                "kforge_lambda_tradeoff": float(self.lambda_tradeoff),
                "kforge_edit_weight_dtype_float32": float(
                    self.edit_weight_dtype == "float32"
                ),
            }
        else:
            edit_metrics = self._apply_kforge_edit(modules, forget_stats, retain_stats)

        runtime = time.time() - start
        metrics = {
            "train_runtime": runtime,
            "train_samples_per_second": 0.0,
            "train_steps_per_second": 0.0,
            "total_flos": 0.0,
            **edit_metrics,
            **spectrum_metrics,
        }
        self.state.global_step = 1
        self.log(metrics)
        return TrainOutput(self.state.global_step, 0.0, metrics)

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        if self.skip_edit:
            logger.info("KFORGE skip_edit=true; not saving an unchanged model")
            return
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _write_spectrum(
        self,
        modules: Dict[str, nn.Linear],
        forget_stats: Dict[str, _KronStats],
        retain_stats: Dict[str, _KronStats],
    ) -> Dict[str, float]:
        if not self.spectrum_output_path:
            return {}

        rows = []
        for name, module in modules.items():
            try:
                s = self._compute_module_spectrum(
                    module.weight.detach(),
                    forget_stats[name],
                    retain_stats[name],
                )
            except RuntimeError as exc:
                logger.warning("KFORGE spectrum skipped %s: %s", name, exc)
                continue

            if s.numel() == 0:
                continue
            top = s[: min(self.spectrum_top_k, s.numel())].tolist()
            quantiles = torch.quantile(
                s,
                torch.tensor(
                    [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
                    dtype=s.dtype,
                    device=s.device,
                ),
            ).tolist()
            rows.append(
                {
                    "module": name,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "num_singular_values": int(s.numel()),
                    "top_singular_values": top,
                    "quantiles": {
                        "q0": quantiles[0],
                        "q25": quantiles[1],
                        "q50": quantiles[2],
                        "q75": quantiles[3],
                        "q90": quantiles[4],
                        "q95": quantiles[5],
                        "q99": quantiles[6],
                        "q100": quantiles[7],
                    },
                }
            )

        output_path = Path(self.spectrum_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2) + "\n")
        logger.info(
            "KFORGE wrote spectrum for %d modules to %s", len(rows), output_path
        )
        return {
            "kforge_spectrum_modules": float(len(rows)),
            "kforge_spectrum_written": 1.0,
        }

    def _log_factor_scales(
        self,
        modules: Dict[str, nn.Linear],
        forget_stats: Dict[str, _KronStats],
        retain_stats: Dict[str, _KronStats],
    ) -> None:
        for name in modules:
            logger.info(
                (
                    "KFORGE factor scales %s "
                    "forget[A=%.6e B=%.6e tokens=%d] "
                    "retain[A=%.6e B=%.6e tokens=%d]"
                ),
                name,
                self._diag_mean(forget_stats[name].a),
                self._diag_mean(forget_stats[name].b),
                forget_stats[name].count,
                self._diag_mean(retain_stats[name].a),
                self._diag_mean(retain_stats[name].b),
                retain_stats[name].count,
            )

    def _matching_linear_modules(self) -> Dict[str, nn.Linear]:
        pattern = re.compile(self.target_modules_regex)
        modules = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear) or not pattern.fullmatch(name):
                continue
            numel = module.weight.numel()
            if self.skip_modules_larger_than and numel > self.skip_modules_larger_than:
                logger.info("KFORGE skipping %s with %d parameters", name, numel)
                continue
            modules[name] = module
            if self.max_target_modules and len(modules) >= self.max_target_modules:
                break
        return modules

    def _calibrate(
        self, split: str, modules: Dict[str, nn.Linear]
    ) -> Dict[str, _KronStats]:
        stats = {
            name: _KronStats(
                a=torch.zeros(
                    module.in_features, module.in_features, dtype=torch.float64
                ),
                b=torch.zeros(
                    module.out_features, module.out_features, dtype=torch.float64
                ),
            )
            for name, module in modules.items()
        }
        activations: Dict[str, torch.Tensor] = {}
        current_token_mask: Optional[torch.Tensor] = None
        handles = []

        def flatten_last_dim(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.detach()
                .reshape(-1, tensor.shape[-1])
                .to(device="cpu", dtype=torch.float64)
            )

        for name, module in modules.items():

            def forward_hook(_module, inputs, _output, module_name=name):
                activations[module_name] = inputs[0].detach()

            def backward_hook(_module, _grad_input, grad_output, module_name=name):
                if module_name not in activations or grad_output[0] is None:
                    return
                x = activations.pop(module_name)
                g = grad_output[0].detach()
                if current_token_mask is not None:
                    x, g = self._mask_token_rows(x, g, current_token_mask)
                x = flatten_last_dim(x)
                g = flatten_last_dim(g)
                if x.shape[0] != g.shape[0]:
                    token_count = min(x.shape[0], g.shape[0])
                    x = x[:token_count]
                    g = g[:token_count]
                stats[module_name].a.add_(x.transpose(0, 1).matmul(x))
                stats[module_name].b.add_(g.transpose(0, 1).matmul(g))
                stats[module_name].count += x.shape[0]

            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))

        try:
            seen = 0
            for batch_idx, inputs in enumerate(self.get_train_dataloader()):
                if (
                    self.max_calibration_batches
                    and batch_idx >= self.max_calibration_batches
                ):
                    break
                if self.max_samples is not None and seen >= self.max_samples:
                    break
                split_inputs = inputs.get(split)
                if split_inputs is None:
                    raise ValueError(
                        f"KFORGE expected '{split}' batches in train_dataset"
                    )
                split_inputs = {
                    "input_ids": split_inputs["input_ids"],
                    "attention_mask": split_inputs["attention_mask"],
                    "labels": split_inputs["labels"],
                }
                batch_size = int(split_inputs["input_ids"].shape[0])
                split_inputs = self._prepare_inputs(split_inputs)
                current_token_mask, valid_token_count = self._causal_lm_token_mask(
                    split_inputs["labels"]
                )
                if valid_token_count == 0:
                    continue
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(**split_inputs)
                # HF causal-LM losses are mean-reduced over valid labels.
                # Multiply by the valid-token count so B estimates sum-loss
                # token gradients and is comparable across batch compositions.
                self.accelerator.backward(outputs.loss * valid_token_count)
                seen += batch_size
        finally:
            current_token_mask = None
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)

        for name, stat in stats.items():
            if stat.count == 0:
                raise RuntimeError(f"KFORGE collected no calibration tokens for {name}")
            if self.normalize_factors:
                stat.a.div_(stat.count)
                stat.b.div_(stat.count)
        logger.info("KFORGE calibrated %s split on %d examples", split, seen)
        return stats

    def _apply_kforge_edit(
        self,
        modules: Dict[str, nn.Linear],
        forget_stats: Dict[str, _KronStats],
        retain_stats: Dict[str, _KronStats],
    ) -> Dict[str, float]:
        edited = 0
        skipped = 0
        sigma_sum = 0.0

        with torch.no_grad():
            if self.edit_weight_dtype == "float32":
                # No forward pass follows the one-shot edit. Upcasting the
                # checkpoint itself keeps small edits representable after save.
                self.model.to(dtype=torch.float32)
            for name, module in modules.items():
                try:
                    delta, top_sigma = self._compute_module_delta(
                        # No optimizer step occurs between calibration and edit;
                        # the calibrated weights are still the current weights.
                        module.weight.detach(),
                        forget_stats[name],
                        retain_stats[name],
                    )
                except RuntimeError as exc:
                    skipped += 1
                    logger.warning("KFORGE skipped %s: %s", name, exc)
                    continue

                module.weight.add_(
                    delta.to(device=module.weight.device, dtype=module.weight.dtype)
                )
                edited += 1
                sigma_sum += top_sigma

        return {
            "kforge_edited_modules": float(edited),
            "kforge_skipped_modules": float(skipped),
            "kforge_rank": float(self.rank),
            "kforge_strength": float(self.strength),
            "kforge_factor_mode_diagonal": float(self.factor_mode == "diagonal"),
            "kforge_use_retain_fisher": float(self.use_retain_fisher),
            "kforge_damping_mode_adaptive": float(self.damping_mode == "adaptive"),
            "kforge_adaptive_damping_coeff": float(self.adaptive_damping_coeff),
            "kforge_damping_floor": float(self.damping_floor),
            "kforge_top_sigma_sum": float(sigma_sum),
            "kforge_edit_variant_wiener_v2": float(self.edit_variant == "wiener_v2"),
            "kforge_lambda_tradeoff": float(self.lambda_tradeoff),
            "kforge_edit_weight_dtype_float32": float(
                self.edit_weight_dtype == "float32"
            ),
        }

    def _compute_module_delta(
        self, weight: torch.Tensor, forget: _KronStats, retain: _KronStats
    ) -> tuple[torch.Tensor, float]:
        device = weight.device
        dtype = torch.float64
        weight64 = weight.to(device="cpu", dtype=dtype)

        l_af, l_bf, l_ar, l_br = self._factor_choleskys(forget, retain)
        if self.edit_variant == "legacy_v1":
            delta, top_sigma = self._compute_module_delta_legacy_v1(
                weight64, l_af, l_bf, l_ar, l_br
            )
        else:
            delta, top_sigma = self._compute_module_delta_wiener_v2(
                weight64, l_af, l_bf, l_ar, l_br
            )
        delta = self.strength * delta
        return delta.to(device=device), top_sigma

    def _factor_choleskys(
        self, forget: _KronStats, retain: _KronStats
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a_f = self._prepare_factor(forget.a, reference=retain.a)
        b_f = self._prepare_factor(forget.b, reference=retain.b)
        if self.use_retain_fisher:
            a_r = self._prepare_factor(retain.a, reference=retain.a)
            b_r = self._prepare_factor(retain.b, reference=retain.b)
        else:
            a_r = torch.eye(a_f.shape[0], dtype=a_f.dtype, device=a_f.device)
            b_r = torch.eye(b_f.shape[0], dtype=b_f.dtype, device=b_f.device)
        return (
            torch.linalg.cholesky(a_f),
            torch.linalg.cholesky(b_f),
            torch.linalg.cholesky(a_r),
            torch.linalg.cholesky(b_r),
        )

    def _compute_module_delta_legacy_v1(
        self,
        weight64: torch.Tensor,
        l_af: torch.Tensor,
        l_bf: torch.Tensor,
        l_ar: torch.Tensor,
        l_br: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        whitened = l_bf.transpose(0, 1).matmul(weight64).matmul(l_af)
        whitened = torch.linalg.solve_triangular(
            l_br.transpose(0, 1), whitened, upper=True
        )
        whitened = self._right_solve_lower_inverse(whitened, l_ar)

        u, s, vh = torch.linalg.svd(whitened, full_matrices=False)
        rank = min(self.rank, s.numel())
        low_rank = (u[:, :rank] * s[:rank]).matmul(vh[:rank, :])

        delta = torch.linalg.solve_triangular(
            l_bf.transpose(0, 1), low_rank, upper=True
        )
        delta = self._right_solve_lower_inverse(delta, l_af)
        return -delta, float(s[:rank].sum().item())

    def _compute_module_delta_wiener_v2(
        self,
        weight64: torch.Tensor,
        l_af: torch.Tensor,
        l_bf: torch.Tensor,
        l_ar: torch.Tensor,
        l_br: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        k_a = torch.linalg.solve_triangular(l_ar, l_af, upper=False)
        k_b = torch.linalg.solve_triangular(l_br, l_bf, upper=False)
        u_a, s_a, vh_a = torch.linalg.svd(k_a, full_matrices=False)
        u_b, s_b, vh_b = torch.linalg.svd(k_b, full_matrices=False)
        v_a = vh_a.transpose(0, 1)
        v_b = vh_b.transpose(0, 1)

        r_matrix = (
            v_b.transpose(0, 1)
            .matmul(l_bf.transpose(0, 1).matmul(weight64).matmul(l_af))
            .matmul(v_a)
        )
        sigma_grid = s_b[:, None] * s_a[None, :]
        sigma2 = sigma_grid.square()
        gain_tilde = sigma2 / (sigma2 + self.lambda_tradeoff)
        y_star = -(gain_tilde * r_matrix)

        u_y, s_y, vh_y = torch.linalg.svd(y_star, full_matrices=False)
        rank = min(self.rank, s_y.numel())
        y_r = (u_y[:, :rank] * s_y[:rank]).matmul(vh_y[:rank, :])

        # The rank-r approximation is built in Y-space. Division maps it back
        # to H-space; damping keeps the generalized singular values positive.
        h_r = y_r / s_b[:, None]
        h_r = h_r / s_a[None, :]

        left = torch.linalg.solve_triangular(
            l_br.transpose(0, 1), u_b.matmul(h_r), upper=True
        )
        delta = self._right_solve_lower_inverse(left.matmul(u_a.transpose(0, 1)), l_ar)
        return delta, float(s_y[:rank].sum().item())

    def _compute_module_spectrum(
        self, weight: torch.Tensor, forget: _KronStats, retain: _KronStats
    ) -> torch.Tensor:
        dtype = torch.float64
        weight64 = weight.to(device="cpu", dtype=dtype)

        l_af, l_bf, l_ar, l_br = self._factor_choleskys(forget, retain)

        whitened = l_bf.transpose(0, 1).matmul(weight64).matmul(l_af)
        whitened = torch.linalg.solve_triangular(
            l_br.transpose(0, 1), whitened, upper=True
        )
        whitened = self._right_solve_lower_inverse(whitened, l_ar)
        return torch.linalg.svdvals(whitened).cpu()

    def _prepare_factor(
        self, matrix: torch.Tensor, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        matrix = 0.5 * (matrix + matrix.transpose(0, 1))
        if self.factor_mode == "diagonal":
            matrix = torch.diag_embed(torch.diagonal(matrix).clamp_min(0.0))
        if self.damping_mode == "adaptive":
            if reference is None:
                reference = matrix
            reference = 0.5 * (reference + reference.transpose(0, 1))
            diag_mean = torch.diagonal(reference).mean().clamp_min(self.damping_floor)
            damping = self.adaptive_damping_coeff
        else:
            diag_mean = torch.diagonal(matrix).mean().clamp_min(self.damping_floor)
            damping = self.damping
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        return matrix + damping * diag_mean * eye

    @staticmethod
    def _diag_mean(matrix: torch.Tensor) -> float:
        return float(torch.diagonal(matrix).mean().item())

    @staticmethod
    def _causal_lm_token_mask(labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        token_mask = torch.zeros_like(labels, dtype=torch.bool)
        valid_loss_mask = labels[:, 1:].ne(-100)
        token_mask[:, :-1] = valid_loss_mask
        return token_mask, int(valid_loss_mask.sum().item())

    @staticmethod
    def _mask_token_rows(
        x: torch.Tensor, g: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim < 2 or g.ndim < 2:
            return x, g
        batch = min(x.shape[0], g.shape[0], token_mask.shape[0])
        seq = min(x.shape[1], g.shape[1], token_mask.shape[1])
        mask = token_mask[:batch, :seq]
        return x[:batch, :seq][mask], g[:batch, :seq][mask]

    @staticmethod
    def _right_solve_lower_inverse(
        matrix: torch.Tensor, lower: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.solve_triangular(
            lower.transpose(0, 1), matrix.transpose(0, 1), upper=True
        ).transpose(0, 1)
