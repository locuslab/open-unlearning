"""
LMCleaner Core Components: 在线遗忘的核心算法组件

包含:
- HVP (Hessian-Vector Product) 计算
- 参数校正向量计算 (前向K步传播)
- 参数校正应用
- 隐私噪声注入 (Phase 4 from paper)
- 训练日志数据结构
- 审计记录

论文参考: LMCleaner: Efficient and Certified Online Unlearning via Truncated Influence Propagation

实现与论文对齐说明:
- Fisher HVP: 论文 Algorithm 1 使用 Hv = g · (g^T v) 近似
- GGN HVP: 使用 Hv = ∇(g^T v) 近似 (更精确但计算量更大)
- 隐私噪声: 论文 Phase 4 注入 N(0, σ²I) 实现 (ε,δ)-certified unlearning
- 阻尼项: 实现添加的数值稳定化技术，论文理论分析未包含
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any, Protocol, Union
import weakref
import gc

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord(dict):
    """审计记录，用于记录遗忘操作的详细信息

    包含论文 Algorithm 1 各阶段的追踪信息:
    - Phase 1: 初始偏差计算
    - Phase 2: 截断影响传播
    - Phase 3: 影响移除
    - Phase 4: 隐私保护 (噪声注入)
    """

    tz: int = 0  # 遗忘批次的步骤
    tau: int = 0  # 当前步骤
    K_used: int = 0  # 实际使用的截断窗口大小
    v_norm: float = 0.0  # 校正向量的范数 (Phase 3 前)
    hvp_calls: int = 0  # HVP调用次数
    mode: str = "GGN"  # Hessian模式: "fisher", "GGN", "diag", "exact"
    damping: float = 0.0  # 阻尼系数 (实现添加，论文未包含)
    # Phase 4: 隐私噪声相关
    noise_sigma: float = 0.0  # 注入的噪声标准差 σ
    noise_injected: bool = False  # 是否注入了隐私噪声
    epsilon: float = 0.0  # (ε,δ)-certified unlearning 的 ε
    delta: float = 0.0  # (ε,δ)-certified unlearning 的 δ
    # 历史参数重建
    used_historical_params: bool = False  # 是否实际使用了历史 θ[s]
    # 效率追踪
    wall_time_ms: float = 0.0  # 该 forget step 的处理耗时(毫秒)

    def __post_init__(self):
        # 使其可以像字典一样使用
        dict.__init__(
            self,
            tz=self.tz,
            tau=self.tau,
            K_used=self.K_used,
            v_norm=self.v_norm,
            hvp_calls=self.hvp_calls,
            mode=self.mode,
            damping=self.damping,
            noise_sigma=self.noise_sigma,
            noise_injected=self.noise_injected,
            epsilon=self.epsilon,
            delta=self.delta,
            used_historical_params=self.used_historical_params,
            wall_time_ms=self.wall_time_ms,
        )


@dataclass
class StepRecord:
    """单步训练记录"""

    step_id: int
    eta: float  # 学习率
    batch_id: Any  # 批次标识符
    u: Optional[torch.Tensor] = None  # 参数更新向量 θ[t+1] - θ[t]
    gbar: Optional[torch.Tensor] = None  # 平均梯度
    theta_ref: Optional[weakref.ref] = None  # 参数弱引用
    batch_data: Optional[Any] = None  # 批次数据(用于HVP计算)
    diag_H: Optional[torch.Tensor] = None  # 对角Hessian近似(可选)

    def __repr__(self):
        return (
            f"StepRecord(step={self.step_id}, eta={self.eta}, batch_id={self.batch_id})"
        )


class LazyLoaderProtocol(Protocol):
    """Protocol for lazy record loaders that support on-demand loading."""

    def load_single_step(
        self, step_id: int, include_tensors: bool = True
    ) -> Optional[Dict]:
        """Load a single step record on demand."""
        ...

    def get_etas_for_steps(self, step_ids: List[int]) -> Dict[int, float]:
        """Get eta (learning rate) values for multiple steps."""
        ...

    @property
    def sample_indices(self) -> Dict[int, List[int]]:
        """Get sample indices mapping."""
        ...


class StepLog:
    """环形缓冲区，存储最近K+Δ个训练步骤的记录"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.step_map = {}  # step_id -> StepRecord (direct reference)

    def add(self, record: StepRecord):
        """添加新的步骤记录"""
        # 如果缓冲区满了,移除最旧的记录
        if len(self.buffer) == self.max_size:
            oldest = self.buffer[0]
            if oldest.step_id in self.step_map:
                del self.step_map[oldest.step_id]

        # 添加新记录
        self.buffer.append(record)
        self.step_map[record.step_id] = record  # Store record directly

    def get(self, step_id: int) -> Optional[StepRecord]:
        """获取指定步骤的记录"""
        return self.step_map.get(step_id)

    def __getitem__(self, step_id: int) -> Optional[StepRecord]:
        return self.get(step_id)

    def has_range(self, start: int, end: int) -> bool:
        """检查是否包含指定范围的所有步骤"""
        return all(self.get(s) is not None for s in range(start, end + 1))

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.step_map.clear()


class HVPConfig:
    """HVP计算配置

    支持的 Hessian 近似模式:
    - "fisher": 论文 Algorithm 1 使用的 Fisher 信息矩阵近似
              Hv = g · (g^T v), 其中 g 是梯度向量
              这是 rank-1 近似，计算高效但精度较低
    - "GGN": 广义 Gauss-Newton 近似
            Hv = ∇(g^T v) = ∇²L · v
            使用二阶自动微分，更精确但计算量更大
    - "diag": 对角 Hessian 近似
    - "exact": 精确 Hessian-vector product (非常慢)

    阻尼项说明 (damping):
        论文原始算法不包含阻尼项。实现添加阻尼项 λ 用于数值稳定化:
        v ← v - η[s] * (H[s] @ v) - η[s] * λ * v
        等价于在 Hessian 上添加正则化: (H + λI) @ v
        当 damping=0 时与论文算法一致。
    """

    def __init__(
        self,
        mode: str = "fisher",  # 默认使用论文的 Fisher 近似
        damping: float = 0.0,  # 默认不添加阻尼，与论文一致
        rank: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        valid_modes = {"fisher", "GGN", "diag", "exact", "low_rank"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")
        self.mode = mode
        self.damping = damping
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.hessian_mode = mode  # 兼容性


def hvp_exact(
    model: nn.Module,
    loss_fn: Callable,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    精确HVP计算(使用二阶autograd)

    Args:
        model: 模型
        loss_fn: 损失函数
        batch_data: 批次数据
        v: 向量
        params: 参数列表(如果为None,使用model.parameters())

    Returns:
        Hessian-vector product
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # 前向传播计算损失
    model.zero_grad()
    outputs = model(**batch_data)
    loss = loss_fn(outputs) if callable(loss_fn) else outputs.loss

    # 计算一阶梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # 计算梯度与v的点积
    dot = sum(
        (g * v_part).sum() for g, v_part in zip(grads, _unflatten_like(v, params))
    )

    # 计算二阶导数
    hvp = torch.autograd.grad(dot, params)

    # 展平并拼接
    return _flatten(hvp)


def hvp_fisher(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
    loss_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Fisher 信息矩阵近似的 HVP (论文 Algorithm 1 使用的方法)

    论文公式: Hv ← g · (g^T v)
    其中 g 是损失关于参数的梯度向量

    这是 rank-1 近似:
    - H ≈ g g^T (外积)
    - Hv = g (g^T v) = g * scalar

    特点:
    - 计算高效: 只需要一次梯度计算 + 向量点积
    - 精度较低: rank-1 近似可能丢失 Hessian 的高阶信息
    - 与论文一致: 这是论文 Algorithm 1 Line 6 描述的方法

    Args:
        model: 模型
        batch_data: 批次数据
        v: 向量
        params: 参数列表 (如果为 None, 使用 model.parameters())
        loss_fn: 自定义损失函数 (优先使用)

    Returns:
        Fisher 近似的 Hessian-vector product
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # 前向传播计算损失
    model.zero_grad()
    outputs = model(**batch_data)

    # 优先使用显式提供的 loss_fn，与其他 HVP 路径保持一致
    if loss_fn is not None:
        loss = loss_fn(outputs, batch_data)
    elif hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    else:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = batch_data.get("labels", None)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
            )
        else:
            raise ValueError("No loss_fn, outputs.loss, or labels found")

    # 计算梯度 g = ∇L
    grads = torch.autograd.grad(loss, params)

    # 展平梯度为向量
    g = _flatten([grad.detach() for grad in grads])

    # 计算 g^T v (标量)
    g_dot_v = torch.dot(g, v)

    # 返回 g * (g^T v) = Hv (Fisher 近似)
    return g * g_dot_v


def hvp_ggn(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    广义Gauss-Newton近似的HVP

    注意: 这与论文 Algorithm 1 的 Fisher 近似不同!
    论文使用: Hv = g · (g^T v) (rank-1 外积)
    GGN 使用: Hv = ∇(g^T v) (通过二阶自动微分)

    GGN: H ≈ J^T H_loss J
    其中 J 是输出关于参数的Jacobian, H_loss 是损失关于输出的Hessian

    对于交叉熵: H_loss ≈ I (忽略二阶项)
    因此: Hv ≈ J^T (J v)

    特点:
    - 比 Fisher 更精确
    - 需要二阶自动微分,计算量更大
    - 适用于需要更高精度的场景
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend

    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # Ensure v matches both device and dtype of model parameters (A40 GPU fix)
    ref = params[0] if params else v
    if v.device != ref.device or v.dtype != ref.dtype:
        v = v.to(ref)

    # Ensure all batch_data tensors are on the same device as the model
    if isinstance(batch_data, dict):
        for k, val in batch_data.items():
            if isinstance(val, torch.Tensor) and val.device != ref.device:
                batch_data[k] = val.to(ref.device)

    # 分解v为参数形状
    v_list = _unflatten_like(v, params)

    # 1. 计算 Jv (Jacobian-vector product)
    model.zero_grad()

    # 使用torch.func.jvp计算Jacobian-vector product
    # 首先前向传播
    # NOTE: Use MATH backend for SDPA to support 2nd order gradients (A40 GPU)
    # Flash/Memory-efficient attention don't support 2nd order derivatives
    with sdpa_kernel(SDPBackend.MATH):
        outputs = model(**batch_data)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    # 计算 ∇params(output) @ v
    # 使用反向传播: 如果设置grad_outputs=v, 得到的是 J^T v
    # 我们需要先计算 Jv, 再计算 J^T(Jv)

    # 使用有限差分近似或者双重反向传播
    # 简化: 使用 GGN ≈ J^T J
    # 实际上对于交叉熵,可以用Fisher信息矩阵近似

    # 计算损失关于logits的梯度
    if hasattr(outputs, "loss"):
        loss = outputs.loss
    else:
        # 计算交叉熵损失
        labels = batch_data.get("labels", None)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
            )
        else:
            raise ValueError("No loss or labels found")

    # 计算梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # 计算 g^T v
    gv = sum((g * v_part).sum() for g, v_part in zip(grads, v_list))

    # 计算 ∇(g^T v) = GGN @ v (近似)
    hvp = torch.autograd.grad(gv, params)

    return _flatten(hvp)


def hvp_diagonal(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    diag_H: Optional[torch.Tensor] = None,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    对角Hessian近似

    Args:
        model: 模型
        batch_data: 批次数据
        v: 向量
        diag_H: 预计算的对角Hessian(如果为None,会计算)
        params: 参数列表

    Returns:
        对角近似的 Hv
    """
    if diag_H is not None:
        return diag_H * v

    # 如果没有预计算,需要计算对角Hessian
    from torch.nn.attention import sdpa_kernel, SDPBackend

    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    # NOTE: Use MATH backend for SDPA to support 2nd order gradients (A40 GPU)
    with sdpa_kernel(SDPBackend.MATH):
        outputs = model(**batch_data)
    loss = outputs.loss if hasattr(outputs, "loss") else outputs

    # 计算梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # 计算每个参数的对角Hessian
    diag_hessian = []
    for g in grads:
        # 对每个梯度分量计算二阶导数
        diag_h = []
        g_flat = g.view(-1)
        for i in range(g_flat.numel()):
            # 这个实现非常慢,仅用于演示
            # 实际应该使用更高效的方法
            h_ii = torch.autograd.grad(g_flat[i], params, retain_graph=True)[0]
            diag_h.append(h_ii.view(-1)[i])
        diag_hessian.append(torch.stack(diag_h))

    diag_H_full = torch.cat(diag_hessian)
    return diag_H_full * v


class HistoricalParamContext:
    """Context manager that temporarily sets model to historical θ[s] during propagation.

    Reconstructs θ[s] = θ[τ] - Σ_{t=s}^{τ-1} u[t] using stored parameter update
    vectors, and advances through θ[s] → θ[s+1] → ... during the propagation loop.
    Restores θ[τ] on exit (including exceptions).

    When disabled or when u[t] vectors are unavailable, acts as a no-op.

    Usage:
        with HistoricalParamContext(model, ...) as ctx:
            for s in range(start, end + 1):
                # model is now at θ[s]
                hvp = hvp_apply(v, srec, cfg, model, ...)
                ctx.advance(s)
        # model is back to θ[τ]
    """

    def __init__(
        self,
        model: nn.Module,
        start: int,
        end: int,
        tau: int,
        enabled: bool,
        step_log: Optional["StepLog"],
        lazy_loader: Optional["LazyLoaderProtocol"],
        use_lazy_loading: bool,
    ):
        self.model = model
        self.start = start
        self.end = end
        self.tau = tau
        self._model_device = next(model.parameters()).device

        self.active = False  # whether historical params are actually in use
        self._theta_tau: Optional[torch.Tensor] = None
        self._theta_current: Optional[torch.Tensor] = None
        self._u_vectors: Dict[int, torch.Tensor] = {}

        if not enabled or (end < start):
            if not enabled:
                logger.debug("Historical parameter reconstruction disabled by user")
            return

        self._try_load(step_log, lazy_loader, use_lazy_loading)

    def _load_u(
        self,
        t: int,
        step_log: Optional["StepLog"],
        lazy_loader: Optional["LazyLoaderProtocol"],
        use_lazy_loading: bool,
    ) -> Optional[torch.Tensor]:
        if use_lazy_loading and lazy_loader is not None:
            rec_dict = lazy_loader.load_single_step(t, include_tensors=True)
            if rec_dict is not None:
                u_val = rec_dict.get("u")
                del rec_dict
                return u_val
        elif step_log is not None:
            srec_tmp = step_log[t]
            if srec_tmp is not None:
                return srec_tmp.u
        return None

    def _try_load(
        self,
        step_log: Optional["StepLog"],
        lazy_loader: Optional["LazyLoaderProtocol"],
        use_lazy_loading: bool,
    ) -> None:
        """Try to load u vectors; set self.active = True on success."""
        # Phase A: load u vectors for propagation window [start, end]
        for t in range(self.start, self.end + 1):
            u_t = self._load_u(t, step_log, lazy_loader, use_lazy_loading)
            if u_t is None:
                self._u_vectors.clear()
                logger.warning(
                    "Cannot reconstruct historical parameters: u[t] not available "
                    f"for all steps in [{self.start}, {self.end}]. "
                    "Falling back to θ[τ]."
                )
                return
            self._u_vectors[t] = u_t.to(self._model_device)

        # Phase B: stream-accumulate tail [end+1, tau-1] without storing
        tail_sum: Optional[torch.Tensor] = None
        for t in range(self.end + 1, self.tau):
            u_t = self._load_u(t, step_log, lazy_loader, use_lazy_loading)
            if u_t is None:
                self._u_vectors.clear()
                logger.warning(
                    "Cannot reconstruct historical parameters: u[t] not available "
                    f"for tail steps in [{self.end + 1}, {self.tau - 1}]. "
                    "Falling back to θ[τ]."
                )
                return
            u_t = u_t.to(self._model_device)
            if tail_sum is None:
                tail_sum = u_t.clone()
            else:
                tail_sum += u_t
            del u_t

        # Compute θ[start] = θ[τ] - Σ u[t]
        self._theta_tau = _get_flat_params(self.model).clone()
        self._theta_current = self._theta_tau.clone()
        for t in range(self.start, self.end + 1):
            self._theta_current -= self._u_vectors[t]
        if tail_sum is not None:
            self._theta_current -= tail_sum

        self.active = True
        logger.debug(
            f"Historical parameter reconstruction enabled for steps "
            f"[{self.start}, {self.end}] ({len(self._u_vectors)} u vectors)"
        )

    def __enter__(self) -> "HistoricalParamContext":
        if self.active:
            _set_flat_params(self.model, self._theta_current)
        return self

    def __exit__(self, *exc) -> None:
        if self.active and self._theta_tau is not None:
            _set_flat_params(self.model, self._theta_tau)
            del self._theta_tau, self._theta_current
            self._u_vectors.clear()
            # Note: keep self.active = True for audit purposes

    def advance(self, s: int) -> None:
        """Advance model from θ[s] to θ[s+1]."""
        if self.active and self._theta_current is not None and s in self._u_vectors:
            self._theta_current += self._u_vectors[s]
            _set_flat_params(self.model, self._theta_current)


@torch.no_grad()
def _set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set model parameters from a flattened vector (in-place)."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            p.copy_(flat_params[offset : offset + n].view_as(p))
            offset += n
    if offset != flat_params.numel():
        raise ValueError(
            f"Flat params size mismatch: consumed {offset}, got {flat_params.numel()}"
        )


@torch.no_grad()
def _get_flat_params(model: nn.Module) -> torch.Tensor:
    """Get model parameters as a flattened, detached vector."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def hvp_apply(
    v: torch.Tensor,
    step_rec: StepRecord,
    cfg: HVPConfig,
    model: nn.Module,
    loss_fn: Optional[Callable] = None,
    batch_reconstructor: Optional[Any] = None,
) -> torch.Tensor:
    """
    应用HVP计算: 返回 H[s] @ v

    根据论文 Algorithm 1, HVP 应在历史参数 θ[s] 处计算。
    调用方 (compute_correction) 负责在调用前将模型参数设置为 θ[s],
    调用后恢复为 θ[τ]。

    Args:
        v: 输入向量
        step_rec: 步骤记录
        cfg: HVP配置
        model: 模型 (调用方应确保参数已设置为历史 θ[s])
        loss_fn: 损失函数
        batch_reconstructor: 批次重建器(当batch_data为None时使用)

    Returns:
        Hessian-vector product
    """
    mode = cfg.hessian_mode

    # 获取批次数据
    batch_data = step_rec.batch_data

    # 如果没有batch_data,尝试重建
    if batch_data is None:
        # 对于diag模式,如果有预计算的diag_H,可以直接使用
        if mode == "diag" and step_rec.diag_H is not None:
            logger.debug(f"Using precomputed diag_H for step {step_rec.step_id}")
            return step_rec.diag_H * v

        # 否则需要重建批次数据
        if batch_reconstructor is None:
            raise ValueError(
                f"No batch data found for step {step_rec.step_id} and no batch_reconstructor provided"
            )

        logger.debug(f"Reconstructing batch data for step {step_rec.step_id}")
        batch_data = batch_reconstructor.get_batch_for_step(step_rec.step_id)

        if batch_data is None:
            raise ValueError(
                f"Failed to reconstruct batch data for step {step_rec.step_id}"
            )

    # 确保数据在正确的设备上
    batch_data = {
        k: val.to(cfg.device) if isinstance(val, torch.Tensor) else val
        for k, val in batch_data.items()
    }

    if mode == "fisher":
        # 论文 Algorithm 1 使用的方法: Hv = g · (g^T v)
        return hvp_fisher(model, batch_data, v, loss_fn=loss_fn)
    elif mode == "diag":
        return hvp_diagonal(model, batch_data, v, step_rec.diag_H)
    elif mode == "GGN":
        return hvp_ggn(model, batch_data, v)
    elif mode == "exact":
        return hvp_exact(model, loss_fn, batch_data, v)
    elif mode == "low_rank":
        # TODO: 实现低秩近似
        logger.warning("Low-rank HVP not implemented, falling back to fisher")
        return hvp_fisher(model, batch_data, v, loss_fn=loss_fn)
    else:
        raise ValueError(f"Unknown hessian_mode: {mode}")


def compute_noise_sigma(
    delta_det: float,
    epsilon: float,
    delta: float,
) -> float:
    """
    计算隐私噪声的标准差 σ (论文 Theorem 2)

    论文公式:
        σ ≥ (Δ_det / ε) * sqrt(2 * log(1.25 / δ))

    其中:
    - Δ_det: 近似误差上界 (deterministic approximation error bound)
    - ε: 隐私参数 epsilon
    - δ: 隐私参数 delta

    Args:
        delta_det: 近似误差上界 ||θ̂ - θ_ideal||₂ ≤ Δ_det
        epsilon: 隐私参数 ε > 0
        delta: 隐私参数 δ ∈ (0, 1)

    Returns:
        噪声标准差 σ

    Note:
        delta_det = 0 is valid (results in sigma = 0, meaning no noise needed).
    """
    if delta_det < 0:
        raise ValueError(f"delta_det must be >= 0, got {delta_det}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    sigma = (delta_det / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    return sigma


def inject_privacy_noise(
    v: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    注入隐私噪声 (论文 Algorithm 1 Phase 4)

    论文公式:
        θ̃[τ] ← θ̂[τ] + N(0, σ²I)

    Args:
        v: 校正向量 (将被添加噪声)
        sigma: 噪声标准差 σ
        generator: 随机数生成器 (用于可复现性)

    Returns:
        添加噪声后的向量: v + N(0, σ²I)
    """
    if sigma <= 0:
        return v

    noise = torch.randn(
        v.shape,
        dtype=v.dtype,
        device=v.device,
        generator=generator,
    ) * sigma

    return v + noise


@torch.no_grad()
def compute_correction(
    tz: int,
    tau: int,
    K: int,
    step_log: Optional[StepLog],
    cfg: HVPConfig,
    model: nn.Module,
    loss_fn: Optional[Callable] = None,
    batch_reconstructor: Optional[Any] = None,
    # Phase 4: 隐私噪声参数
    epsilon: float = 0.0,
    delta: float = 1e-5,
    delta_det: Optional[float] = None,
    # On-demand loading support for large K values
    lazy_loader: Optional[LazyLoaderProtocol] = None,
    initial_record: Optional[StepRecord] = None,
    # Historical parameter reconstruction
    use_historical_params: bool = True,
) -> Tuple[torch.Tensor, AuditRecord]:
    """
    计算参数校正向量(前向K步传播) - 论文 Algorithm 1 完整实现

    算法:
    Phase 1: 计算初始偏差 v0 = -u[tz] 或 v0 = η[tz] * gbar[tz]
    Phase 2: 前向传播 v[s+1] = v[s] - η[s] * (H[s] @ v[s]) - η[s] * λ * v[s]
             其中 H[s] 在历史参数 θ[s] 处计算 (论文 Algorithm 1 Line 6)
    Phase 3: 返回校正向量 v (将在外部应用: θ ← θ + v)
    Phase 4: 可选的隐私噪声注入

    历史参数重建 (论文 Section 5 - Parameter Snapshots):
        θ[s] = θ[τ] - Σ_{t=s}^{τ-1} u[t]
        其中 u[t] = θ[t+1] - θ[t] 是每步的参数更新向量。
        当 use_historical_params=True 且 u[t] 向量可用时，在每步传播前
        将模型参数临时设为 θ[s], 计算 HVP 后恢复为 θ[τ]。
        当 u[t] 不可用或 use_historical_params=False 时，使用 θ[τ]。

    ! 注意 (内存开销):
    !   启用 use_historical_params 会额外占用约 1x 模型参数量的 GPU 显存
    !   (用于维护 theta_current flat 向量) 以及读取窗口内 u[t] 的 IO 开销。
    !   对于显存紧张的环境，可设置 use_historical_params=False 退化为旧行为。
    !   详见 docs/historical_params_hvp.md

    注意 (与论文的差异):
    - 阻尼项 λ 是实现添加的数值稳定化技术，论文理论分析未包含
    - 设置 cfg.damping=0 以与论文算法完全一致

    Args:
        tz: 遗忘批次的步骤
        tau: 当前步骤
        K: 截断窗口大小
        step_log: 步骤日志 (如果 lazy_loader 提供，可以为 None)
        cfg: HVP配置
        model: 模型
        loss_fn: 损失函数
        batch_reconstructor: 批次重建器(可选)
        epsilon: 隐私参数 ε (>0 时启用噪声注入)
        delta: 隐私参数 δ (默认 1e-5)
        delta_det: 近似误差上界 (如果为 None, 使用 ||v||₂ 作为估计)
        lazy_loader: 懒加载器，用于大 K 值时按需加载记录 (避免 OOM)
        initial_record: 初始步骤记录 (tz 对应的记录，用于 Phase 1)
        use_historical_params: 是否在历史参数 θ[s] 处计算 HVP (默认 True)
            True: 论文 Algorithm 1 的精确实现，额外 ~1x 模型参数量显存
            False: 在 θ[τ] 处计算 HVP，节省显存但引入近似误差

    Returns:
        (v, audit_record): 校正向量和审计记录
    """
    # 确定是否使用 lazy loading 模式
    use_lazy_loading = lazy_loader is not None

    # 获取初始记录
    if initial_record is not None:
        rec = initial_record
    elif step_log is not None:
        rec = step_log[tz]
    else:
        raise ValueError("Either step_log or initial_record must be provided")

    if rec is None:
        raise ValueError(f"Step {tz} not found in step log")

    if rec.u is None and (rec.gbar is None or rec.eta is None):
        raise ValueError(f"Step {tz} missing update vector u or (gbar, eta)")

    # Phase 1: 初始偏差 v0
    if rec.u is not None:
        # v0 = -u[tz] = -(θ[tz+1] - θ[tz]) = θ[tz] - θ[tz+1]
        # 因为 u[tz] = -η[tz] * gbar[tz], 所以 -u[tz] = η[tz] * gbar[tz]
        v = -rec.u.clone()
    else:
        # v0 = η[tz] * gbar[tz]
        v = rec.eta * rec.gbar.clone()

    # 确保 v 在正确的设备上 (从训练日志加载的 u/gbar 可能在 CPU 上)
    v = v.to(cfg.device)

    # 清理初始记录
    if use_lazy_loading and initial_record is not None:
        del rec
        gc.collect()

    # Phase 2: 前向传播窗口
    start = tz + 1
    end = min(tz + K, tau - 1)
    K_used = max(0, end - start + 1)

    hvp_calls = 0

    # 检查是否有足够的步骤记录 (only for non-lazy mode)
    if not use_lazy_loading and step_log is not None:
        if not step_log.has_range(start, end):
            logger.warning(f"Missing some steps in range [{start}, {end}]")

    # 预先获取所有需要的 eta 值 (轻量操作，可以批量获取)
    if use_lazy_loading and lazy_loader is not None:
        needed_steps = list(range(start, end + 1))
        eta_map = lazy_loader.get_etas_for_steps(needed_steps)
    else:
        eta_map = None

    # --- 历史参数重建 (via context manager) ---
    # Only construct when there are propagation steps and flag is enabled
    hist_ctx = HistoricalParamContext(
        model=model,
        start=start,
        end=end,
        tau=tau,
        enabled=use_historical_params and K_used > 0,
        step_log=step_log,
        lazy_loader=lazy_loader,
        use_lazy_loading=use_lazy_loading,
    )

    with hist_ctx:
        for s in range(start, end + 1):
            # 获取步骤记录
            if use_lazy_loading and lazy_loader is not None:
                # 优化: 当 batch_reconstructor 可用时，只需要 eta
                if batch_reconstructor is not None and cfg.hessian_mode in ("GGN", "fisher"):
                    if s not in eta_map:
                        logger.warning(f"Step {s} eta not found, skipping")
                        hist_ctx.advance(s)
                        continue
                    srec = StepRecord(
                        step_id=s,
                        eta=eta_map[s],
                        batch_id=s,
                        u=None,
                        gbar=None,
                        diag_H=None,
                        batch_data=None,
                    )
                else:
                    rec_dict = lazy_loader.load_single_step(s, include_tensors=True)
                    if rec_dict is None:
                        logger.warning(f"Step {s} not found, skipping")
                        hist_ctx.advance(s)
                        continue
                    srec = StepRecord(
                        step_id=rec_dict["step_id"],
                        eta=rec_dict["eta"],
                        batch_id=rec_dict["batch_id"],
                        u=rec_dict.get("u"),
                        gbar=rec_dict.get("gbar"),
                        diag_H=rec_dict.get("diag_H"),
                        batch_data=rec_dict.get("batch_data"),
                    )
                    del rec_dict
            else:
                srec = step_log[s] if step_log else None
                if srec is None:
                    logger.warning(f"Step {s} not found, skipping")
                    hist_ctx.advance(s)
                    continue

            # 计算 H[s] @ v (模型参数已是 θ[s] 或 θ[τ])
            with torch.enable_grad():
                hvp = hvp_apply(v, srec, cfg, model, loss_fn, batch_reconstructor)

            eta = srec.eta
            v = v - eta * hvp

            if cfg.damping > 0:
                v = v - eta * cfg.damping * v

            hvp_calls += 1
            hist_ctx.advance(s)

            if use_lazy_loading:
                del srec, hvp
                if hvp_calls % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # 记录 Phase 3 前的 v_norm
    v_norm_before_noise = float(v.norm().item())

    # Phase 4: 隐私噪声注入 (论文 Algorithm 1)
    noise_sigma = 0.0
    noise_injected = False

    if epsilon > 0:
        # 如果未提供 delta_det, 使用 v_norm 作为估计
        # WARNING: v_norm 不是 ||θ̂ - θ_ideal||₂ 的保证上界
        # 根据 Theorem 2, 应该提供一个经过证明的 Δ_det 上界
        # 使用 v_norm 可能导致 σ 过小，从而无法保证 (ε,δ)-certified unlearning
        if delta_det is None:
            logger.warning(
                "delta_det not provided, using v_norm as estimate. "
                "This may not be a valid upper bound for certified unlearning. "
                "For guaranteed (ε,δ)-certified unlearning, provide a proven Δ_det bound."
            )
            delta_det = v_norm_before_noise

        # 计算噪声标准差
        noise_sigma = compute_noise_sigma(delta_det, epsilon, delta)

        # 注入噪声
        v = inject_privacy_noise(v, noise_sigma)
        noise_injected = True

        logger.info(
            f"Phase 4: Injected privacy noise with σ={noise_sigma:.6f} "
            f"(ε={epsilon}, δ={delta})"
        )

    # 创建审计记录
    audit = AuditRecord(
        tz=tz,
        tau=tau,
        K_used=K_used,
        v_norm=v_norm_before_noise,
        hvp_calls=hvp_calls,
        mode=cfg.hessian_mode,
        damping=cfg.damping,
        noise_sigma=noise_sigma,
        noise_injected=noise_injected,
        epsilon=epsilon if noise_injected else 0.0,
        delta=delta if noise_injected else 0.0,
        used_historical_params=hist_ctx.active,
    )

    return v, audit


@torch.no_grad()
def apply_correction(
    v: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> None:
    """
    将校正向量v应用到参数: θ ← θ + v

    Args:
        v: 校正向量(展平的)
        params: 参数列表
    """
    offset = 0
    for p in params:
        n = p.numel()
        v_part = v[offset : offset + n].view_as(p)
        p.add_(v_part)
        offset += n

    if offset != v.numel():
        raise ValueError(f"Vector size mismatch: {offset} vs {v.numel()}")


# 辅助函数


def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    """将张量列表展平为单个向量"""
    if not tensors:
        return torch.empty(0)  # Use empty() for consistent dtype (float32)
    return torch.cat([t.view(-1) for t in tensors])


def _unflatten_like(
    flat: torch.Tensor,
    target: List[torch.Tensor],
) -> List[torch.Tensor]:
    """将展平的向量还原为目标张量的形状列表"""
    result = []
    offset = 0
    for t in target:
        n = t.numel()
        result.append(flat[offset : offset + n].view_as(t))
        offset += n
    return result


def compute_param_update_vector(
    old_params: List[torch.Tensor],
    new_params: List[torch.Tensor],
) -> torch.Tensor:
    """
    计算参数更新向量: u = new_params - old_params

    Args:
        old_params: 旧参数列表
        new_params: 新参数列表

    Returns:
        展平的更新向量
    """
    updates = [new - old for old, new in zip(old_params, new_params)]
    return _flatten(updates)


def clone_parameters(model: nn.Module) -> List[torch.Tensor]:
    """
    克隆模型参数并移到CPU以减少GPU显存压力 (Fix #3)

    Returns:
        List of cloned parameters on CPU
    """
    return [p.clone().detach().cpu() for p in model.parameters() if p.requires_grad]
