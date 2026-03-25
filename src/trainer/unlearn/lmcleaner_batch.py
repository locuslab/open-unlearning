"""
LMCleaner Batch-Level Implementation: 批次级在线遗忘

论文: LMCleaner: Efficient and Certified Online Unlearning via Truncated Influence Propagation

算法特点 (论文 Algorithm 1):
- Phase 1: 初始偏差 v = θ[t_z+1] - θ[t_z] = -u[t_z]
- Phase 2: 截断影响传播 v[s+1] = (I - η_s H[s]) v[s], s ∈ [t_z+1, min(t_z+K, τ-1)]
- Phase 3: 影响移除 θ̂[τ] = θ[τ] - v (实现中: θ ← θ + v, 因为 v = -u)
- Phase 4: 隐私保护 θ̃[τ] = θ̂[τ] + N(0, σ²I) 实现 (ε,δ)-certified unlearning

存储复杂度: O((N/B) * p) vs 样本级的 O(N * p)

用法:
1. 预训练时使用TrainingLogger记录训练轨迹
2. 加载预训练模型和训练日志
3. 运行遗忘(自动应用参数校正)
4. 评估遗忘效果

实现说明:
- hessian_mode="fisher": 论文 Algorithm 1 使用的方法 Hv = g·(g^T v)
- hessian_mode="GGN": 使用二阶自动微分，更精确但计算量更大
- epsilon > 0: 启用 Phase 4 隐私噪声注入
- damping=0: 与论文一致 (damping>0 是额外的数值稳定化)
"""

import gc
import logging
import time
from pathlib import Path
from typing import List, Optional, Any

import torch
import torch.nn as nn

from trainer.unlearn.base import UnlearnTrainer
from .lmcleaner_core import (
    HVPConfig,
    AuditRecord,
    StepRecord,
    StepLog,
    compute_correction,
    apply_correction,
    compute_noise_sigma,
    inject_privacy_noise,
)
from ..training_logger import TrainingLogger, BatchReconstructor, LazyRecordLoader

logger = logging.getLogger(__name__)


class LMCleanerBatchLevel(UnlearnTrainer):
    """
    批次级LMCleaner在线遗忘

    实现论文 Algorithm 1 的完整四阶段算法:
    - Phase 1: 初始偏差计算
    - Phase 2: 截断影响传播 (K步)
    - Phase 3: 影响移除
    - Phase 4: 隐私噪声注入 (可选, 实现 certified unlearning)

    Args:
        training_log_dir: 训练日志目录(TrainingLogger输出)
        K: 截断窗口大小 (论文建议 K=64-1000)
        hessian_mode: HVP模式
            - "fisher": 论文 Algorithm 1 使用的 Fisher 近似 Hv = g·(g^T v)
            - "GGN": 广义 Gauss-Newton 近似 (更精确但计算量更大)
            - "diag": 对角 Hessian 近似
            - "exact": 精确 HVP (非常慢)
        damping: 阻尼系数λ (论文未包含，默认0以与论文一致)
        max_step: 最大步数(用于epoch-based评估,只考虑<=max_step的步骤)
        apply_immediately: 是否在初始化时立即应用遗忘(否则在train()时)
        audit_dir: 审计日志输出目录
        epsilon: 隐私参数 ε (>0 时启用 Phase 4 噪声注入)
        delta: 隐私参数 δ (默认 1e-5)
    """

    def __init__(
        self,
        training_log_dir: str,
        K: int = 800,
        hessian_mode: str = "fisher",  # 默认使用论文的 Fisher 近似
        damping: float = 0.0,  # 默认不添加阻尼，与论文一致
        max_step: Optional[int] = None,
        apply_immediately: bool = False,
        audit_dir: Optional[str] = None,
        finetune_dataset_path: str = "locuslab/TOFU",
        finetune_dataset_name: str = "full",
        finetune_dataset_split: str = "train",
        # Phase 4: 隐私噪声参数
        epsilon: float = 0.0,  # 默认不注入噪声
        delta: float = 1e-5,
        # ! 历史参数重建: 在 θ[s] 处计算 HVP (论文 Algorithm 1)
        # ! 额外占用 ~1x 模型参数量 GPU 显存 + 读取 u[t] 的 IO 开销
        # ! 显存紧张时设为 False 退化为在 θ[τ] 处近似计算
        # ! 详见 docs/historical_params_hvp.md
        use_historical_params: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 保存原始finetune数据集参数 (用于批次重建)
        self.finetune_dataset_path = finetune_dataset_path
        self.finetune_dataset_name = finetune_dataset_name
        self.finetune_dataset_split = finetune_dataset_split
        self._finetune_dataset = None  # 延迟加载

        self.training_log_dir = Path(training_log_dir)
        self.K = K
        self.hessian_mode = hessian_mode
        self.damping = damping
        self.max_step = max_step
        self.apply_immediately = apply_immediately

        # Phase 4: 隐私参数
        self.epsilon = epsilon
        self.delta = delta

        # 历史参数重建开关
        self.use_historical_params = use_historical_params

        # HVP配置
        self.hvp_config = HVPConfig(
            mode=hessian_mode,
            damping=damping,
            device=str(self.model.device) if hasattr(self.model, "device") else "cuda",
            dtype=next(self.model.parameters()).dtype,
        )

        # 审计日志
        self.audit_dir = Path(audit_dir) if audit_dir else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        # 加载训练日志元数据 (不加载tensor，使用懒加载以避免内存爆炸)
        self.training_logger = TrainingLogger(log_dir=str(self.training_log_dir))
        self.training_logger.load_from_disk(load_tensors=False)

        # 创建懒加载器用于按需加载 step records
        self.lazy_loader = LazyRecordLoader(self.training_log_dir)
        self.lazy_loader.build_index()

        # 创建批次重建器(用于轻存储模式)
        # 注意: dataset和data_collator会在train()方法中设置
        self.batch_reconstructor = None

        # 审计记录列表
        self.audit_records: List[AuditRecord] = []

        # 是否已应用遗忘
        self.unlearning_applied = False

        logger.info(
            f"LMCleanerBatchLevel initialized: K={K}, hessian_mode={hessian_mode}, "
            f"damping={damping}, epsilon={epsilon}, delta={delta}, "
            f"use_historical_params={use_historical_params}"
        )

        # 立即应用遗忘(如果设置)
        if self.apply_immediately:
            self._apply_unlearning()

    def _get_forget_batch_ids(self) -> List[Any]:
        """
        获取需要遗忘的批次ID列表

        从forget_dataset中提取批次ID
        假设数据集有batch_id属性,或者使用数据索引作为batch_id
        """
        forget_batch_ids = []

        # 尝试多种属性名获取 forget 数据集
        forget_ds = None
        for attr in ["forget_dataset", "forget"]:
            if hasattr(self.train_dataset, attr):
                forget_ds = getattr(self.train_dataset, attr)
                break

        if forget_ds is not None:
            # 如果数据集有batch_id属性
            if hasattr(forget_ds, "batch_ids"):
                forget_batch_ids = forget_ds.batch_ids
            # 否则使用索引作为batch_id
            else:
                # 获取forget样本的原始索引
                if hasattr(forget_ds, "original_indices"):
                    forget_batch_ids = forget_ds.original_indices
                else:
                    # 假设batch_id就是数据集索引
                    forget_batch_ids = list(range(len(forget_ds)))

        logger.info(f"Found {len(forget_batch_ids)} forget batches")
        return forget_batch_ids

    def _get_forget_steps(self, forget_batch_ids: List[Any]) -> List[int]:
        """
        获取forget批次对应的训练步骤

        Args:
            forget_batch_ids: 批次ID列表 (实际上是样本索引)

        Returns:
            步骤ID列表
        """
        forget_steps = []
        forget_set = set(forget_batch_ids)

        # 优先使用 lazy_loader.sample_indices 来查找包含 forget 样本的步骤
        sample_indices = None
        if hasattr(self, "lazy_loader") and hasattr(self.lazy_loader, "sample_indices"):
            sample_indices = self.lazy_loader.sample_indices
        elif hasattr(self.training_logger, "sample_indices_per_step"):
            sample_indices = self.training_logger.sample_indices_per_step

        if sample_indices:
            for step_id, sample_list in sample_indices.items():
                step_id_int = int(step_id) if isinstance(step_id, str) else step_id
                for sample_idx in sample_list:
                    if sample_idx in forget_set:
                        forget_steps.append(step_id_int)
                        break  # 只需要知道这个步骤包含 forget 样本
            logger.info(f"Found {len(forget_steps)} forget steps via sample_indices")
        else:
            # 回退到旧的 batch_index 查找方式
            for batch_id in forget_batch_ids:
                step_id = self.training_logger.get_batch_step(batch_id)
                if step_id is not None:
                    forget_steps.append(step_id)
                else:
                    logger.debug(f"Batch {batch_id} not found in training log")
            logger.info(f"Found {len(forget_steps)} forget steps via batch_index")

        return sorted(set(forget_steps))  # 去重并排序

    def _apply_unlearning(self):
        """
        应用在线遗忘

        对每个forget批次:
        1. 定位其训练步骤tz
        2. 计算参数校正向量v (前向K步传播)
        3. 应用校正: θ ← θ + v
        4. 记录审计信息

        使用懒加载方式按需加载步骤数据，避免内存爆炸。
        """
        if self.unlearning_applied:
            logger.warning("Unlearning already applied, skipping")
            return

        # 获取forget批次和对应的步骤
        forget_batch_ids = self._get_forget_batch_ids()
        forget_steps = self._get_forget_steps(forget_batch_ids)

        if not forget_steps:
            logger.warning("No forget steps found, skipping unlearning")
            return

        # 初始化批次重建器(如果需要)
        if self.batch_reconstructor is None and hasattr(self, "data_collator"):
            if self._finetune_dataset is None:
                logger.info(f"Loading original finetune dataset: {self.finetune_dataset_path}/{self.finetune_dataset_name}")
                self._finetune_dataset = self._load_finetune_dataset()
                logger.info(f"Loaded finetune dataset with {len(self._finetune_dataset)} samples")

            self.batch_reconstructor = BatchReconstructor(
                training_logger=self.training_logger,
                dataset=self._finetune_dataset,
                data_collator=self.data_collator,
            )
            logger.info("Initialized BatchReconstructor for batch data reconstruction")

        # 当前步骤(训练结束时的步骤，或用户指定的max_step)
        tau = self.max_step if self.max_step is not None else self.training_logger.current_step

        # 过滤forget steps
        original_count = len(forget_steps)
        forget_steps = [s for s in forget_steps if s <= tau]
        if len(forget_steps) < original_count:
            logger.info(
                f"Filtered forget steps: {original_count} -> {len(forget_steps)} "
                f"(max_step={tau})"
            )

        if not forget_steps:
            logger.warning("No forget steps after filtering, skipping unlearning")
            self.unlearning_applied = True
            return

        params = [p for p in self.model.parameters() if p.requires_grad]

        logger.info(f"Applying unlearning for {len(forget_steps)} batches")
        logger.info(f"Target step (tau): {tau}, K: {self.K}")

        for i, tz in enumerate(forget_steps):
            logger.info(f"Processing forget step {i + 1}/{len(forget_steps)}: tz={tz}")
            _step_start_time = time.perf_counter()

            try:
                # 只加载初始记录 (tz 对应的记录，用于 Phase 1)
                initial_rec_dict = self.lazy_loader.load_single_step(tz, include_tensors=True)

                if initial_rec_dict is None:
                    logger.warning(f"No record found for step {tz}, skipping")
                    continue

                # 创建初始 StepRecord
                initial_record = StepRecord(
                    step_id=initial_rec_dict["step_id"],
                    eta=initial_rec_dict["eta"],
                    batch_id=initial_rec_dict["batch_id"],
                    u=initial_rec_dict.get("u"),
                    gbar=initial_rec_dict.get("gbar"),
                    diag_H=initial_rec_dict.get("diag_H"),
                    batch_data=initial_rec_dict.get("batch_data"),
                )
                del initial_rec_dict  # 释放原始 dict

                # 调用核心计算函数 (使用 lazy loading 模式)
                v, audit = compute_correction(
                    tz=tz,
                    tau=tau,
                    K=self.K,
                    step_log=None,  # 不使用预加载的 step_log
                    cfg=self.hvp_config,
                    model=self.model,
                    batch_reconstructor=self.batch_reconstructor,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    lazy_loader=self.lazy_loader,  # 传入 lazy_loader 用于按需加载
                    initial_record=initial_record,  # 传入初始记录
                    use_historical_params=self.use_historical_params,
                )

                # Phase 3: 应用校正
                apply_correction(v, params)

                # Record wall time for this forget step
                _step_elapsed_ms = (time.perf_counter() - _step_start_time) * 1000
                audit.wall_time_ms = _step_elapsed_ms
                audit["wall_time_ms"] = _step_elapsed_ms
                self.audit_records.append(audit)

                logger.info(
                    f"Applied correction for step {tz}: "
                    f"v_norm={audit.v_norm:.6f}, K_used={audit.K_used}, hvp_calls={audit.hvp_calls}, "
                    f"time={_step_elapsed_ms:.0f}ms"
                    + (f", noise_σ={audit.noise_sigma:.6f}" if audit.noise_injected else "")
                )

                # 清理内存
                del initial_record, v
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to apply correction for step {tz}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if self.audit_dir:
            self._save_audit_records()

        self.unlearning_applied = True
        logger.info("Unlearning applied successfully")

    def _save_audit_records(self):
        """保存审计记录到磁盘"""
        import json

        audit_file = self.audit_dir / "audit_records.json"

        # 转换为可序列化的格式
        serializable_records = [dict(record) for record in self.audit_records]

        with open(audit_file, "w") as f:
            json.dump(serializable_records, f, indent=2)

        logger.info(f"Saved {len(self.audit_records)} audit records to {audit_file}")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        重写损失计算

        对于LMCleaner,主要是通过参数校正实现遗忘,
        训练阶段可以选择性地在retain数据上微调
        """
        # 如果有retain数据,计算retain loss
        if "retain" in inputs:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_outputs = model(**retain_inputs)
            loss = retain_outputs.loss
        else:
            # 返回一个很小的损失
            loss = torch.tensor(0.0, device=next(model.parameters()).device)

        return (loss, None) if return_outputs else loss

    def _load_finetune_dataset(self):
        """
        加载原始 finetune 数据集用于批次重建

        Returns:
            可索引的数据集对象
        """
        from datasets import load_dataset
        from data.utils import preprocess_chat_instance

        # 加载 HuggingFace 数据集
        hf_dataset = load_dataset(
            self.finetune_dataset_path,
            name=self.finetune_dataset_name,
            split=self.finetune_dataset_split,
        )

        # 获取 tokenizer 和模板参数
        tokenizer = self.tokenizer
        # 从 config 中获取 template_args (如果有的话)
        template_args = getattr(self, "template_args", {
            "sys_prompt": None,
            "incontext_prompt": None,
            "incontext_response": None,
        })

        # 创建简单的数据集包装器
        class SimpleQADataset:
            def __init__(self, data, tokenizer, template_args, max_length=512):
                self.data = data
                self.tokenizer = tokenizer
                self.template_args = template_args
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                question = item["question"]
                answer = item["answer"]

                # 使用 preprocess_chat_instance 进行 tokenization
                tokenized = preprocess_chat_instance(
                    self.tokenizer,
                    self.template_args,
                    [question],
                    [answer],
                    self.max_length,
                    predict_with_generate=False,
                )

                return {
                    "input_ids": tokenized["input_ids"],
                    "labels": tokenized["labels"],
                    "attention_mask": tokenized["attention_mask"],
                    "index": idx,
                }

        return SimpleQADataset(hf_dataset, tokenizer, template_args)

    def train(self, resume_from_checkpoint=None):
        """
        重写训练方法

        在训练开始前应用遗忘,然后可选地在retain数据上微调

        Args:
            resume_from_checkpoint: 兼容父类接口，LMCleaner 不使用此参数
        """
        # 应用遗忘(如果尚未应用)
        if not self.unlearning_applied:
            self._apply_unlearning()

        # 如果有retain数据且需要微调,调用父类训练
        if hasattr(self.train_dataset, "retain_dataset"):
            logger.info("Fine-tuning on retain data...")
            return super().train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            logger.info("No retain data, skipping training")
            # 返回空的训练结果
            return None

    def save_model(self, output_dir: str):
        """
        保存遗忘后的模型

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 确保已应用遗忘
        if not self.unlearning_applied:
            self._apply_unlearning()

        # 保存模型
        self.model.save_pretrained(output_dir)

        # 保存tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 保存遗忘元信息
        import json

        meta = {
            "method": "LMCleanerBatchLevel",
            "K": self.K,
            "hessian_mode": self.hessian_mode,
            "damping": self.damping,
            "max_step": self.max_step,
            "num_forget_batches": len(self.audit_records),
            "total_corrections": sum(r["hvp_calls"] for r in self.audit_records),
        }

        meta_file = output_dir / "unlearning_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved unlearned model to {output_dir}")


# 便捷函数


def run_lmcleaner_batch_unlearning(
    model: nn.Module,
    training_log_dir: str,
    forget_batch_ids: List[Any],
    K: int = 800,
    hessian_mode: str = "GGN",
    damping: float = 1e-4,
    output_dir: Optional[str] = None,
) -> nn.Module:
    """
    运行批次级LMCleaner遗忘(独立函数)

    使用懒加载方式按需加载步骤数据，避免内存爆炸。

    Args:
        model: 预训练模型
        training_log_dir: 训练日志目录
        forget_batch_ids: 需要遗忘的批次ID列表
        K: 截断窗口
        hessian_mode: HVP模式
        damping: 阻尼系数
        output_dir: 输出目录(可选)

    Returns:
        遗忘后的模型
    """
    from .lmcleaner_core import StepLog

    training_log_dir = Path(training_log_dir)

    # 加载训练日志元数据 (不加载tensor)
    training_logger = TrainingLogger(log_dir=str(training_log_dir))
    training_logger.load_from_disk(load_tensors=False)

    # 创建懒加载器
    lazy_loader = LazyRecordLoader(training_log_dir)
    lazy_loader.build_index()

    # 获取forget步骤
    forget_steps = []
    for batch_id in forget_batch_ids:
        step_id = training_logger.get_batch_step(batch_id)
        if step_id is not None:
            forget_steps.append(step_id)

    # HVP配置
    hvp_config = HVPConfig(
        mode=hessian_mode,
        damping=damping,
        device=str(model.device) if hasattr(model, "device") else "cuda",
        dtype=next(model.parameters()).dtype,
    )

    # 当前步骤
    tau = training_logger.current_step

    # 参数列表
    params = [p for p in model.parameters() if p.requires_grad]

    # 应用遗忘(使用懒加载)
    audit_records = []

    for tz in forget_steps:
        # 计算需要加载的步骤范围
        start_step = tz
        end_step = min(tz + K, tau - 1)
        needed_steps = list(range(start_step, end_step + 1))

        # 按需加载这些步骤的记录
        records = lazy_loader.load_steps(needed_steps, include_tensors=True)

        if not records:
            logger.warning(f"No records found for step {tz}, skipping")
            continue

        # 创建临时的 StepLog
        temp_step_log = StepLog(max_size=len(records) + 10)
        for rec_dict in records:
            step_record = StepRecord(
                step_id=rec_dict["step_id"],
                eta=rec_dict["eta"],
                batch_id=rec_dict["batch_id"],
                u=rec_dict.get("u"),
                gbar=rec_dict.get("gbar"),
                diag_H=rec_dict.get("diag_H"),
            )
            temp_step_log.add(step_record)

        v, audit = compute_correction(
            tz=tz,
            tau=tau,
            K=K,
            step_log=temp_step_log,
            cfg=hvp_config,
            model=model,
        )

        apply_correction(v, params)
        audit_records.append(audit)

        logger.info(f"Applied correction for step {tz}: v_norm={audit['v_norm']:.6f}")

        # 清理内存
        del records, temp_step_log, v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存模型(如果指定输出目录)
    if output_dir:
        output_dir = Path(output_dir)
        model.save_pretrained(output_dir)
        logger.info(f"Saved unlearned model to {output_dir}")

    return model
