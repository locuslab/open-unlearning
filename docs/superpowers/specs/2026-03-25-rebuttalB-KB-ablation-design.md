# 实验B：`(K, B)` 联合 ablation + forget 数据分布分析

> Status: **In Progress**
> Created: 2026-03-25
> Branch: `rebuttalB`
> Worktree: `/tmp/rebuttalB`

---

## 1. 实验目标

系统回答以下四类问题，形成论文中的一组完整 ablation 证据链：

1. **`K` 的影响** — 截断传播窗口对 utility、forget quality、MIA、运行时间的影响
2. **`B` 的影响** — 训练 batch size 对 batch-level unlearning 的行为影响
3. **`(K, B)` 交互效应** — 是否存在明显交互或 Pareto front
4. **forget 数据分布** — forget 样本在训练 mini-batch 中的分布特征

---

## 2. 核心假设

- **H1**: K 的收益存在饱和区间
- **H2**: 更大的 B 增大 batch-level 遗忘的 collateral damage
- **H3**: K 与 B 存在交互效应
- **H4**: forget set 的 batch 分布是解释性能差异的关键中介变量

---

## 3. 实验设置

### 模型
- Llama-3.2-1B-Instruct

### 训练协议
- **单 epoch**，AdamW，lr=1e-5，weight_decay=0.01，seed=0
- 每个 B 对应独立训练日志
- Collator: `DataCollatorForSupervisedDatasetwithIndex`

### Unlearning
- 方法: LMCleaner batch-level
- Forget set: TOFU forget01 (40 samples)
- 固定 Hessian mode、damping、privacy/noise

---

## 4. 变量设计

### B (训练 effective batch size)

| B | micro_batch | grad_accum | steps/epoch | train_logs 大小 |
|---|:-:|:-:|:-:|:-:|
| 8 | 8 | 1 | 500 | 1.2 TB |
| 16 | 4 | 4 | 250 | 588 GB |
| 32 | 32 | 1 | 125 | 291 GB |
| 64 | 64 | 1 | 62 | 148 GB |
| 128 | 64 | 2 | 31 | 74 GB |
| 256 | 64 | 4 | 15 | 37 GB |

**注意**: B=16 使用已有训练日志（sample_indices 已修复为完整 16 indices/step），其余为新训。

### K (截断传播窗口)
K 的上界受 steps/epoch 限制：

| B | max K | 可用 K 值 |
|---|:-:|---|
| 8 | 500 | 10, 25, 50, 100, 200, 500 |
| 16 | 250 | 10, 25, 50, 100, 200 |
| 32 | 125 | 10, 25, 50, 100 |
| 64 | 62 | 10, 25, 50 |
| 128 | 31 | 10, 25 |
| 256 | 15 | 10 |

---

## 5. 评价指标

### 5.1 Utility / Forget / MIA
通过 TOFU eval pipeline (`src/eval.py`) 输出 `TOFU_SUMMARY.json`：
- `model_utility` ↑
- `forget_Q_A_ROUGE` ↓
- `forget_truth_ratio`
- `extraction_strength` ↓
- `privleak` ↓ (更负更好)
- `mia_min_k` → 接近 0.5

### 5.2 Efficiency
- wall-clock time (`efficiency_metrics.json`)
- HVP calls (audit records)
- GPU memory peak

### 5.3 Forget 分布统计
脚本: `scripts/rebuttalB_forget_distribution.py`

| 指标 | 说明 |
|------|------|
| `num_affected_batches` | 包含 ≥1 个 forget 样本的 batch 数 |
| `forget_per_batch_mean/median/max` | 每个受影响 batch 中 forget 样本数 |
| `collateral_ratio` | `1 - (#forget / B)`，benign 样本被误伤比例 |
| `dispersion_gini` | 分散度 Gini 系数 |
| `dispersion_entropy` | 分散度 Shannon 熵 |

---

## 6. 待解决问题

### 已解决
- [x] TrainingLogger 只记录最后一个 micro-batch 的 indices → 修复为累积所有 micro-batch
- [x] B=16 旧日志 sample_indices 不完整 → 通过 seed=0 重放 DataLoader 重建
- [x] B=16 旧日志只保留第一个 epoch，释放存储

### 未解决
1. **LMCleaner 调用脚本** — 每个 (K, B) 组合需要指向正确的 train_log_dir 和 model checkpoint
2. **Distance to Retrain 指标** — 需要对每个 B 做"删除 forget 后重训"的 gold reference
3. **Order sensitivity** — 定义不清，需要明确计算方法
4. **单 epoch 模型的 eval 基线** — 需要先跑每个 B 的 full model eval 作为参考上界
5. **K 上界与 B 矛盾** — 大 B 下可用 K 值极少，实验矩阵不是完整矩形

---

## 7. 执行策略

### Stage 1: 探路（单 seed）
1. 固定 B=16（已有数据），扫 K ∈ {10, 25, 50, 100, 200}
2. 固定 K=50，扫 B ∈ {8, 16, 32, 64}（B=128/256 的 K=50 超上界）
3. 跑 forget 分布统计

### Stage 2: 补全矩阵
跑满所有合法 (K, B) 组合

### Stage 3: 补 seed
对关键组合补到 3 seeds

---

## 8. 训练数据状态

| B | 目录 | 状态 |
|---|------|------|
| 8 | `/workspace/saves/train_logs/rebuttalB_B8_seed0` | ✅ 完成 |
| 16 | `/workspace/saves/train_logs/llama32_1b_tofu_safe` | ✅ 修复 (epoch 0 only) |
| 32 | `/workspace/saves/train_logs/rebuttalB_B32_seed0` | ✅ 完成 |
| 64 | `/workspace/saves/train_logs/rebuttalB_B64_seed0` | ✅ 完成 |
| 128 | `/workspace/saves/train_logs/rebuttalB_B128_seed0` | ✅ 完成 |
| 256 | `/workspace/saves/train_logs/rebuttalB_B256_seed0` | ✅ 完成 |

总存储: ~2.3 TB

---

## 9. 论文输出

### 主表
完整 (K, B) ablation 表，含 utility、forget、MIA、time、分布统计

### 图组
- Figure B1: K 曲线（每个 B 一条线）
- Figure B2: 分布图（affected batches vs B, collateral ratio vs B, forget-per-batch histogram）
- Figure B3: (B × K) 热力图
- Figure B4: 代表性组合重点展示

### 叙述结构
1. K ablation — 截断窗口的收益/成本折中
2. B ablation + forget 分布 — batch-level unlearning 对 mini-batch 结构的敏感性
3. 联合分析 — 推荐默认配置

---

## 10. 代码变更

### 已提交 (branch: rebuttalB)
- `src/trainer/base.py` — 修复 TrainingLogger gradient accumulation indices 累积
- `scripts/rebuttalB_train.sh` — B ablation 训练脚本
- `scripts/rebuttalB_forget_distribution.py` — forget 分布分析脚本

### 待实现
- LMCleaner (K, B) sweep 脚本
- Eval pipeline 脚本
- 结果汇总和绘图脚本
