# OpenUnlearning / TOFU 代码导读 + 使用 Demo

本目录是对该仓库（TOFU 官方维护的新代码库 `open-unlearning`）的学习笔记和可运行示例。

## 一、配置参数是如何传递的？ —— Hydra 分层组合

整个项目用 [Hydra](https://hydra.cc) 管理配置，**完全不靠手写 argparse**。两个入口
`src/train.py` / `src/eval.py` 都用 `@hydra.main` 装饰，启动时把一棵 `DictConfig`
配置树注入 `main(cfg)`。

### 1. 配置组合（defaults 列表）
顶层配置（如 `configs/train.yaml`）通过 `defaults:` 列表把各组件拼起来：

```yaml
# configs/train.yaml
defaults:
  - model: Llama-3.2-3B-Instruct   # configs/model/*.yaml
  - trainer: finetune              # configs/trainer/*.yaml
  - data: finetune                 # configs/data/*.yaml
  - collator: DataCollatorForSupervisedDataset
  - eval: tofu                     # configs/eval/*.yaml
  - paths: default
  - experiment: null               # 可选实验包
```

每个组件是 `configs/<组>/<名>.yaml`。这样模型、训练器、数据、评测彼此解耦。

### 2. experiment 包：一次性覆盖一整组默认值
`configs/experiment/finetune/tofu/default.yaml` 顶部用 `# @package _global_`
+ `override`，把上面的默认值整组替换并设好实验超参：

```yaml
# @package _global_
defaults:
  - override /model: Llama-3.2-1B-Instruct
  - override /trainer: finetune
  - override /data/datasets@data.train: TOFU_QA_full
  - override /eval: tofu
trainer:
  args: {learning_rate: 1e-5, num_train_epochs: 5}
```

### 3. 命令行覆盖（最常用）
任意叶子参数都能在命令行点号覆盖；加号 `+` 表示新增不存在的 key：

```bash
python src/train.py experiment=finetune/tofu/default \
  model=Llama-3.2-1B-Instruct \
  trainer.args.num_train_epochs=10 \
  task_name=my_run \
  +model.peft_args.r=16          # + 表示新增字段
```

### 4. 变量插值与 @package
- 插值 `${...}`：如 `eval.tofu.forget_split: ${forget_split}`，多处共享同一值。
- `# @package eval.tofu.metrics.forget_quality`（写在指标 yaml 第一行）会把该文件
  内容塞进配置树的指定位置 —— 这就是“在 `eval/tofu.yaml` 里 import 一个指标名，
  它的配置就自动出现在 `metrics` 下”的原理。

### 5. 配置 → 代码：handler 注册表模式
配置里大量出现 `handler: XXX`。代码用「注册表」把字符串映射到具体类/函数：

| 组件 | 注册表 | 加载函数 |
|------|--------|----------|
| 模型 | `MODEL_REGISTRY` | `src/model/__init__.py: get_model` |
| 训练器 | `TRAINER_REGISTRY` | `src/trainer/__init__.py: load_trainer` |
| 评测器 | `EVALUATOR_REGISTRY` | `src/evals/__init__.py: get_evaluators` |
| 指标 | `METRICS_REGISTRY` | `src/evals/metrics/__init__.py: get_metrics` |

例如 `trainer.handler=GradDiff` → 找到 `GradDiff` 类并用 `trainer.args` /
`trainer.method_args` 实例化。新增方法只要写好类 + `_register_*` 即可。

---

## 二、实现了哪些评测指标？对应哪段代码

评测主流程：`src/evals/base.py: Evaluator.evaluate()` 遍历 `metrics`，每个指标是一个
被 `@unlearning_metric` 装饰（`src/evals/metrics/base.py`）的函数。每个指标 yaml 用
`defaults:` 声明它需要的 **数据集 / collator / 预计算指标 / 参照日志**，框架在
`base.py: prepare_kwargs_evaluate_metric` 里自动准备好再调用函数。

| 指标 | 含义 | 实现函数 | 文件 |
|------|------|----------|------|
| **Verbatim Probability** (`forget/retain_Q_A_Prob`) | 标准答案的归一化条件概率 | `probability`, `probability_w_options` | `src/evals/metrics/memorization.py` |
| **Verbatim ROUGE** (`*_Q_A_ROUGE`) | 生成文本与标准答案的 ROUGE-L recall | `rouge` | `memorization.py` |
| **Truth Ratio** (`forget/retain_Truth_Ratio`) | 正确答案 vs 扰动错误答案的似然比 | `truth_ratio` | `memorization.py` |
| **Exact Memorization (EM)** | 教师强制下 argmax 预测命中目标 token 比例 | `exact_memorization` | `memorization.py` |
| **Extraction Strength (ES)** | 需要多少前缀才能逐字续写出剩余答案 | `extraction_strength` | `memorization.py` |
| **Forget Quality** | forget 模型 vs retain 模型 truth-ratio 分布的 KS 检验 p 值 | `ks_test` | `src/evals/metrics/privacy.py` |
| **Model Utility** | 多个 retain/real-author/world-fact 指标的调和平均 | `hm_aggregate` | `src/evals/metrics/utility.py` |
| **PrivLeak** | forget vs retain 的 MIA AUC 相对差 | `privleak`, `rel_diff` | `privacy.py` |
| **Classifier Prob / Gibberish** | 用分类器判断生成是否乱码 | `classifier_prob` | `utility.py` |
| **6 种 MIA 攻击** | LOSS / ZLib / Reference / GradNorm / MinK / MinK++ | `mia_*` | `src/evals/metrics/mia/*.py` |
| **lm-eval-harness** | MMLU/GSM8K 等通用基准 | `LMEvalEvaluator` | `src/evals/lm_eval.py` |

底层公共算子（概率、生成相似度、批量推理）在 `src/evals/metrics/utils.py`：
`evaluate_probability` / `eval_text_similarity` / `run_batchwise_evals` /
`tokenwise_vocab_logprobs`。

TOFU 默认启用哪些指标，见 `configs/eval/tofu.yaml` 的 `defaults` 列表（其余被注释，
取消注释即可启用）。

**预计算依赖示例**：`model_utility` 的 yaml 里用
`.@pre_compute.retain_Q_A_Prob: retain_Q_A_Prob` 声明它依赖 9 个子指标；
`forget_quality` 声明它依赖 `forget_Truth_Ratio`（pre_compute）和 retain 模型日志
（`reference_logs` → `retain_logs_path`）。这正是评测时必须传 `retain_logs_path` 的原因。

---

## 三、三个使用 Demo

| 脚本 | 作用 |
|------|------|
| `1_finetune.sh` | 全参数微调一个 TOFU 目标模型 |
| `2_eval.sh` | 在 TOFU benchmark 上评测一个模型 |
| `3_lora_finetune.sh` | 用 LoRA 微调（含对框架的最小扩展） |
| `4_unlearn.sh` | 用**已有**遗忘方法（GradDiff）做 unlearning |
| `5_custom_unlearn.sh` | 运行**自定义**遗忘算法 BoundedGradDiff |

### 遗忘（unlearning）流程要点
- 入口同样是 `src/train.py`（`mode=unlearn`），但数据用 `data=unlearn`，会同时加载
  `forget` + `retain` 两个数据集；collator 把每个 batch 组织成
  `{"forget": {...}, "retain": {...}}`（见 `src/data/unlearn.py`）。
- 一个遗忘方法的本质 = 一个 trainer 类的 `compute_loss(model, inputs)`：
  - GradAscent：`loss = -forget_loss`
  - GradDiff：`loss = γ·(-forget_loss) + α·retain_loss`
  - NPO/SimNPO/DPO/RMU…各自不同。

### 如何加入你自己的遗忘算法（官方推荐做法）
`docs/contributing.md` 明确**鼓励**贡献自有方法。标准三步（本仓库已用
`BoundedGradDiff` 做了完整示例）：

1. **实现 trainer 类**（`src/trainer/unlearn/bounded_grad_diff.py`）：
   - 继承一个已有基类。继承 `GradDiff` 可白拿 `compute_retain_loss` /
     `gamma` / `alpha` / `ref_model`；继承更底层的 `UnlearnTrainer` 则更自由。
   - **唯一必须实现的函数是 `compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)`**。
     需要额外超参就再写 `__init__`（额外参数来自配置的 `method_args`）。
     其余（训练循环、保存、评测、deepspeed）都由父类 `UnlearnTrainer` /
     HF `Trainer` 处理，一般无需改动。
2. **注册**（`src/trainer/__init__.py`）：`_register_trainer(BoundedGradDiff)`。
3. **写配置**（`configs/trainer/BoundedGradDiff.yaml`）：`handler` 写类名，
   `method_args` 写你的超参。

**调用链**：`src/train.py` → `load_trainer()` 按 `handler` 从 `TRAINER_REGISTRY`
取出类，用 `trainer.args`(TrainingArguments) + `**method_args` 实例化 →
`trainer.train()` → HF Trainer 的训练循环每个 step 调用你的 `compute_loss`。

> 进一步（可选）：在 `community/methods/<你的方法>/` 放一个 `README.md` + `run.sh`
> 记录超参和复现命令，并把结果填到 `community/leaderboard.md`。

---

## 四、关于你的几个问题

**1. experiment 模块是不是"配好的配置清单"？不用它就得自己配每一部分吗？**
是的。`configs/experiment/**` 就是一份**预设好的整套实验清单**：它用
`# @package _global_` + `override` 一次性把 model/trainer/data/eval 等都换成某次实验
该用的值，并填好超参与变量插值（如 `forget_split` 联动到各处）。不用 experiment 也行，
但你就得在命令行/顶层 yaml 里**逐个**指定 `model=… trainer=… data=… eval=…` 以及所有
联动参数，很繁琐。所以惯例：**用 experiment 打底，再用命令行覆盖个别参数**。

**2. config 的整体结构（每部分配什么）**

| 配置组 | 路径 | 配的是什么 |
|--------|------|-----------|
| 顶层入口 | `configs/{train,unlearn,eval}.yaml` | `defaults:` 列表，决定加载哪些组件；`mode`、`task_name`、`seed` |
| model | `configs/model/*.yaml` | `model_args`(HF from_pretrained 参数、路径、dtype、attn)、`tokenizer_args`、`template_args`(对话模板)、(本仓库新增)`peft_args` |
| trainer | `configs/trainer/*.yaml` | `handler`(方法类名)、`args`(HF `TrainingArguments`：lr/epochs/batch/优化器/保存/eval 策略…)、`method_args`(方法自有超参) |
| data | `configs/data/*.yaml` + `data/datasets/*` | 用哪些数据集（finetune 只有 train；unlearn 有 forget+retain）、数据集 `handler`、`hf_args`(path/name/split)、question/answer_key、max_length |
| collator | `configs/collator/*.yaml` | 批处理拼接逻辑、padding 方式 |
| eval | `configs/eval/*.yaml` + `eval/<bench>_metrics/*` | `handler`(评测器)、启用哪些 `metrics`、各指标的数据集/`pre_compute`/参照日志、`forget/holdout_split`、`retain_logs_path`、`batch_size` |
| paths | `configs/paths/default.yaml` | 输出目录等路径 |
| hydra | `configs/hydra/*.yaml` | Hydra 运行时（日志、输出目录命名） |
| experiment | `configs/experiment/**` | 上面各组的**整套覆盖清单**，代表一次具体实验 |

**3. eval 跑完会有图表吗？**
**不会。** 评测只产出两个 JSON：`<Bench>_EVAL.json`（逐样本细分）和
`<Bench>_SUMMARY.json`（每个指标的聚合值），没有任何 matplotlib/绘图代码。
要图的话：训练侧 `trainer.args.report_to=tensorboard`（默认）会写 TensorBoard 曲线，
可 `tensorboard --logdir saves/`；想要评测柱状图/对比图需自己读 SUMMARY.json 画。

### 关于 LoRA 的说明
原框架**默认不支持 LoRA**（见 `docs/components.md`）。本 demo 做了一处最小、向后兼容
的扩展：

1. `src/model/__init__.py` 的 `get_model()` 在加载基座模型后，若 model 配置含
   `peft_args`，则调用新增的 `get_peft_lora_model()` 用 `peft` 包套 LoRA adapter；
   `peft_args.path` 可指向已有 adapter 用于续训/评测。
2. 新增 `configs/model/Llama-3.2-1B-Instruct-LoRA.yaml`（在基座配置上加 `peft_args`）。

需先 `pip install peft`。没有 `peft_args` 的旧配置完全不受影响。

> 运行前请确认已 `pip install ".[lm-eval]"` 并 `python setup_data.py --eval_logs`
> 下载评测所需的 retain 参照日志。

---

## 五、运行环境 / 常见坑（实跑前必看）

### 1. 前置依赖
```bash
pip install ".[lm-eval]"          # 安装框架 + 评测依赖
pip install peft                  # 仅 LoRA demo 需要
python setup_data.py --eval_logs       # 下载 retain 参照日志到 saves/eval/
```
- `setup_data.py --eval_logs` 下载的是各 retain 模型的评测日志（`saves/eval/.../TOFU_EVAL.json`）。
  `forget_quality` / `privleak` 这类指标要拿它和参照模型对比。**没有它不会崩**，但这些
  指标会是 `None`。不需要时把命令里的 `retain_logs_path=...` 整行删掉即可。

### 2. 指定 device（CPU / 某号 GPU）
训练和评测的控制方式不同：

| 场景 | 怎么指定 |
|------|----------|
| **评测** `src/eval.py` | 模型加载用 `model.model_args.device_map`（`eval.yaml` 默认 `cuda`）。可 `... model.model_args.device_map=cuda:0` 或 `device_map=cpu` |
| **训练** `src/train.py` | 配置里**没有** device_map，由 HF `Trainer`+`accelerate` 放卡。用环境变量控制 |

```bash
CUDA_VISIBLE_DEVICES=2 python src/train.py ...                # 单卡(推荐用环境变量锁物理卡)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml src/train.py ...   # 多卡
python src/train.py ... trainer.args.use_cpu=true            # CPU 训练(慢, 仅调试用)
```

> ⚠️ **CPU / 非 GPU 必看**：模型配置默认 `attn_implementation: flash_attention_2`
> + `torch_dtype: bfloat16`，flash-attn **只能在 GPU 跑**。CPU 时必须同时改：
> `model.model_args.attn_implementation=eager model.model_args.torch_dtype=float32`。
> 另外训练时模型先在 CPU 初始化再搬到 GPU，会打印一句 "attempting to use Flash
> Attention 2.0 with a model not initialized on GPU" —— **这是无害警告**，可忽略；
> 想消掉就用 `attn_implementation=sdpa`（性能几乎一致且不挑设备）。

> 🚨 **共享集群上的 DataParallel OOM**：用 `python src/train.py`（非 accelerate）启动时，
> 若**没设 `CUDA_VISIBLE_DEVICES`**，HF Trainer 会把所有可见 GPU 都拿来做
> `nn.DataParallel`——只要其中一张被别人占满，复制模型时就 `CUDA out of memory`
> （报错栈里会出现 `torch/nn/parallel/_functions.py ... gather along dimension 0`）。
> **务必先 `nvidia-smi` 选一张空闲卡并锁定单卡**：`CUDA_VISIBLE_DEVICES=<id> bash demos/4_unlearn.sh`。
> demos 里的训练/评测脚本已内置 `export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}`，
> 默认锁 0 号卡，可在外部覆盖。注意：eval（step 0）能跑通、训练第一步才 OOM，正是
> 因为评测单卡执行、训练才触发多卡复制。

### 3. 缓存路径 HF_HOME
- 设 **`HF_HOME`** 一个变量即可，模型和数据集都会用它：模型 → `$HF_HOME/hub`，
  数据集 → `$HF_HOME/datasets`（HF 标准布局）。
- 数据集是直接从 HF Hub 拉（`load_dataset("locuslab/TOFU", ...)`），**不需要**手动下到
  `./data/`；`configs/paths/default.yaml` 里的 `data_dir` 实际未被加载代码使用。
- 首次运行会下载模型+数据，属正常；之后命中缓存复用。

> 注：早期版本 `src/model/__init__.py` 误把 `cache_dir=$HF_HOME` 传给 `from_pretrained`，
> 导致模型落在 `$HF_HOME/models--...` 而非 `$HF_HOME/hub/`。本仓库已修复，现在符合 HF 约定。

### 4. 已修复的两个框架 bug（bf16 相关）
本仓库分支已修复，跑前 `git pull` 即可：
- **`TypeError: Got unsupported ScalarType BFloat16`**：bf16 模型评测时
  `evaluate_probability` 直接 `.numpy()` 崩溃（numpy 无 bfloat16）。已在
  `src/evals/metrics/utils.py` 改为 `.float().cpu().numpy()`。
- **HF_HOME 缓存层级**：见上条第 3 点。

### 5. task_name 注意
`task_name` 仅决定输出目录 `saves/<mode>/<task_name>/`，不影响算法逻辑。
**不同实验务必用不同 `task_name`**，否则会写进同一目录互相覆盖。
