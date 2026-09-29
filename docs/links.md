# 🔗 Links and References

Links to research papers and resources corresponding to implemented features in this repository. Please feel free to fill in any missing references!

[`OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics`](https://arxiv.org/abs/2506.12618) introduces

- The technical report on OpenUnlearning, its design, features and other details.
- A meta-evaluation framework to benchmark unlearning evaluations on a set of 450+ open sourced models.
- Results benchmarking 8 diverse unlearning methods in one place using 10 evaluation metrics on TOFU.

---

## 📌 Table of Contents

- [🔗 Links and References](#-links-and-references)
  - [📌 Table of Contents](#-table-of-contents)
  - [📗 Implemented Methods](#-implemented-methods)
  - [📘 Benchmarks](#-benchmarks)
  - [📙 Evaluation Metrics](#-evaluation-metrics)
  - [🌐 Useful Links](#-useful-links)
    - [📚 Surveys](#-surveys)
    - [🐙 Other GitHub Repositories](#-other-github-repositories)

---

## 📗 Implemented Methods

| Method               | Resource                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GradAscent, GradDiff | Naive baselines found in many papers including MUSE, TOFU etc.                                                                                                                               |
| NPO                  | Paper[📄](https://arxiv.org/abs/2404.05868), Code [🐙](https://github.com/licong-lin/negative-preference-optimization)                                                                             |
| SimNPO               | Paper[📄](https://arxiv.org/abs/2410.07163), Code [🐙](https://github.com/OPTML-Group/Unlearn-Simple)                                                                                              |
| IdkDPO               | TOFU ([📄](https://arxiv.org/abs/2401.06121))                                                                                                                                                   |
| RMU                  | WMDP paper ([🐙](https://github.com/centerforaisafety/wmdp/tree/main/rmu), [🌐](https://www.wmdp.ai/)), later used in G-effect ([🐙](https://github.com/tmlr-group/G-effect/blob/main/dataloader.py)) |
| UNDIAL               | Paper[📄](https://arxiv.org/pdf/2402.10052), Code [🐙](https://github.com/dong-river/LLM_unlearning/tree/main)                                                                                     |
| AltPO                | Paper[📄](https://arxiv.org/pdf/2409.13474), Code [🐙](https://github.com/molereddy/Alternate-Preference-Optimization)                                                                             |
| SatImp               | Paper[📄](https://arxiv.org/pdf/2505.11953), Code [🐙](https://github.com/Puning97/SatImp-for-LLM-Unlearning)                                                                                      |
| WGA (G-effect)       | Paper[📄](https://arxiv.org/pdf/2502.19301), Code [🐙](https://github.com/tmlr-group/G-effect)                                                                                                     |
| CE-U (Cross-Entropy unlearning)       | Paper[📄](https://arxiv.org/pdf/2503.01224)                                                                                                     |
| PDU                  | Paper [📄](https://arxiv.org/abs/2506.05314) |
| T3-Unlearning               | Paper[📄](https://arxiv.org/abs/2602.10217), Code [🐙](https://github.com/jacob-block/T3-Unlearning)                                                                                     |


---

## 📘 Benchmarks

| Benchmark | Resource                                 |
| --------- | ---------------------------------------- |
| TOFU      | Paper[📄](https://arxiv.org/abs/2401.06121) |
| MUSE      | Paper[📄](https://arxiv.org/abs/2407.06460) |
| WMDP      | Paper[📄](https://arxiv.org/abs/2403.03218) |

---

## 📙 Evaluation Metrics

| Metric                                                                       | Resource                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Verbatim Probability / ROUGE, simple QA-ROUGE                                | Naive metrics found in many papers including MUSE, TOFU etc.                                                                                                                                                                            |
| Membership Inference Attacks (LOSS, ZLib, Reference, GradNorm, MinK, MinK++) | MIMIR ([🐙](https://github.com/iamgroot42/mimir)), MUSE ([📄](https://arxiv.org/abs/2407.06460))                                                                                                                                              |
| PrivLeak                                                                     | MUSE ([📄](https://arxiv.org/abs/2407.06460))                                                                                                                                                                                              |
| Forget Quality, Truth Ratio, Model Utility                                   | TOFU ([📄](https://arxiv.org/abs/2401.06121))                                                                                                                                                                                              |
| Extraction Strength (ES)                                                     | Carlini et al., 2021 ([📄](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting)), used for unlearning in Wang et al., 2025 ([📄](https://openreview.net/pdf?id=wUtCieKuQU))                                    |
| Exact Memorization (EM)                                                      | Tirumala et al., 2022 ([📄](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html)), used for unlearning in Wang et al., 2025 ([📄](https://openreview.net/pdf?id=wUtCieKuQU)) |
| lm-evaluation-harness                                                        | Repository: [💻](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)                                                                                                                                                                        |

---

## 🌐 Useful Links

### 📚 Surveys

- [Machine Unlearning in 2024](https://ai.stanford.edu/~kzliu/blog/unlearning)
- [Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/abs/2402.08787)

### 🐙 Other GitHub Repositories

- [TOFU Benchmark (original)](https://github.com/locuslab/tofu)
- [MUSE Benchmark (original)](https://github.com/swj0419/muse_bench)
- [Unlearning Comparator](https://github.com/gnueaj/Machine-Unlearning-Comparator)
- [Awesome LLM Unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)
- [Awesome Machine Unlearning](https://github.com/tamlhp/awesome-machine-unlearning)
- [Awesome GenAI Unlearning](https://github.com/franciscoliu/Awesome-GenAI-Unlearning)
