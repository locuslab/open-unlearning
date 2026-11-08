# K-FORGE: Fisher-Guided Initialization for Preference-Based LLM Unlearning (Findings of IJCNLP-AACL 2026)

- Authors: Daniil Moskovskiy (AIRI, Skoltech), Alexander Panchenko (Skoltech, AIRI)
- Paper code and experiments: https://github.com/etomoscow/kron_unlearn

## Method

K-FORGE (Kronecker-Fisher One-shot Rank-r Generalized Erasure) is a training-free, one-shot unlearning edit. For every regex-selected `torch.nn.Linear` layer (default: MLP `down_proj`), it estimates empirical K-FAC-style Kronecker factors on forget and retain calibration batches — `A = E[x xᵀ]` from layer inputs, `B = E[g gᵀ]` from output gradients — simultaneously diagonalizes the forget/retain pairs, applies a retain-aware Wiener filter with a retain budget (`lambda_tradeoff`), truncates to rank `r`, and folds the negative edit into the checkpoint. The only backward passes are the calibration passes; the edit itself is closed-form. Calibration currently runs single-process.

K-FORGE can be used in two ways:

1. **One-shot unlearning**: stronger edits lower Forget Q/A Probability but progressively spend Model Utility, so the raw edited model is a tunable trade-off rather than a free lunch.
2. **Initialization for iterative unlearning (recommended; the paper's headline usage)**: run NPO or SimNPO from the edited checkpoint. With setup time and FLOPs fully charged, K-FORGE-initialized NPO reduces Forget Q/A Probability by 22.3% (p = 0.007) and improves Model Utility while using less total wall time and compute than training from the pretrained checkpoint (TOFU `forget10`, Llama-3.2-1B-Instruct, n = 3).

## Setup

`run.sh` reproduces the two-stage flow on TOFU `forget10` with Llama-3.2-1B-Instruct:

- **Stage 1** computes the one-shot K-FORGE edit (`wiener_v2`, paper configuration) and saves the edited checkpoint under `saves/unlearn/tofu_kforge_edit`.
- **Stage 2** runs downstream unlearning from the edited checkpoint. Set `DOWNSTREAM_TRAINER=SimNPO` to use SimNPO instead of the default NPO; for a one-shot-only run, stop after stage 1.

```bash
bash community/methods/KFORGE/run.sh

# SimNPO downstream instead of NPO
DOWNSTREAM_TRAINER=SimNPO bash community/methods/KFORGE/run.sh
```

Strength `alpha` (`trainer.method_args.strength`) controls the utility/forgetting trade-off of the edit; the paper reports α ∈ {0.45, 0.60} for initialized runs and α ∈ [0.1, 0.8] for the one-shot sweep. Starting values around 0.45 are a reasonable default on 1B-scale models; increase `rank`, `strength`, and `max_calibration_batches` for stronger edits after checking memory use.

## Hyperparameters (paper configuration, TOFU `forget10`, Llama-3.2-1B-Instruct)

| Argument | Value | Notes |
|---|---|---|
| `target_modules_regex` | `.*mlp\.down_proj$` | Full-match regex over module names |
| `rank` | 2 | Truncated SVD rank per layer |
| `strength` | 0.45 (also 0.60) | Multiplier α on the negative edit |
| `edit_variant` | `wiener_v2` | Retain-aware Wiener edit (theorem-backed) |
| `lambda_tradeoff` | 0.01 | Retain-Fisher budget (used by `wiener_v2` only) |
| `max_target_modules` | 4 | Memory guard; set `null` for all matched modules |
| `max_calibration_batches` | 8 | Forget and retain batches for factor estimation |
| `edit_weight_dtype` | `float32` | Upcasts the model to fp32 before folding the edit, so small α·δ survive the `add_`; `preserve` keeps bf16, where a tiny α·δ can round to zero |
| `damping` | 1e-4 | Relative Tikhonov damping before Cholesky |

## Results

Primary endpoint: **Forget Q/A Probability** (probability of gold forget answers; lower is better). TOFU `forget10`, Llama-3.2-1B-Instruct, n = 3 seeds, mean ± std.

| Steps | Method | Init | Utility ↑ | Forget Prob. ↓ | Extraction ↓ |
|---|---|---|---|---|---|
| 50 | NPO | scratch | 0.509 ± 0.004 | 0.093 ± 0.001 | **0.068 ± 0.002** |
| | | K-FORGE (0.450) | 0.565 ± 0.003 | 0.047 ± 0.003 | 0.072 ± 0.003 |
| | | K-FORGE (0.600) | **0.574 ± 0.004** | **0.045 ± 0.003** | 0.071 ± 0.001 |
| | SimNPO | scratch | **0.577 ± 0.006** | 0.662 ± 0.003 | 0.270 ± 0.010 |
| | | K-FORGE (0.450) | 0.572 ± 0.007 | 0.542 ± 0.002 | 0.192 ± 0.002 |
| | | K-FORGE (0.600) | 0.571 ± 0.005 | **0.534 ± 0.002** | **0.188 ± 0.005** |
| 100 | NPO | scratch | 0.569 ± 0.009 | 0.043 ± 0.002 | **0.065 ± 0.003** |
| | | K-FORGE (0.450) | 0.587 ± 0.002 | **0.029 ± 0.002** | 0.073 ± 0.003 |
| | | K-FORGE (0.600) | **0.593 ± 0.005** | 0.031 ± 0.002 | 0.073 ± 0.003 |
| | SimNPO | scratch | **0.582 ± 0.002** | 0.518 ± 0.010 | 0.172 ± 0.013 |
| | | K-FORGE (0.450) | 0.579 ± 0.003 | 0.413 ± 0.008 | **0.143 ± 0.002** |
| | | K-FORGE (0.600) | 0.582 ± 0.002 | **0.407 ± 0.011** | 0.144 ± 0.005 |
| 250 | NPO | scratch | 0.595 ± 0.009 | 0.029 ± 0.002 | **0.064 ± 0.003** |
| | | K-FORGE (0.450) | 0.601 ± 0.003 | **0.021 ± 0.001** | 0.072 ± 0.003 |
| | | K-FORGE (0.600) | **0.603 ± 0.004** | 0.022 ± 0.001 | 0.074 ± 0.001 |
| | SimNPO | scratch | **0.596 ± 0.003** | 0.337 ± 0.005 | 0.116 ± 0.004 |
| | | K-FORGE (0.450) | 0.595 ± 0.001 | 0.276 ± 0.012 | 0.111 ± 0.003 |
| | | K-FORGE (0.600) | 0.592 ± 0.004 | **0.272 ± 0.008** | **0.110 ± 0.001** |

The one-time setup for one edited layer costs ≈ 81 s and 3.6 × 10¹⁴ FLOPs at 1B scale (≈ 7–8 downstream training steps); NPO comparisons above remain favorable with setup charged.

Known boundaries (reported in the paper): direct relearning recovers the forgotten data (no recovery resistance), MUSE gains vary by metric, and Qwen2.5-1.5B is unchanged at reported precision.

## Citation

```bibtex
@inproceedings{moskovskiy2026kforge,
  title     = {K-FORGE: Fisher-Guided Initialization for Preference-Based LLM Unlearning},
  author    = {Moskovskiy, Daniil and Panchenko, Alexander},
  booktitle = {Findings of the Association for Computational Linguistics: ICJNLP-AACL 2026},
  year      = {2026}
}
```
