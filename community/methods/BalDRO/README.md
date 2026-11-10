# BalDRO: A Distributionally Robust Optimization Based Framework for Large Language Model Unlearning

We propose [BalDRO](https://arxiv.org/abs/2601.09172), a distributionally robust optimization framework that adaptively emphasizes hard-to-unlearn samples. BalDRO-G selects hard groups, while BalDRO-DV smoothly reweights individual losses through the Donsker-Varadhan dual. Both variants support NPO, SimNPO, and SatImp.

## Setup

Use the standard Open-Unlearning environment and prepare the TOFU or MUSE benchmark; no additional dependencies are required. The training objective is

$$
\mathcal{L}(\theta)=\gamma\mathcal{L}_{\mathrm{forget}}(\theta)+\alpha\mathcal{L}_{\mathrm{retain}}(\theta),
$$

where `gamma` and `alpha` control the forget-retain trade-off.

**BalDRO-G** selects the $k$ forget samples with the lowest length-normalized negative log-likelihood $\bar{\ell}_{\mathrm{NLL}}$, then applies the base unlearning objective to this group:

$$
G=\operatorname{TopK}_{i\in B_f}\!\left(-\bar{\ell}_{\mathrm{NLL}}(x_i;\theta),k\right),\qquad
\mathcal{L}_{\mathrm{forget}}^{\mathrm{G}}=\frac{1}{k}\sum_{i\in G}\ell_i(\theta).
$$

Here $\ell_i$ is the base method's per-sample forget loss and $k=\max(1,\lfloor r|B_f|\rfloor)$. The selection ratio $r$ is `sampling_ratio` (default `0.5`, range $(0,1]$); a random retain group of up to $k$ samples supplies the retain loss.

**BalDRO-DV** replaces the mean forget loss with

$$
\mathcal{L}_{\mathrm{forget}}^{\mathrm{DV}}=\tau_f\log\left(\frac{1}{|B_f|}\sum_{i\in B_f}\exp\left(\frac{\ell_i(\theta)}{\tau_f}\right)\right).
$$

The positive temperature $\tau_f$ is `beta_dv_forget` (default `1.0`); smaller values emphasize larger losses more strongly. Defaults are `forget_dro=true` and `retain_dro=false`. Optional retain-side DV uses `beta_dv_retain` (default `1.0`) and supports NLL retention only. Base-method parameters remain `beta` for NPO, `beta` and `delta` for SimNPO, and `beta1` and `beta2` for SatImp.

## Results

The following scripts reproduce the TOFU and MUSE experiment grids. Run from the repository root after preparing the benchmark's retained-model evaluation logs. Scripts use GPU `0` and W&B; authenticate with W&B or set `REPORTTO=none` in the selected script.

### BalDRO-G

| Base method | Trainer | TOFU | MUSE |
| --- | --- | --- | --- |
| NPO | `BalDRO/GroupNPO` | [Script](scripts/tofu/train_tofu_groupnpo.sh) | [Script](scripts/muse/train_muse_npo_group.sh) |
| SimNPO | `BalDRO/GroupSimNPO` | [Script](scripts/tofu/train_tofu_groupsimnpo.sh) | [Script](scripts/muse/train_muse_simnpo_group.sh) |
| SatImp | `BalDRO/GroupSatImp` | [Script](scripts/tofu/train_tofu_groupsatimp.sh) | [Script](scripts/muse/train_muse_satimp_group.sh) |

### BalDRO-DV

| Base method | Trainer | TOFU | MUSE |
| --- | --- | --- | --- |
| NPO | `BalDRO/DrNPO` | [Script](scripts/tofu/train_tofu_drnpo.sh) | [Script](scripts/muse/train_muse_drnpo.sh) |
| SimNPO | `BalDRO/DrSimNPO` | [Script](scripts/tofu/train_tofu_drsimnpo.sh) | [Script](scripts/muse/train_muse_drsimnpo.sh) |
| SatImp | `BalDRO/DrSatImp` | [Script](scripts/tofu/train_tofu_drsatimp.sh) | [Script](scripts/muse/train_muse_drsatimp.sh) |

For example:

```bash
bash community/methods/BalDRO/scripts/tofu/train_tofu_drnpo.sh
bash community/methods/BalDRO/scripts/muse/train_muse_simnpo_group.sh
```

Each script runs its full parameter grid. Compared with the source scripts, trainer names use the `BalDRO/` prefix, `run_name` uses Hydra's `+` syntax, and `do_save=True` is omitted because Open-Unlearning saves the final model automatically.

## Citation

```bibtex
@inproceedings{shao2026baldro,
  title={Baldro: A distributionally robust optimization based framework for large language model unlearning},
  author={Shao, Pengyang and Zhai, Naixin and Chen, Lei and Yang, Yonghui and Zhu, Fengbin and Yang, Xun and Wang, Meng},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={8874--8884},
  year={2026}
}
```
