# The More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning

- Authors: Anna Borisiuk, Andrey Savchenko, Alexander Panchenko, Elena Tutubalina
- Link: https://arxiv.org/abs/2608.14229

Popular facts are harder to unlearn than rare ones. AdaPop scales the forgetting pressure on each fact by its popularity, and adjusts the retain weight during training so that the retain loss does not drift.

# Method

AdaPop builds on WGA (weighted gradient ascent, see `community/methods/WGA`).

- **Popularity-aware forget loss.** Each forget token loss $\ell_{i,t}$ is weighted by $\mathrm{stopgrad}(p_{i,t})^{\beta_i}$, where $p_{i,t}$ is the model's probability of the token and
  $\beta_i = \mathrm{clip}(a \cdot s_i^{-b},\ 0.05,\ 2.0)$, with $s_i$ the popularity of fact $i$.
  Popular facts get a small $\beta_i$, so the weight stays large and the pressure is kept up until the fact is forgotten. Rare facts get a large $\beta_i$, so the weight fades quickly once the fact is forgotten, which avoids over-erasing them.
- **Dual-ascent retain weight.** The loss is $\gamma L_f + \alpha_k L_r$ with $\alpha_k = \mathrm{clip}(\alpha_0 + \lambda_k, 0, \gamma)$. At the end of every epoch, $\lambda$ is updated from the relative increase $\delta$ of the retain loss (EMA) over the first epoch:
  $\lambda \leftarrow \mathrm{clip}(\lambda + \eta_\lambda(\delta - \varepsilon),\ 0,\ \lambda_{\max})$.
  This removes the need to grid-search the retain weight $\alpha$.

Implementation: [`src/trainer/unlearn/adapop.py`](../../../src/trainer/unlearn/adapop.py), config: [`configs/trainer/AdaPop.yaml`](../../../configs/trainer/AdaPop.yaml).

# Popularity

AdaPop reads the popularity of every forget example from a numeric (int / float) column of the forget dataset.

- `trainer.method_args.use_popularity=false` (default): no column is needed. Every fact gets the same popularity `default_popularity` (100 by default), i.e. $\beta \approx 1.5$ for all facts, and AdaPop reduces to WGA with the adaptive retain weight.
- `trainer.method_args.use_popularity=true`: set the column with `+data.forget.<DATASET>.args.popularity_key=<column>` (`<DATASET>` is the forget dataset's config name, e.g. `DUET_QA_forget`). Training stops with an error if the column is missing or is not numeric. This is the setting of the paper.

The defaults $a = 58.7$, $b = 0.796$ come from two anchor points on the DUET `pop_sum` scale: a rare fact with popularity $s_r = 100$ gets $\beta_r = 1.5$, a popular fact with $s_p = 3000$ gets $\beta_p = 0.1$ ($\beta = 1$ at a popularity of about 167). For a popularity column on another scale, pick a rare and a popular anchor on that scale and set

$$b = \frac{\ln(\beta_r / \beta_p)}{\ln(s_p / s_r)}, \qquad a = \beta_r \, s_r^{\,b}$$

via `trainer.method_args.beta_a` and `trainer.method_args.beta_b`.

## Annotating popularity with Wikidata

[`wikidata_popularity.py`](wikidata_popularity.py) is the script used to annotate the facts of [DUET](https://huggingface.co/datasets/SwetieePawsss/DUET). It takes `(subjectLabel, relation, objectLabel)` triples, links them to Wikidata through the public SPARQL endpoint (no API key), and writes `pop_subject` / `pop_object` (sitelinks + identifiers + statements of the item) and `pop_sum = pop_subject + pop_object`, which is the popularity used by AdaPop. See the script header for usage. Any other popularity proxy (page views, corpus frequency, an LLM judge) works too, as long as it separates rare from popular facts.

# Setup

- **Hyperparameters:** in the paper (DUET and RWKU; Llama, Qwen and Gemma models), AdaPop was run with LoRA adapters (rank 32, alpha 64) at learning rate 1e-4 for 5 epochs with an effective batch size of 32, with the controller defaults of `configs/trainer/AdaPop.yaml`. The commands in `run.sh` use full-parameter unlearning, so the learning rate has to be re-tuned (1e-5 is a starting point); they are not expected to reproduce the paper's numbers exactly.
- **Computational setup:** one A100 GPU per run.

# Running

TOFU, without popularity (all facts get popularity 100):

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  trainer=AdaPop trainer.method_args.use_popularity=false \
  forget_split=forget10 retain_split=retain90 task_name=tofu_AdaPop
```

DUET with Wikidata popularity (needs the DUET benchmark configs, `experiment=unlearn/duet/default`):

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/duet/default \
  trainer=AdaPop trainer.method_args.use_popularity=true \
  +data.forget.DUET_QA_forget.args.popularity_key=pop_sum \
  forget_split=city_forget_popular_5 task_name=duet_AdaPop_popular
```

`run.sh` contains both settings together with the evaluation commands.

# Citation

```bibtex
@misc{borisiuk2026popularharderforgetadaptive,
      title={The More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning}, 
      author={Anna Borisiuk and Andrey Savchenko and Alexander Panchenko and Elena Tutubalina},
      year={2026},
      eprint={2608.14229},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.14229}, 
}
```
