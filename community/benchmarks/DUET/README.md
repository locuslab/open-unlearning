# DUET: Dual Unlearning Evaluation across Training stages

- Paper: [Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning](https://aclanthology.org/2026.findings-acl.1287/) (Findings of ACL 2026, [arXiv:2602.19612](https://arxiv.org/abs/2602.19612))
- Authors: Anna Borisiuk, Andrey Savchenko, Alexander Panchenko, Elena Tutubalina
- Dataset: [🤗 SwetieePawsss/DUET](https://huggingface.co/datasets/SwetieePawsss/DUET)
- Target models: [🤗 SwetieePawsss/DUET_ft_models](https://huggingface.co/SwetieePawsss/DUET_ft_models)

DUET studies how the popularity (salience) of a fact affects unlearning. It contains question-answer pairs built from Wikidata triples. Each fact carries a popularity score, and the forget sets are split into **rare** and **popular** facts of the same size, so the same method can be compared on facts the model has seen rarely or often.

# Datasets

All splits are in [`SwetieePawsss/DUET`](https://huggingface.co/datasets/SwetieePawsss/DUET) (`question`, `answer`, plus Wikidata ids and popularity columns: `pop_subject`, `pop_object`, `pop_sum`).

City domain (used by the default configs):

| Split | Examples | Description |
|---|---|---|
| `city_forget_rare_5` / `city_forget_popular_5` | 482 | forget set, rare / popular facts |
| `city_forget_rare_1` / `city_forget_popular_1` | 96 | smaller forget sets |
| `city_forget_rare_10` / `city_forget_popular_10` | 965 | larger forget sets |
| `city_fast_retain_500` | 500 | retain set used for training and evaluation |
| `city_retain_intersection_{98,90,80}` | 9,466 / 8,694 / 7,728 | full retain sets for the 1 / 5 / 10 % forget sets |

The general domain has the same structure without the `city_` prefix (`forget_rare_5`, `forget_popular_5`, `retain_intersection_90`, ...). The forget and retain sets of a pair do not overlap.

# Models

The target (pre-unlearning) models are instruction models finetuned on DUET for 3 epochs. They are stored as subfolders of [`SwetieePawsss/DUET_ft_models`](https://huggingface.co/SwetieePawsss/DUET_ft_models):

| Subfolder | Model config |
|---|---|
| `llama-3.1-8b-instruct-tripunlamb-ft` | `Llama-3.1-8B-Instruct` |
| `qwen2.5-7b-instruct-tripunlamb-ft` | `Qwen2.5-7B-Instruct` |
| `gemma-7b-it-tripunlamb-ft` | `gemma-7b-it` |

Select one with `model=<model config> model_subfolder=<subfolder>`. The models were finetuned on the plain `question answer` text of the `city_full` (Llama) or `filtered` (Qwen, Gemma) split. Unlearning and evaluation use the chat template with the system prompt of the DUET paper (`You are terse and fact-only QA assistant. Reply to the question clearly with few-words correct answer.`), set in the experiment configs. Gemma's chat template has no system role, so `run.sh` sets `model.template_args.system_prompt=null` for it.

# Evaluation

`configs/eval/duet.yaml` reports ROUGE-L recall of greedy generations on the forget split (`forget_Q_A_ROUGE`, lower is better) and on the retain split (`retain_Q_A_ROUGE`, higher is better). As in the paper, the retain score is measured on the same retain split that is used during unlearning. Comparing the forget score on the `rare` and `popular` forget sets shows how much harder popular facts are to unlearn.

# Running

Unlearn the Llama target model on the rare forget set with `GradDiff`, then evaluate:

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/duet/default \
  trainer=GradDiff forget_split=city_forget_rare_5 retain_split=city_fast_retain_500 \
  task_name=duet_llama_GradDiff_rare

python src/eval.py --config-name=eval.yaml experiment=eval/duet/default \
  forget_split=city_forget_rare_5 retain_split=city_fast_retain_500 \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/duet_llama_GradDiff_rare \
  model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/duet_llama_GradDiff_rare \
  model_subfolder="" task_name=duet_llama_GradDiff_rare
```

Evaluate the target model before unlearning:

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/duet/default \
  forget_split=city_forget_popular_5 task_name=duet_llama_target_popular
```

`run.sh` runs a set of baselines on both forget sets for all three target models.

## Setup

- **Hyperparameters:** the default config uses learning rate 1e-5, 5 epochs, effective batch size 32 and the `paged_adamw_8bit` optimizer. These are not tuned; tune the learning rate and the method's own parameters for each model.
- **Compute:** the target models have 7-8B parameters. The default config uses `paged_adamw_8bit` to reduce optimizer memory. For methods that keep a reference model (NPO, SimNPO, KL retain loss) or for smaller GPUs, use DeepSpeed or several GPUs (see [`docs/experiments.md`](../../../docs/experiments.md)).

# Citation

```bibtex
@inproceedings{borisiuk-etal-2026-anatomy,
    title = "Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning",
    author = "Borisiuk, Anna  and
      Savchenko, Andrey  and
      Panchenko, Alexander  and
      Tutubalina, Elena",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1287/",
    doi = "10.18653/v1/2026.findings-acl.1287",
    pages = "25846--25859",
    ISBN = "979-8-89176-395-1",
    }
```
