# PALU

Implementation of [PALU](https://arxiv.org/abs/2601.03190) (Naixin Zhai, University of Science and Technology of China, 2601.03190, [nxZhai/PALU](https://github.com/nxZhai/PALU), Stars: 16 as of 2026-11-05): *Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning*.

## Method and configuration

Select `trainer=PALU` or use `experiment=unlearn/tofu/palu`, which also loads the annotated forget data. At each selected sensitive position, PALU selects the reference model's top-k vocabulary candidates and moves their current logits toward the current model's vocabulary-wide mean. 

`first_n` selects a prefix of each contiguous sensitive span, independently for each example; EOS is excluded from sensitive positions.

| Parameter under `trainer.method_args` | Default | Meaning |
| --- | --- | --- |
| `top_k` | `1` | Reference vocabulary candidates to flatten; capped at vocabulary size. |
| `first_n` | `-1` | Prefix length per sensitive span; non-positive values select every sensitive token. |
| `target_mode` | `mean` | Current vocabulary mean, matching the source objective. |
| `non_sensitive_constraint_type` | `kl` | Preserve non-sensitive answer tokens with `kl`, or with `nll` on reference argmax tokens. |
| `kl_direction` | `reverse` | `forward`, `reverse`, or `mixed`, defined below. |
| `gamma`, `alpha` | `1.0`, `1.0` | Sensitive loss weight and shared preservation weight. |
| `retain_loss_type` | `NLL` | Existing Open-Unlearning retain objective; `KL` also remains available. |

## Annotated data

The three directories under [data/](data/) are copied byte-for-byte from the [official code's processed TOFU data](https://github.com/nxZhai/PALU/tree/main/data/palu). Together they occupy 909,489 bytes (about 0.87 MiB).

| Directory | Examples |
| --- | ---: |
| `forget01_with_common_words_gpt` | 40 |
| `forget05_with_common_words_gpt` | 200 |
| `forget10_with_common_words_gpt` | 400 |

Each directory contains the original Arrow file, `dataset_info.json`, and `state.json`, readable with `datasets.load_from_disk`. The columns are `question`, `answer`, `target_words`, and `common_words`. Sensitive spans in `target_words` use half-open character offsets `[start, end)` into the answer. `common_words` is preserved; as in the source implementation, all answer tokens outside sensitive spans receive the non-sensitive label, so this column is not needed to build masks. Examples with no annotated sensitive span remain usable for preservation.

`PALUDataset` requires a fast tokenizer. It uses the model's existing prompt template and maps character spans using native tokenizer offsets. Labels are token IDs for sensitive tokens, `-200` for other answer tokens, and `-100` for prompt/padding positions. The custom forget labels are never passed to the language model's cross-entropy loss. Retain data and evaluation data use the existing TOFU loaders and standard labels.

These are annotations of the [TOFU dataset](https://huggingface.co/datasets/locuslab/TOFU), whose dataset card specifies the MIT license. The original implementation's MIT notice is preserved in [LICENSE](LICENSE).

## Run an example

Run from the repository root:

```bash
bash community/methods/PALU/run.sh
```
