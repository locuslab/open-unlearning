# RepSelect: Robust LLM Unlearning via Representation Selectivity

- Authors: Filip Sondej, Yushi Yang, Adam Mahdi
- Paper: https://arxiv.org/abs/2606.17168
- Code: https://github.com/filyp/RepSelect

Unlearning methods that follow the raw forget gradient also edit representations that the forget set shares with the retain set.
Those edits disrupt general capabilities, and they are the ones that fine-tuning attacks restore most easily.
RepSelect keeps only the part of the forget gradient that is specific to the forget set, and applies it as a single update of the MLP weights.

For each MLP weight matrix (`gate_proj`, `up_proj`, `down_proj` of every layer):

1. One pass over the retain set accumulates the weight gradient, and its top `n_pcs` singular triplets `U, S, V` are computed.
2. One pass over the forget set accumulates the weight gradient.
3. The forget gradient is collapsed along the retain singular vectors, on the input side (`V`) and on the output side (`U`): its component along each vector is divided by `S^2`. The rest of the gradient is left as is.
4. The collapsed gradient is subtracted from the weight once, scaled by `strength`.

There is no optimizer and no training loop: the whole method is one pass over the data and one SVD per weight matrix.

Optionally (`lora_learning_rate`), a LoRA is first trained for one pass to lower the forget loss, and is kept active while the two gradients are computed.
It is removed before the update.
This elicitation is off by default.

Note: the main results in the paper use an older RepSelect variant, while this implementation in open-unlearning is the recommended better variant, based on follow-up results (see Appendix A in the paper). Changes: 1. we now compute SVD on the retain, not the forget distribution; 2. during the collapse, we now divide by the *squared* singular values; 3. for simplicity LoRA elicitation is off by default.

# Setup

- **Hyperparameters:**
  - `trainer.method_args.strength` (default 2.5): the size of the single update step, and the main parameter to tune. It trades forgetting against utility.
  - `trainer.method_args.n_pcs` (default 512): the number of retain singular vectors to collapse along, per side.
  - `trainer.method_args.lora_learning_rate` (default `null`): SGD learning rate of the LoRA elicitation, e.g. 0.01; `null` skips it. Needs `peft`.
- **Computational Setup:** One GPU. The method needs one pass (forward and backward) over the retain batches and one over the forget batches, plus one low-rank SVD per MLP weight matrix. The elicitation adds one more pass over the forget batches.
- **Other Minor Details:** The model has to expose its blocks as `model.model.layers[i].mlp` with `gate_proj`, `up_proj`, `down_proj` (Llama, Qwen, Gemma, Mistral/Zephyr), or with transformers' fused MoE `experts` module. The retain pass covers the whole retain set, unlike other methods here, which sample one retain example per forget example. For huge retain sets, you may cap it in the data config, e.g. `data.retain.TOFU_QA_retain.args.hf_args.split='"train[:10000]"'`.

# Results

Run `community/methods/RepSelect/run_with_relearning_attack.sh` to reproduce the table below, or `community/methods/RepSelect/run.sh` for plain unlearning and the standard TOFU evaluation.

RepSelect's purpose is robustness to attacks, so we evaluate it under a relearning attack, with held-out evaluation as in [Deeb & Roger (2024)](https://arxiv.org/abs/2410.08827).
Each model is unlearned on all of TOFU `forget10`, then the attacker fine-tunes it for one epoch on the first half of `forget10` (10 authors), and we evaluate on the second half (the other 10 authors).
The attacker never sees the evaluated questions, so what comes back on them was still in the weights.

Llama-3.2-1B-Instruct, values are pre-attack → post-attack on the held-out half, `model_utility` is pre-attack, one run per row:

| model                             | forget_Q_A_Prob | forget_Q_A_ROUGE | extraction_strength | model_utility |
| --------------------------------- | --------------- | ---------------- | ------------------- | ------------- |
| Target model (no unlearning)      | 0.885 → 0.875   | 0.827 → 0.773    | 0.732 → 0.682       | 0.598         |
| Retain model (never saw forget10) | 0.123 → 0.210   | 0.379 → 0.369    | 0.062 → 0.071       | 0.591         |
| NPO                               | 0.102 → 0.505   | 0.302 → 0.445    | 0.103 → 0.178       | 0.577         |
| RepSelect, strength 0.05          | 0.477 → 0.554   | 0.488 → 0.489    | 0.205 → 0.226       | 0.595         |
| RepSelect, strength 0.1           | 0.295 → 0.412   | 0.418 → 0.430    | 0.121 → 0.148       | 0.591         |
| RepSelect, strength 0.2           | 0.172 → 0.290   | 0.375 → 0.403    | 0.087 → 0.102       | 0.570         |

- RepSelect uses `n_pcs=512` and no LoRA elicitation.
- NPO uses the repository's default hyperparameters for 250 steps.
- The retain model was never trained on `forget10`: its row shows how much the attack raises the scores on its own.
- The attack's learning rate (1e-5, batch size 8) is the value from {1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4} that recovers the most forget probability on RepSelect (strength 0.1). It was not tuned against NPO.

Note: in TOFU, the knowledge to forget was added to the model by a short fine-tuning.
[Deeb & Roger (2024)](https://arxiv.org/abs/2410.08827) found that such recently added knowledge is easier to unlearn robustly than knowledge learned during pretraining.
So the numbers above, for every method, likely overestimate robustness on real pretrained knowledge.

The results of each run are written to `saves/unlearn/<task_name>/evals_heldout/TOFU_SUMMARY.json` (pre-attack) and `saves/finetune/<task_name>_relearned/evals_heldout/TOFU_SUMMARY.json` (post-attack).


# Citation

```bibtex
@inproceedings{sondej2027repselect,
  title={RepSelect: Robust LLM Unlearning via Representation Selectivity},
  author={Sondej, Filip and Yang, Yushi and Mahdi, Adam},
  booktitle={Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2027},
  note={To appear. arXiv:2606.17168}
}
```
