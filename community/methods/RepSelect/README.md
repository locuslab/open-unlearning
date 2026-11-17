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
  - `trainer.method_args.strength` (default 0.05): the size of the single update step, and the main parameter to tune. It trades forgetting against utility.
  - `trainer.method_args.n_pcs` (default 512): the number of retain singular vectors to collapse along, per side.
  - `trainer.method_args.lora_learning_rate` (default `null`): SGD learning rate of the LoRA elicitation, e.g. 0.01; `null` skips it. Needs `peft`.
  - `trainer.args.per_device_train_batch_size` only affects speed and memory. `learning_rate`, the scheduler, optimizer, weight decay and number of epochs have no effect.
- **Computational Setup:** One GPU. The method needs one pass (forward and backward) over the retain batches and one over the forget batches, plus one low-rank SVD per MLP weight matrix. The elicitation adds one more pass over the forget batches.
- **Other Details:** The model has to expose its blocks as `model.model.layers[i].mlp` with `gate_proj`, `up_proj`, `down_proj` (Llama, Qwen, Gemma, Mistral/Zephyr), or with transformers' fused MoE `experts` module.

# Results

Run `community/methods/RepSelect/run.sh` to unlearn TOFU `forget10` from Llama-3.2-1B-Instruct and evaluate it.

TOFU `forget10`, Llama-3.2-1B-Instruct, `n_pcs=512`, no LoRA elicitation, one run per strength:

| strength           | forget_quality | forget_Q_A_Prob | forget_Q_A_ROUGE | extraction_strength | model_utility |
| ------------------ | -------------- | --------------- | ---------------- | ------------------- | ------------- |
| 0 (starting model) | 8.1e-22        | 0.880           | 0.821            | 0.704               | 0.598         |
| 0.05               | 3.2e-15        | 0.462           | 0.477            | 0.191               | 0.595         |
| 0.1                | 4.3e-10        | 0.282           | 0.395            | 0.114               | 0.591         |
| 0.2                | 7.7e-04        | 0.162           | 0.357            | 0.081               | 0.570         |

The results of each run are written to `saves/unlearn/<task_name>/evals/TOFU_SUMMARY.json`.

Since relearning attacks aren't yet implemented in open-unlearning, the results above are only pre-attack.

Note: RepSelect's purpose is to be robust to attacks.
It applies a single edit instead of many optimizer steps, which is good for attack robustness, but doesn't push the pre-attack forget scores as low.
For low forget scores both pre- and post-attack, you may want to also run additional unlearning on top of RepSelect.

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
