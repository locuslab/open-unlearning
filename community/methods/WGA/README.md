# Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond (ICLR 2025)

- Authors: Qizhou Wang, Jin Peng Zhou, Zhanke Zhou, Saebyeol Shin, Bo Han, Kilian Q Weinberger
- Link​: https://arxiv.org/pdf/2502.19301

# Setup
- Hyperparameters: The original paper uses LLaMA-2-7B and Phi-1.5 to fine-tune the model, with learning rates set to 1e-5 and 2e-5, respectively.  The effective batch size is 16 (batch_size * gradient_accumulation). The other important hyperparemeters is beta, which is set to 1.0.

- Computation Setup: All experiments are run on one A100-80G.

# Results
Run `run.sh` script.

# Citation
```bibtex
@article{wang2025rethinking,
  title={Rethinking llm unlearning objectives: A gradient perspective and go beyond},
  author={Wang, Qizhou and Zhou, Jin Peng and Zhou, Zhanke and Shin, Saebyeol and Han, Bo and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:2502.19301},
  year={2025}
}
```