# UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models (NAACL 2025)

- Authors: Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramón Huerta, Ivan Vulić
- Link​: https://arxiv.org/pdf/2402.10052

# Setup
- Hyperparameters: The original paper uses Llama-2 7B with LoRA to tune the model (rank=8, alpha=16) and learning rate of 1e-4. It's suggested to search the learning rate over [1e-5, 3e-4, 1e-4], and use an effective batch size of 32 (batch_size * gradient_accumulation). The other important hyperparemeter is beta, the strength of penalty, which typically takes a number between [3,10,30]. If we change to other models, adjusting learning rate accordingly.

- Computation Setup: All experiments are run on one A100.
- Other Details: The original paper does not use the retain set and aims to retain knowledge in all domains, not just on the retain set. So alpha is set to 0. Practionioners could search over the alpha or gamma to better retain the performance on the retain set.

# Results
Run `run.sh` script.

# Citation
@misc{dong2024undial,
      title={UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models}, 
      author={Yijiang River Dong and Hongzhou Lin and Mikhail Belkin and Ramon Huerta and Ivan Vulić},
      year={2024},
      eprint={2402.10052},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.10052}, 
}