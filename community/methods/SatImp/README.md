# Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning (ICML 2025)

- Authors: Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, and Bo Han
- Link​: https://arxiv.org/pdf/2505.11953

# Setup
- Hyperparameters: The original paper uses LLaMA-2-7B and Phi-1.5 to fine-tune the model, with learning rates set to 1e-5 and 2e-5, respectively.  The effective batch size is 16 (batch_size * gradient_accumulation). The other important hyperparemeters is beta1 and beta2, which are set to 5.0 and 1.0, respectively. The rate between forget loss and retain loss is set to 10.
  
- Notice that there is a consistency gap between ES metric and FQ\MU, the original paper considers ES as the primary metric, thus we set the hyperparemeters gamma=1.0 amd alpha=0.1. Practionioners could search over the alpha or gamma to explore better FQ/MU. Results on WMDP and MUSE are also influenced by the ratio between forget loss and retain loss.

- Computation Setup: All experiments are run on one A100-80G.

# Results
Run `run.sh` script.

# Citation
```bibtex
@article{yang2025exploring,
  title={Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning},
  author={Yang, Puning and Wang, Qizhou and Huang, Zhuo and Liu, Tongliang and Zhang, Chengqi and Han, Bo},
  journal={arXiv preprint arXiv:2505.11953},
  year={2025}
}
```