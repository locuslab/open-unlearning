# GRU

- **Paper Title**: GRU: Mitigating the Trade-off Between Unlearning and Retention for Large Language Models
- **Authors**: Yue Wang, Qizhou Wang, Feng Liu, Wei Huang, Yali Du, Xiaojiang Du, Bo Han
- **Links**: [arXiv:2503.09117](https://arxiv.org/abs/2503.09117)


This work proposes **Gradient Rectified Unlearning (GRU)**, a general framework for improving unlearning performance without sacrificing retention in large language models. GRU modifies the gradient update rule to remove the component of the unlearning gradient that conflicts with the retention gradient.

# Setup

- **Hyperparameters & Search Space**:
  - Gradient EMA smoothing factor \(\gamma \in \{0.8, 0.9, 0.95, \text{N/A}\}\)


- **GPU Type**: NVIDIA A100 80GB  
- **GPU Usage**: Current code supports **single GPU execution only**. Multi-GPU support is not yet implemented.

- **DeepSpeed Configuration**:  
  GRU currently **does not support DeepSpeed** due to its reliance on fine-grained gradient manipulation. Please ensure DeepSpeed is disabled for all GRU experiments.

# Results


# Citation


If you use this work, please cite:

```bibtex

@misc{wang2025grumitigatingtradeoffunlearning,
      title={GRU: Mitigating the Trade-off between Unlearning and Retention for Large Language Models},
      author={Yue Wang and Qizhou Wang and Feng Liu and Wei Huang and Yali Du and Xiaojiang Du and Bo Han},
      year={2025},
      eprint={2503.09117},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.09117},
}
```