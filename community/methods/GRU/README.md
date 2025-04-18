# TITLE

- **Paper Title**: GRU: Mitigating the Trade-off Between Unlearning and Retention for Large Language Models
- **Authors**: Yue Wang, Qizhou Wang, Feng Liu, Wei Huang, Yali Du, Xiaojiang Du, Bo Han
- **Links**: [arXiv:2503.09117](https://arxiv.org/abs/2503.09117)


Provide a concise summary of your method details and its contributions. Please avoid using images to keep the repository size manageable.

# Setup

Please include the experimental setup such as

- [ ] **Hyperparameters & Search Space:** Specify key hyperparameters, their search ranges, number of trials etc.
- [ ] **Computational Setup:** Mention the type and number of GPUs used.
- [ ] **DeepSpeed Configuration:** If any modifications were made to the default DeepSpeed config, specify them here. (You may include the config as a code block.)
- [ ] **Other Details:** Any additional setup details crucial for reproducing your method.


## Computational Setup


- **GPU Details**: NVIDIA A100 80GB
- **GPU Count**: The code for our method currently supports single GPU execution. We plan to enhance the codebase in the future to support multi-GPU configurations.


# Results

To replicate your results, provide a `run.sh` script that contains all necessary commands to reproduce the final results. Ensure the script is well-documented.

It would be appreciated if you can upload the final unlearned model(s) along with their `evals` folders to HuggingFace and provide the link(s) here. As the evaluations are updated, this would help us re-evaluate your model(s).

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