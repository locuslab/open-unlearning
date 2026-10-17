# Temper-Then-Tilt: Principled Unlearning for Generative Models through Tempering and Classifier Guidance (ICML 2026)

- Authors: Jacob L. Block, Mehryar Mohri, Aryan Mokhtari, Sanjay Shakkottai
- ArXiv: https://arxiv.org/abs/2602.10217
- Project Page: https://github.com/jacob-block/T3-Unlearning

## Setup

The script `run.sh` runs a small sweep for hyperparameter search locally on a single GPU. For optimal performance, hyperparameters should be tuned for each specific user setup and environment. Since our method only trains a small classifier on top of the base LLM, it can be trained with minimal resources and overhead. Please see the paper and project page for more details. 

## Citation

```bibtex
@inproceedings{block:2026:T3-Unlearning,
  title={Temper-Then-Tilt: Principled Unlearning for Generative Models through Tempering and Classifier Guidance},
  author={Jacob L. Block and Mehryar Mohri and Aryan Mokhtari and Sanjay Shakkottai},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```