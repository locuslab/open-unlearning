# Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models
- Authors: Anmol Mekala, Vineeth Dorna, Shreya Dubey, Abhishek Lalwani, David Koleczek, Mukund Rungta, Sadid Hasan, Elita Lobo
- Paper Link: https://arxiv.org/pdf/2409.13474
- Code Link: https://github.com/molereddy/Alternate-Preference-Optimization


LLMs struggle to suppress forget set responses using only negative feedback during unlearning, often resulting in inconsistent outputs, reduced utility, and potential privacy risks. To address this, AltPO enables stable and effective unlearning by combining negative feedback on the forget set along with positive feedback through plausible alternative responses.


## Setup

#### Generate Alternate Dataset

The following command generates alternate responses for TOFU, which are then used for unlearning.
```python
python generate.py dataset_config.dataset_kwargs.name=forget10
```

#### Hyperparameters & Search Space
The original paper experiments with LLaMA2-7B; however, the following parameter ranges are reasonable to explore. You can adjust them based on the model and task. Perform a grid search over: beta in [0.05, 0.1, 0.5], learning rate in [1e-5, 2e-5, 5e-5], and alpha in [1, 2, 5].

#### Computational Setup
All experiments in `run.sh` are run on single A100 GPU. If larger models are used you can use deepspeed to launch the unlearning job.


## Results
Run `run.sh` script.


## Citation
```bibtex
@article{mekala2024alternate,
  title={Alternate preference optimization for unlearning factual knowledge in large language models},
  author={Mekala, Anmol and Dorna, Vineeth and Dubey, Shreya and Lalwani, Abhishek and Koleczek, David and Rungta, Mukund and Hasan, Sadid and Lobo, Elita},
  journal={arXiv preprint arXiv:2409.13474},
  year={2024}
}
```