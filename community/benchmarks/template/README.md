# TITLE

- Paper title, authors, links.

Provide a concise summary of your benchmark details and its contributions. Please avoid using images to keep the repository size manageable.

# Datasets

Use a clear and consistent naming convention for dataset splits.

- [ ] Provide a link to find/download the datasets (preferably HuggingFace).

# Models


- [ ] Upload any unlearning target or reference retain models for unlearning preferably on HuggingFace and provide the path.
- [ ] Model creation details and how they fit in benchmark.

# Baselines & Results

Discuss the baselines used and their results.


## Setup
Please include the experimental setup for the baselines

- [ ] **Hyperparameters & Search Space:** Specify key hyperparameters, their search ranges, number of trials etc.
- [ ] **Computational Setup:** Mention the type and number of GPUs used.
- [ ] **DeepSpeed Configuration** (if used): If any modifications were made to the default DeepSpeed config, specify them here. (You may include the config as a code block.)
- [ ] **Other Details:** Any additional setup details crucial for reproducing your method.

To replicate your results, provide a `run.sh` script that contains all necessary commands to reproduce the final results. Ensure the script is well-documented.


# Citation


If you use this work, please cite:

```bibtex

<YOUR CITATION bibtex>

@misc{openunlearning2025,
  title={OpenUnlearning: A Unified Framework for LLM Unlearning Benchmarks},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Kolter, J Zico and Maini, Pratyush},
  year={2025},
  howpublished={\url{https://github.com/locuslab/open-unlearning}},
  note={Accessed: February 27, 2025}
}
```