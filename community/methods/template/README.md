# TITLE

- Paper title, authors, links.


Provide a concise summary of your method details and its contributions. Please avoid using images to keep the repository size manageable.

# Setup

Please include the experimental setup such as

- [ ] **Hyperparameters & Search Space:** Specify key hyperparameters, their search ranges, number of trials etc.
- [ ] **Computational Setup:** Mention the type and number of GPUs used.
- [ ] **DeepSpeed Configuration** (if used): If any modifications were made to the default DeepSpeed config, specify them here. (You may include the config as a code block.)
- [ ] **Other Details:** Any additional setup details crucial for reproducing your method.

# Results

To replicate your results, provide a `run.sh` script that contains all necessary commands to reproduce the final results. Ensure the script is well-documented.

It would be appreciated if you can upload the final unlearned model(s) along with their `evals` folders to HuggingFace and provide the link(s) here. As the evaluations are updated, this would help us re-evaluate your model(s).

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