# Contributing

Everyone is welcome to contribute, and every contribution is valued. Aside from coding components, answering questions, assisting others, and improving documentation are all appreciated.

You can also help by spreading the word! If you find this project useful, please share it with others, cite it, link it on your repositories and posts, or simply ⭐️ the repo to show your support.

> 🤝 This guide is heavily borrowed from awesome [transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) guide to contributing.

## Ways to Contribute

There are several ways you can contribute to OpenUnlearning:

* Fix issues with the existing code.
* Submit issues related to bugs or desired new features.
* Support new components (models, datasets, collator etc).
* Implement new unlearning methods.
* Implement new evaluations.
* Contribute to the documentation.

Once your feature is added you may also link the relevant paper in [`docs/links.md`](../docs/links.md)

## Fixing Issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#create-a-pull-request) and open a Pull Request!

## Submitting a Bug-Related Issue or Feature Request

Do your best to follow these guidelines when submitting a bug-related issue or a feature request. It will make it easier for us to come back to you quickly and with good feedback.

### Did You Find a Bug?

Before you report an issue, we would really appreciate it if you could **make sure the bug was not already reported** (use the search bar on GitHub under Issues). Please try to ensure that the bug is in OpenUnlearning itself, and not your code.

Please include the following information in your issue so we can quickly resolve it:

* A short, self-contained, code snippet that allows us to reproduce the bug.
* The **full** traceback if an exception is raised.
* The hardware used to run the experiment, including specifications such as the number and type of GPUs etc.
* The hydra config file corresponding to the experiment if needed (since these files ae long you may link them or use a markdown dropdown in your issue).
* Attach any other additional information, like screenshots, you think may help.

### Do You Want a New Feature?

If there is a new feature you'd like to see in OpenUnlearning, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

## Do You Want to Support New Components?

Adding a new component listed below requires defining a new class, registering it, and creating a configuration file. Learn more about adding new components in [`docs/components.md`](docs/components.md).

1. [Trainer](components#trainer) - Algorithm used in LLM training or unlearning  
2. [Dataset](components#dataset) - Dataset class for preprocessing raw data  
3. [Evaluation Metric](components#evaluation-metric) - Metric class implementing model evaluation  
4. [Benchmark](components#benchmark) - Suite combining multiple evaluation metrics  
5. [Model](components#model) - LLM used in unlearning  
6. [Collator](components#collator) - Handles data collation logic  
7. [Experiment](components#experiment) - Combines components into a final experiment config  

> [!IMPORTANT]  
> **We especially encourage** contributions of methods and benchmarks that you've created, since you best understand them and know how to use them. We are ready to expedite their integration into OpenUnlearning.  
> When facing difficulties implementing any component, please contact the maintainers to join our discord where we can go in detail with the implementations.

## Contributing a New Unlearning Method

### 1. Implement an Unlearning Trainer

Your method might require a custom loss function, or other trainer related modifications which go here.  
Refer to our [Trainer implementation guide](components.md#trainer) to ensure your method integrates well with our framework.

### 2. Detail Commands to Be Run

Some methods might involve multiple commands or steps while unlearning: ensure you write a clear `.sh` file that documents this.

### 3. Run and Tune Your Method on Relevant Benchmarks

- Once implemented, evaluate your method on applicable benchmarks using the best possible parameters.
- Create a folder [`community/methods/<YOUR_METHOD>`](../community/methods) and include a README file in it, explaining the method details, hyper-parameters, strategy/logic for selecting the best model for unlearning etc.
- Include a bash script `run.sh` with the exact bash command needed to replicate your results.

### 4. Update Leaderboard and Upload Model

Don't forget to add your results to the [leaderboard](results.md) and upload your unlearned model to HuggingFace for broader accessibility and reproducibility. Also, if applicable, add a link to your paper in [`docs/links.md`](../docs/links.md)

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli repo create {benchmark}-{model}-{datasplit}-{method}
cd <CHECKPOINT_DIR>

git init
git remote add origin https://huggingface.co/<username>/{benchmark}-{model}-{datasplit}-{method}
git add .
git commit -m "Initial commit"
git push origin main
```

---

## Contributing to Unlearning Benchmark Evaluations

Evaluating LLM unlearning is essential for assessing the effectiveness of different unlearning methods. While various benchmarks and metrics exist, identifying the most suitable ones for capturing the nuances of unlearning remains an open challenge.

Your contributions toward defining or improving evaluation methods can significantly advance unlearning research. By proposing reliable benchmarks, you help ensure that unlearning methods are both effective and aligned with real-world requirements.

- To add a new unlearning evaluation metric, refer to our [Metric Implementation Guide]((components.md#evaluation-metric).).
- To integrate new datasets and models, follow our [Components Guide](components.md).

### Steps to add a new Unlearning Benchmark

1. **Prepare Datasets & Models** – Create your dataset and train models to generate fine-tuned or retained models.  
2. **Define a New Benchmark** (if needed) – Follow the [Benchmark Guide]((components.md#benchmark)) to implement a new evaluation benchmark.  
3. **Run and Tune Baseline Methods** – Evaluate existing unlearning methods on your benchmark and optimize them.  
4. **Document & Share Findings** – Provide detailed steps for reproduction in [`community/benchmarks/<YOUR_BENCHMARK>`](../community/benchmarks). Also, if applicable, add a link to your paper in [`docs/links.md`](../docs/links.md)

---

## Do You Want to Add Documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate. We'll be happy to make the changes or help you make a contribution!

---

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or issues to make sure nobody is already working on the same thing. If you are unsure, it is always a good idea to open an issue to get some feedback.

Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/huggingface/transformers) by clicking on the **[Fork](https://github.com/huggingface/transformers/fork)** button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/open-unlearning.git
   cd open-unlearning
   git remote add upstream https://github.com/locuslab/open-unlearning.git
   ```

3. You can work on the forked main branch or create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

4. Set up the environment in dev mode after following steps in [Quick Start](../README.md#-quickstart). This installs other packages such as `ruff`, `precommit` etc.

   ```bash
   pip install .[dev]
   ```

5. Develop the features in your fork/branch.

   As you work on your code, you should make sure the code is linted and formatted correctly.

   OpenUnlearning relies on `ruff` to lint & format its source code consistently. After you make changes, to check the quality of code, run

   ```bash
   make quality
   ```

   If you prefer to apply the style corrections:

   ```bash
   make style
   ```

   Once you're happy with your changes, add the changed files with `git add` and record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please remember to write [good commit messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request. Make sure you tick off all the boxes on our [checklist](#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.

7. Please bear with us maintainers with the changes we require! We want to ensure we keep the repository clean and easily extensible. As you make your updates: you may want to work in your local branch and push the changes to your fork, since everyone can see the changes in the pull request. Changes pushed to the fork will automatically appear in the pull request.

### Pull Request Checklist

☐ The pull request title should summarize your contribution.  
☐ If your pull request addresses an issue, please mention the issue number in the pull request description to make sure they are linked (and people viewing the issue know you are working on it).  
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.  
☐ Make sure existing tests and checks, if any, pass.  
☐ Make methods having informative docstrings.  