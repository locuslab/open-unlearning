# Constrained Entropic Unlearning: A Primal-Dual Framework for Large Language Models


 We propose a new formulation of LLM unlearning
as a constrained optimization problem: forgetting is enforced via a novel logitmargin flattening loss
 that explicitly drives the output distribution toward uniformity on a designated forget set,
 while retention is preserved through a hard constraint on a separate retain set. 
We solve the constrained problem using a scalable primal-dual algorithm that exposes the 
 trade-off between forgetting and  retention through the dynamics of the dual variable.

# Setup

Experimental setup

-  **Hyperparameters & Search Space:** Please see the corresponding [paper](https://arxiv.org/abs/2506.05314) for details of the hyperparameter. Importantly
    to produce good results using our method, it is vital the hyperparameter `retain_loss_eps` is set to an appropriate value.
    To choose such a value, simply look at the value of the retain loss of the pretrained model and choose
    an appropriately larger value than this starting value.

    Note that our method's loss is a quadratic function of a difference in the logit spaces. Consequently, 
    the value of this loss can be large. As a result, it is natural that we set the initial parameter of the
    retain loss preference to 50 or 100.
-  **Computational Setup:** Please see the Supplementary Material in the paper.

# Results

Please see the `run.sh` script that contains all necessary commands to reproduce the final results.

All unlearned models are available under https://huggingface.co/tamarsonha. 

# Citation


If you use this work, please cite:


[//]: @misc{entesari2025constrainedentropicunlearningprimaldual,
      title={Constrained Entropic Unlearning: A Primal-Dual Framework for Large Language Models}, 
      author={Taha Entesari and Arman Hatami and Rinat Khaziev and Anil Ramakrishna and Mahyar Fazlyab},
      year={2025},
      eprint={2506.05314},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.05314},}