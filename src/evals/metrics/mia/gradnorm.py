"""
Gradient-norm attack. Proposed for MIA in multiple settings, and particularly
experimented for pre-training data and LLMs in https://arxiv.org/abs/2402.17012
"""

import torch
from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs


# DO NOT use gradnorm in a way so that it runs when your accumulated gradients during training aren't used yet
# gradnorm zeros out the gradients of the model during its computation
class GradNormAttack(Attack):
    def setup(self, p, **kwargs):
        if p not in [1, 2, float("inf")]:
            raise ValueError(f"Invalid p-norm value: {p}")
        self.p = p

    def compute_batch_values(self, batch):
        """Compute gradients of examples w.r.t model parameters. More grad norm => more loss."""
        self.model.train()
        batch_log_probs = tokenwise_logprobs(self.model, batch, grad=True)
        batch_loss = [-torch.mean(lps) for lps in batch_log_probs]
        batch_grad_norms = []
        for sample_loss in batch_loss:
            sample_grad_norms = []
            self.model.zero_grad()
            sample_loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    sample_grad_norms.append(param.grad.detach().norm(p=self.p))
            batch_grad_norms.append(torch.stack(sample_grad_norms).mean())
        self.model.eval()
        return batch_grad_norms

    def compute_score(self, sample_stats):
        """Return negative gradient norm as the attack score."""
        return sample_stats.cpu().to(torch.float32).numpy()
