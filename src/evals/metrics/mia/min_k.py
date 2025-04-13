"""
Min-k % Prob Attack: https://arxiv.org/pdf/2310.16789.pdf
"""

import numpy as np
from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs


class MinKProbAttack(Attack):
    def setup(self, k=0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        """Get token-wise log probabilities for the batch."""
        return tokenwise_logprobs(self.model, batch, grad=False)

    def compute_score(self, sample_stats):
        """Score single sample using min-k negative log probs scores attack."""
        lp = sample_stats.cpu().numpy()
        if lp.size == 0:
            return 0

        num_k = max(1, int(len(lp) * self.k))
        sorted_vals = np.sort(lp)
        return -np.mean(sorted_vals[:num_k])
