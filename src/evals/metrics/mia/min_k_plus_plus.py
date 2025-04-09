import torch as torch
import numpy as np
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.utils import tokenwise_vocab_logprobs, tokenwise_logprobs


class MinKPlusPlusAttack(MinKProbAttack):
    def compute_batch_values(self, batch):
        """Get both token-wise and vocab-wise log probabilities for the batch."""
        vocab_log_probs = tokenwise_vocab_logprobs(self.model, batch, grad=False)
        token_log_probs = tokenwise_logprobs(self.model, batch, grad=False)
        return [
            {"vocab_log_probs": vlp, "token_log_probs": tlp}
            for vlp, tlp in zip(vocab_log_probs, token_log_probs)
        ]

    def compute_score(self, sample_stats):
        """Score using min-k negative log probs scores with vocab-wise normalization."""
        all_probs = sample_stats["vocab_log_probs"]
        target_prob = sample_stats["token_log_probs"]

        if len(target_prob) == 0:
            return 0

        # Compute normalized scores using vocab distribution
        mu = (torch.exp(all_probs) * all_probs).sum(-1)
        sigma = (torch.exp(all_probs) * torch.square(all_probs)).sum(-1) - torch.square(
            mu
        )

        # Handle numerical stability
        sigma = torch.clamp(sigma, min=1e-6)
        scores = (target_prob.cpu().numpy() - mu.cpu().numpy()) / torch.sqrt(
            sigma
        ).cpu().numpy()

        # Take bottom k% as the attack score
        num_k = max(1, int(len(scores) * self.k))
        return -np.mean(sorted(scores)[:num_k])
