"""
zlib-normalization Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""

import zlib

from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import (
    evaluate_probability,
    extract_target_texts_from_processed_data,
)


class ZLIBAttack(Attack):
    def setup(self, tokenizer=None, **kwargs):
        """Setup tokenizer."""
        self.tokenizer = tokenizer or self.model.tokenizer

    def compute_batch_values(self, batch):
        """Get loss and text for batch."""
        eval_results = evaluate_probability(self.model, batch)
        texts = extract_target_texts_from_processed_data(self.tokenizer, batch)
        return [{"loss": r["avg_loss"], "text": t} for r, t in zip(eval_results, texts)]

    def compute_score(self, sample_stats):
        """Score using loss normalized by compressed text length."""
        text = sample_stats["text"]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        return sample_stats["loss"] / zlib_entropy
