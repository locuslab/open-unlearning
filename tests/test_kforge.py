"""CPU regression checks for the K-FORGE edit math. Run with: python tests/test_kforge.py"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainer.unlearn.kforge import _KronStats, KFORGE  # noqa: E402


def make_trainer(**overrides):
    """Bypass HF Trainer init; the edit math only reads these attributes."""
    trainer = KFORGE.__new__(KFORGE)
    values = dict(
        rank=64,
        strength=1.0,
        damping=1e-3,
        damping_mode="fixed",
        adaptive_damping_coeff=0.1,
        damping_floor=1e-12,
        factor_mode="kron",
        use_retain_fisher=True,
        normalize_factors=True,
        edit_variant="wiener_v2",
        lambda_tradeoff=0.01,
        edit_weight_dtype="float32",
    )
    values.update(overrides)
    for name, value in values.items():
        setattr(trainer, name, value)
    return trainer


def diagonal_stats(m, n, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.diag(torch.rand(m, generator=gen, dtype=torch.float64) + 0.5)
    b = torch.diag(torch.rand(n, generator=gen, dtype=torch.float64) + 0.5)
    return _KronStats(a=a, b=b, count=32)


def dense_spd(m, seed):
    gen = torch.Generator().manual_seed(seed)
    root = torch.randn(m, m, generator=gen, dtype=torch.float64)
    return root @ root.T + torch.eye(m, dtype=torch.float64) * 0.1


def random_weight(n, m, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, m, generator=gen, dtype=torch.float64)


class KFORGEEditMathTests(unittest.TestCase):
    def test_causal_lm_token_mask_shifts_left(self):
        labels = torch.tensor(
            [
                [1, 2, 3, -100, -100],
                [4, -100, 5, -100, -100],
            ]
        )
        mask, count = KFORGE._causal_lm_token_mask(labels)
        expected = torch.zeros_like(labels, dtype=torch.bool)
        expected[0, :2] = True  # inputs predicting labels at positions 1, 2
        expected[1, 1] = True  # input predicting the label at position 2
        self.assertEqual(count, 3)
        self.assertTrue(torch.equal(mask, expected))

    def test_prepare_factor_symmetrizes_and_damps_relative_to_own_scale(self):
        trainer = make_trainer(damping=0.1)
        matrix = torch.tensor([[4.0, 1.0], [0.0, 100.0]])
        out = trainer._prepare_factor(matrix)
        expected = torch.tensor([[4.0, 0.5], [0.5, 100.0]]) + 0.1 * 52.0 * torch.eye(2)
        self.assertTrue(torch.allclose(out, expected))

    def test_wiener_v2_diagonal_full_rank_matches_closed_form(self):
        m, n, lam, strength, damping = 6, 4, 0.3, 0.7, 1e-3
        trainer = make_trainer(
            factor_mode="diagonal",
            strength=strength,
            lambda_tradeoff=lam,
            damping=damping,
            use_retain_fisher=True,
        )
        forget = diagonal_stats(m, n, seed=0)
        retain = diagonal_stats(m, n, seed=1)
        weight = random_weight(n, m, seed=2)

        delta, _ = trainer._compute_module_delta(weight, forget, retain)

        def damped_diag(stats):
            # fixed-mode damping adds damping * mean(diag) to each factor
            a_eps = damping * stats.a.diagonal().mean()
            b_eps = damping * stats.b.diagonal().mean()
            return stats.a.diagonal() + a_eps, stats.b.diagonal() + b_eps

        df_a, df_b = damped_diag(forget)
        dr_a, dr_b = damped_diag(retain)
        d_f = df_b[:, None] * df_a[None, :]
        d_r = dr_b[:, None] * dr_a[None, :]
        expected = -strength * (d_f / (d_f + lam * d_r)) * weight
        self.assertTrue(torch.allclose(delta.cpu(), expected, atol=1e-8))

    def test_wiener_v2_kron_lambda_zero_full_rank_returns_negative_weight(self):
        m, n = 6, 4
        trainer = make_trainer(
            rank=64,  # >= min(n, m): truncation disabled
            lambda_tradeoff=0.0,
            strength=0.75,
        )
        forget = _KronStats(a=dense_spd(m, 3), b=dense_spd(n, 4), count=32)
        retain = _KronStats(a=dense_spd(m, 5), b=dense_spd(n, 6), count=32)
        weight = random_weight(n, m, seed=7)

        delta, top_sigma = trainer._compute_module_delta(weight, forget, retain)

        self.assertEqual(delta.shape, weight.shape)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertTrue(torch.allclose(delta.cpu(), -0.75 * weight, atol=1e-8))
        self.assertGreater(top_sigma, 0.0)


if __name__ == "__main__":
    unittest.main()
