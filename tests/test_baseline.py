"""Baseline tests: scientific correctness and cache-naming.

Uses identity-matrix embeddings (orthonormal basis vectors) where BMA and ES
behaviour is fully predictable without any approximation:

  BMA with eye(N) embedding
  ─────────────────────────
  row_max for gene a_i = max_j sim(a_i, b_j) = 1 if a_i ∈ B, else 0
  → BMA(A, B) = 2·|A∩B| / (|A|+|B|)

  Consequences:
    BMA(A, A) = 1.0                  (identical sets)
    BMA(A, B) = 0.0  when A∩B = ∅   (disjoint sets)

  Null distribution (random sets of size m from N orthogonal genes):
    E[BMA_null] = 2·m·k / (N·(m+k))  ← well below 1.0 for m,k ≪ N
    → identical pair z-score >> 0
    → disjoint pair z-score < 0

  ES with eye(N) and ranked list = [0, 1, ..., L-1]
  ──────────────────────────────────────────────────
  col_max[j] = 1 if ranked[j] ∈ gene_set, else 0
  → gene set at top   (positions 0..m-1)  → ES > 0
  → gene set at bottom (positions L-m..L-1) → ES < 0
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import func_optimized as bma
from func_gsea import NullCacheESBetter as NullCacheES, compute_es_score, compute_ranked_emb


class BMABaselineTests(unittest.TestCase):
    N = 50  # gene vocabulary; eye(N) gives N orthonormal basis vectors

    def setUp(self):
        self.E = np.eye(self.N, dtype=np.float32)
        self.pop = np.arange(self.N, dtype=np.int32)

    def _cache(self, size_pairs, ite=50):
        c = bma.NullCacheBMA()
        c.precompute(self.E, self.pop, size_pairs, ite=ite, seed=42, verbose=False)
        return c

    def test_bma_identical_sets_is_one(self):
        idx = np.arange(5, dtype=np.int32)
        self.assertAlmostEqual(bma.compute_bma_numba(self.E, idx, idx), 1.0, places=5)

    def test_bma_disjoint_sets_is_zero(self):
        x = np.arange(5, dtype=np.int32)
        y = np.arange(5, 10, dtype=np.int32)
        self.assertAlmostEqual(bma.compute_bma_numba(self.E, x, y), 0.0, places=5)

    def test_identical_pair_zscore_positive(self):
        m = 5
        idx = np.arange(m, dtype=np.int32)
        cache = self._cache({(m, m)})
        score = bma.compute_bma_numba(self.E, idx, idx)  # = 1.0
        self.assertGreater(cache.get_zscore(score, m, m), 0.0)

    def test_disjoint_pair_zscore_negative(self):
        m = 5
        x = np.arange(m, dtype=np.int32)
        y = np.arange(m, 2 * m, dtype=np.int32)
        cache = self._cache({(m, m)})
        score = bma.compute_bma_numba(self.E, x, y)  # = 0.0
        self.assertLess(cache.get_zscore(score, m, m), 0.0)

    def test_overlapping_beats_disjoint_in_full_pipeline(self):
        """Batched pipeline: overlapping query ranks higher than disjoint."""
        m = 4
        q_idx = {"q": np.arange(m, dtype=np.int32)}
        bg_idx = {
            "same":     np.arange(m, dtype=np.int32),
            "disjoint": np.arange(m, 2 * m, dtype=np.int32),
        }
        cache = self._cache({(m, m)})
        bq = bma.precompute_term_embedding_blocks(self.E, q_idx)
        bb = bma.precompute_term_embedding_blocks(self.E, bg_idx)

        zs, _, _ = bma.score_bma_zscore_matrix_batched(
            ["q"], ["same", "disjoint"], cache, bq, bb,
            symmetric=False, n_workers=1,
        )
        self.assertGreater(zs[0, 0], zs[0, 1],
                           "overlapping pair must rank above disjoint pair")


class GSEABaselineTests(unittest.TestCase):
    N = 40   # total genes
    L = 20   # ranked list length
    M = 5    # gene set size

    def setUp(self):
        self.E = np.eye(self.N, dtype=np.float32)
        self.pop = np.arange(self.N, dtype=np.int32)
        self.ranked_emb = compute_ranked_emb(
            self.E, np.arange(self.L, dtype=np.int32)
        )

    def _cache(self, sizes, ite=50):
        c = NullCacheES()
        c.precompute(self.E, self.pop, sizes, self.ranked_emb,
                     ite=ite, seed=42, verbose=False)
        return c

    def test_es_top_gene_set_positive(self):
        """Gene set at top of ranked list must have ES > 0."""
        gene_set = np.arange(self.M, dtype=np.int32)
        self.assertGreater(compute_es_score(self.E, gene_set, self.ranked_emb), 0.0)

    def test_es_bottom_gene_set_negative(self):
        """Gene set at bottom of ranked list must have ES < 0."""
        gene_set = np.arange(self.L - self.M, self.L, dtype=np.int32)
        self.assertLess(compute_es_score(self.E, gene_set, self.ranked_emb), 0.0)

    def test_top_gene_set_zscore_positive(self):
        gene_set = np.arange(self.M, dtype=np.int32)
        cache = self._cache({self.M})
        score = compute_es_score(self.E, gene_set, self.ranked_emb)
        self.assertGreater(cache.get_zscore(score, self.M), 0.0)

    def test_bottom_gene_set_zscore_negative(self):
        gene_set = np.arange(self.L - self.M, self.L, dtype=np.int32)
        cache = self._cache({self.M})
        score = compute_es_score(self.E, gene_set, self.ranked_emb)
        self.assertLess(cache.get_zscore(score, self.M), 0.0)

    def test_top_ranks_above_bottom(self):
        """Top-enriched set must have strictly higher z-score than bottom-depleted set."""
        top_set = np.arange(self.M, dtype=np.int32)
        bot_set = np.arange(self.L - self.M, self.L, dtype=np.int32)
        cache = self._cache({self.M})
        z_top = cache.get_zscore(compute_es_score(self.E, top_set, self.ranked_emb), self.M)
        z_bot = cache.get_zscore(compute_es_score(self.E, bot_set, self.ranked_emb), self.M)
        self.assertGreater(z_top, z_bot)


class CacheNamingTests(unittest.TestCase):
    """suggest_path returns content-addressed filenames."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.E  = rng.normal(size=(20, 8)).astype(np.float32)
        self.p1 = np.arange(10, dtype=np.int32)
        self.p2 = np.arange(10, 20, dtype=np.int32)
        ranked_idx = np.arange(8, dtype=np.int32)
        self.re = compute_ranked_emb(self.E, ranked_idx)

    def test_bma_same_inputs_same_path(self):
        p1 = bma.NullCacheBMA.suggest_path("cache", self.E, self.p1, self.p2)
        p2 = bma.NullCacheBMA.suggest_path("cache", self.E, self.p1, self.p2)
        self.assertEqual(p1, p2)

    def test_bma_different_embedding_different_path(self):
        E2 = self.E * 2.0
        p1 = bma.NullCacheBMA.suggest_path("cache", self.E,  self.p1, self.p2)
        p2 = bma.NullCacheBMA.suggest_path("cache", E2, self.p1, self.p2)
        self.assertNotEqual(p1, p2)

    def test_bma_different_population_different_path(self):
        p1 = bma.NullCacheBMA.suggest_path("cache", self.E, self.p1, self.p2)
        p2 = bma.NullCacheBMA.suggest_path("cache", self.E, self.p2, self.p1)
        self.assertNotEqual(p1, p2)

    def test_bma_path_under_base_dir(self):
        path = bma.NullCacheBMA.suggest_path("mydir", self.E, self.p1, self.p2)
        self.assertTrue(path.startswith("mydir" + os.sep))
        self.assertTrue(path.endswith(".pkl"))

    def test_es_same_inputs_same_path(self):
        p1 = NullCacheES.suggest_path("cache", self.E, self.p1, self.re)
        p2 = NullCacheES.suggest_path("cache", self.E, self.p1, self.re)
        self.assertEqual(p1, p2)

    def test_es_different_ranked_emb_different_path(self):
        ranked2 = self.re[::-1].copy()
        p1 = NullCacheES.suggest_path("cache", self.E, self.p1, self.re)
        p2 = NullCacheES.suggest_path("cache", self.E, self.p1, ranked2)
        self.assertNotEqual(p1, p2)

    def test_bma_path_is_deterministic_across_calls(self):
        paths = [bma.NullCacheBMA.suggest_path("cache", self.E, self.p1, self.p2)
                 for _ in range(5)]
        self.assertEqual(len(set(paths)), 1)


if __name__ == "__main__":
    unittest.main()
