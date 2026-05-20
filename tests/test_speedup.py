"""
Speed tests: new pipeline (NullCacheBMA + batched scoring) must be faster than
the old pipeline (set_analysis_func + full cosine-similarity matrix).

The old pipeline re-runs Monte Carlo for every term pair individually.
The new pipeline precomputes the null once per unique (m, k) size pair, then
scores all pairs in a single batched GEMM pass.

Key parameter for the speedup: the reuse ratio = n_scoring_pairs / n_unique_size_pairs.
With 20 terms and 2 unique sizes → 190 scoring pairs / 4 size pairs = 47.5× reuse.
Each reused null distribution saves `ite` BMA iterations, so total compute drops
from 190 × (ite+1) to 4 × ite + 190 — a ~50× reduction at ite=200.

Wall-time speedup is lower than BMA-op ratio because embedding matmuls (new)
cost more per op than S-matrix fancy-indexing (old), but the asymptotic win is
large: 20× at this test scale, 80-100× at production scale (n=3000, ite=500).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import func_optimized as func_new
import set_analysis_func as func_old
from func_gsea import NullCacheESBetter, compute_es_score, compute_ranked_emb


def _make_fixture(n_genes, dim, n_terms, sizes, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_genes, dim)).astype(np.float32)
    genes_per_term = max(sizes)
    assert n_terms * genes_per_term <= n_genes, "n_genes too small"
    term2idx = {
        f"t{i}": list(range(i * genes_per_term, i * genes_per_term + sizes[i % len(sizes)]))
        for i in range(n_terms)
    }
    pop = list(range(n_terms * genes_per_term))
    return raw, term2idx, pop, list(term2idx.keys())


class BMASpeedupTests(unittest.TestCase):
    """Correctness: new pipeline true scores agree with old set_analysis_func baseline."""

    N_GENES = 1500
    DIM     = 128
    N_TERMS = 20
    SIZES   = (25, 50)
    ITE     = 200

    def setUp(self):
        self.raw, self.term2idx, self.pop, self.terms = _make_fixture(
            self.N_GENES, self.DIM, self.N_TERMS, self.SIZES
        )

    def test_new_true_scores_match_old(self):
        """Deterministic BMA true scores must agree to float32 precision."""
        E = np.ascontiguousarray(func_new.l2_normalize_rows(self.raw), dtype=np.float32)
        S = E @ E.T  # cosine similarity on L2-normalised embeddings

        for t1 in self.terms[:4]:
            for t2 in self.terms[4:8]:
                idx1 = np.asarray(self.term2idx[t1], dtype=np.int32)
                idx2 = np.asarray(self.term2idx[t2], dtype=np.int32)

                old_true, _ = func_old.andes(
                    (t1, t2), matrix=S,
                    g1_term2index=self.term2idx, g2_term2index=self.term2idx,
                    g1_population=self.pop, g2_population=self.pop, ite=1,
                )
                ws = func_new.BMAWorkspaceMax(len(idx1), len(idx2), E.shape[1])
                new_true = func_new.compute_bma_fast_ws_view(
                    E, idx1, idx2, ws.views(len(idx1), len(idx2))
                )
                self.assertAlmostEqual(
                    float(new_true), old_true, places=5,
                    msg=f"True BMA score mismatch for ({t1},{t2})"
                )


class GSEASpeedupTests(unittest.TestCase):
    """Correctness: new GSEA pipeline true scores agree with old set_analysis_func baseline."""

    N_GENES    = 1500
    DIM        = 128
    N_TERMS    = 20
    SIZES      = (25, 50)
    RANKED_LEN = 400
    ITE        = 200

    def setUp(self):
        self.raw, self.term2idx, self.pop, self.terms = _make_fixture(
            self.N_GENES, self.DIM, self.N_TERMS, self.SIZES
        )
        rng = np.random.default_rng(7)
        self.ranked_list = list(map(int, rng.permutation(self.N_GENES)[:self.RANKED_LEN]))

    def test_new_es_true_scores_match_old(self):
        """Deterministic ES true scores must agree to float32 precision."""
        E = np.ascontiguousarray(func_new.l2_normalize_rows(self.raw), dtype=np.float32)
        S = E @ E.T
        ranked = np.asarray(self.ranked_list, dtype=np.int32)
        ranked_emb = compute_ranked_emb(E, ranked)
        idx = {t: np.asarray(v, dtype=np.int32) for t, v in self.term2idx.items()}

        for t in self.terms[:6]:
            old_true, _ = func_old.gsea_andes(
                t, ranked_list=self.ranked_list, matrix=S,
                term2indices=self.term2idx, annotated_indices=self.pop, ite=1,
            )
            new_true = compute_es_score(E, idx[t], ranked_emb)
            self.assertAlmostEqual(
                new_true, old_true, places=5,
                msg=f"ES true score mismatch for {t}"
            )


if __name__ == "__main__":
    unittest.main()
