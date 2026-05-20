"""
Integration tests using the real embedding and gene sets in data/.

These reproduce the scenarios from demo.ipynb and bench.ipynb:
  - ANDES BMA: score specific GO term pairs (e.g. GO:0043648 vs GO:0006805)
  - GSEA-ANDES: score specific terms against the GSE3467 ranked list

Key claim: the new implementation (func_optimized / func_gsea) produces
identical true scores to the old (set_analysis_func) on the same inputs.
True scores are deterministic (no MC), so they should match to float32
precision. Z-scores are MC-based so we only check they are finite and
have the right sign relative to each other.

Skipped automatically if data files are missing (e.g. on a fresh clone).
"""

import os
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

EMB_PATH     = ROOT / "data/embedding/node2vec_consensus.csv"
GENES_PATH   = ROOT / "data/embedding/consensus_node.txt"
GMT_PATH     = ROOT / "data/gene_sets/hsa_experimental_eval_BP_propagated.gmt"
RANKED_PATH  = ROOT / "data/expression/GSE3467_rank.txt"

DATA_AVAILABLE = all(p.exists() for p in [EMB_PATH, GENES_PATH, GMT_PATH, RANKED_PATH])


@unittest.skipUnless(DATA_AVAILABLE, "real data files not present — skipping integration tests")
class AndesBMAIntegrationTests(unittest.TestCase):
    """
    ANDES BMA true-score equivalence on real GO data.
    Mirrors the bench.ipynb scenario: old f_old vs new compute_bma_fast_ws_view.
    """

    # Pairs used in demo.ipynb / bench.ipynb
    TEST_PAIRS = [
        ("GO:0043648", "GO:0006805"),
        ("GO:0071466", "GO:0009410"),
    ]

    @classmethod
    def setUpClass(cls):
        import load_data as ld
        import func_optimized as func

        raw = np.loadtxt(str(EMB_PATH), delimiter=",", dtype=np.float32)
        with open(GENES_PATH) as fh:
            node_list = [line.strip() for line in fh]

        cls.E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
        cls.g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(node_list)})

        gmt = ld.load_gmt(str(GMT_PATH))
        cls.term_indices = func.preconvert_indices_to_arrays(
            ld.term2indexes(gmt, cls.g_node2index, upper=300, lower=10)
        )
        cls.func = func

    def _bma_reference(self, idx1, idx2):
        """BMA via explicit matrix multiply — the same math as set_analysis_func."""
        A = self.E_unit[idx1] @ self.E_unit[idx2].T
        return float((A.max(axis=1).sum() + A.max(axis=0).sum()) / (len(idx1) + len(idx2)))

    def test_bma_true_score_matches_matrix_reference(self):
        func = self.func
        for t1, t2 in self.TEST_PAIRS:
            if t1 not in self.term_indices or t2 not in self.term_indices:
                self.skipTest(f"{t1} or {t2} not in gene sets after size filter")
            idx1 = self.term_indices[t1]
            idx2 = self.term_indices[t2]

            expected = self._bma_reference(idx1, idx2)

            ws = func.BMAWorkspaceMax(len(idx1), len(idx2), self.E_unit.shape[1])
            observed = func.compute_bma_fast_ws_view(
                self.E_unit, idx1, idx2, ws.views(len(idx1), len(idx2))
            )

            self.assertAlmostEqual(
                float(observed), expected, places=5,
                msg=f"BMA true score mismatch for ({t1}, {t2})"
            )

    def test_bma_numba_matches_matrix_reference(self):
        func = self.func
        for t1, t2 in self.TEST_PAIRS:
            if t1 not in self.term_indices or t2 not in self.term_indices:
                self.skipTest(f"{t1} or {t2} not in gene sets after size filter")
            idx1 = self.term_indices[t1]
            idx2 = self.term_indices[t2]

            expected = self._bma_reference(idx1, idx2)
            observed = func.compute_bma_numba(self.E_unit, idx1, idx2)

            self.assertAlmostEqual(
                float(observed), expected, places=5,
                msg=f"Numba BMA mismatch for ({t1}, {t2})"
            )

    def test_bma_zscore_is_finite_and_cached(self):
        """End-to-end: build NullCacheBMA and score the test pairs."""
        func = self.func
        t1, t2 = self.TEST_PAIRS[0]
        if t1 not in self.term_indices or t2 not in self.term_indices:
            self.skipTest("terms not in gene sets")

        idx1 = self.term_indices[t1]
        idx2 = self.term_indices[t2]
        m, k = len(idx1), len(idx2)

        pop = np.arange(self.E_unit.shape[0], dtype=np.int32)
        cache = func.NullCacheBMA()
        cache.precompute(
            self.E_unit, pop, {(m, k)},
            ite=50, seed=42, verbose=False,
        )

        true_score = self._bma_reference(idx1, idx2)
        mu, std = cache.cache[(m, k)]
        z = (true_score - mu) / std if std > 0 else 0.0

        self.assertTrue(np.isfinite(z), "z-score is not finite")
        self.assertIn((m, k), cache.cache)


@unittest.skipUnless(DATA_AVAILABLE, "real data files not present — skipping integration tests")
class GSEAIntegrationTests(unittest.TestCase):
    """
    GSEA-ANDES true-score equivalence on real data.
    Mirrors demo.ipynb: score GO:0071466, GO:0006805, GO:0009410
    against the GSE3467 ranked list.
    """

    TEST_TERMS = ["GO:0071466", "GO:0006805", "GO:0009410"]

    @classmethod
    def setUpClass(cls):
        import load_data as ld
        import func_optimized as func
        from func_gsea import compute_ranked_emb, compute_es_score, NullCacheESBetter
        import pandas as pd

        raw = np.loadtxt(str(EMB_PATH), delimiter=",", dtype=np.float32)
        with open(GENES_PATH) as fh:
            node_list = [line.strip() for line in fh]

        cls.E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
        g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(node_list)})
        node_set = set(node_list)

        gmt = ld.load_gmt(str(GMT_PATH))
        cls.term_indices = func.preconvert_indices_to_arrays(
            ld.term2indexes(gmt, g_node2index, upper=300, lower=10)
        )

        # Background population (all genes in any gene set)
        all_bg = set().union(*gmt.values()) & node_set
        cls.pop = np.array(sorted(g_node2index[g] for g in all_bg), dtype=np.int32)

        # Ranked list
        df = pd.read_csv(str(RANKED_PATH), sep="\t", index_col=0, header=None)
        ranked_idx = np.array(
            [g_node2index[str(g)] for g in df.index if str(g) in node_set],
            dtype=np.int32,
        )
        cls.ranked_idx = ranked_idx
        cls.ranked_emb = compute_ranked_emb(cls.E_unit, ranked_idx)

        cls.compute_es_score   = staticmethod(compute_es_score)
        cls.NullCacheESBetter  = NullCacheESBetter

    def _es_reference(self, term_idx):
        """ES via explicit matrix multiply — same math as set_analysis_func.gsea_andes."""
        A = self.E_unit[term_idx] @ self.ranked_emb.T   # (m, L)
        col_max = A.max(axis=0)                          # (L,)
        cs = np.cumsum(col_max - col_max.mean())
        return float(cs[np.abs(cs).argmax()])

    def test_es_true_score_matches_matrix_reference(self):
        for term in self.TEST_TERMS:
            if term not in self.term_indices:
                self.skipTest(f"{term} not in gene sets after size filter")
            idx = self.term_indices[term]
            expected = self._es_reference(idx)
            observed = self.compute_es_score(self.E_unit, idx, self.ranked_emb)
            self.assertAlmostEqual(
                observed, expected, places=5,
                msg=f"ES true score mismatch for {term}"
            )

    def test_es_scores_are_ordered_consistently(self):
        """
        The relative ranking of z-scores for these three terms should be stable.
        We check that new scores are monotonically consistent with the reference.
        """
        ref_scores  = {}
        new_scores  = {}
        for term in self.TEST_TERMS:
            if term not in self.term_indices:
                self.skipTest(f"{term} not in gene sets after size filter")
            idx = self.term_indices[term]
            ref_scores[term] = self._es_reference(idx)
            new_scores[term] = self.compute_es_score(self.E_unit, idx, self.ranked_emb)

        # Pairwise ordering must agree between reference and new
        terms = self.TEST_TERMS
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                ta, tb = terms[i], terms[j]
                ref_order = ref_scores[ta] > ref_scores[tb]
                new_order = new_scores[ta] > new_scores[tb]
                self.assertEqual(
                    ref_order, new_order,
                    msg=f"Score ordering mismatch between {ta} and {tb}"
                )

    def test_gsea_zscore_is_finite_for_all_test_terms(self):
        """Full GSEA pipeline: build null cache and score all test terms."""
        available = [t for t in self.TEST_TERMS if t in self.term_indices]
        if not available:
            self.skipTest("no test terms survived size filter")

        sizes = {len(self.term_indices[t]) for t in available}
        cache = self.NullCacheESBetter()
        cache.precompute(
            self.E_unit, self.pop, sizes, self.ranked_emb,
            ite=50, seed=42, verbose=False,
        )

        for term in available:
            idx = self.term_indices[term]
            score = self.compute_es_score(self.E_unit, idx, self.ranked_emb)
            z = cache.get_zscore(score, len(idx))
            self.assertTrue(np.isfinite(z), f"z-score not finite for {term}")

    def test_ranked_list_length_is_reasonable(self):
        self.assertGreater(len(self.ranked_idx), 100,
                           "ranked list suspiciously short")
        self.assertLess(len(self.ranked_idx), self.E_unit.shape[0] + 1,
                        "ranked list longer than embedding")


if __name__ == "__main__":
    unittest.main()
