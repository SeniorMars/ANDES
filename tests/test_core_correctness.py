import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import func_optimized as bma
from func_gsea import (
    NullCacheESBetter as NullCacheES,
    compute_es_score,
    compute_es_trace,
    compute_ranked_emb,
)


def bma_from_matrix(matrix):
    row_max = matrix.max(axis=1)
    col_max = matrix.max(axis=0)
    return float((row_max.sum() + col_max.sum()) / (matrix.shape[0] + matrix.shape[1]))


def es_from_matrix(matrix):
    col_max = matrix.max(axis=0)
    cs = np.cumsum(col_max - col_max.mean())
    return float(cs[np.abs(cs).argmax()])


class BMACorrectnessTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        raw = rng.normal(size=(32, 12)).astype(np.float32)
        self.E = np.ascontiguousarray(bma.l2_normalize_rows(raw), dtype=np.float32)

    def test_bma_numba_matches_matrix_reference(self):
        x_idx = np.array([0, 2, 4, 6, 8], dtype=np.int32)
        y_idx = np.array([1, 3, 5, 7, 9, 11], dtype=np.int32)
        expected = bma_from_matrix(self.E[x_idx] @ self.E[y_idx].T)
        observed = bma.compute_bma_numba(self.E, x_idx, y_idx)
        self.assertAlmostEqual(observed, expected, places=6)

    def test_bma_block_scorer_matches_take_scorer(self):
        x_idx = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        y_idx = np.array([6, 7, 8, 9, 10], dtype=np.int32)
        ws = bma.BMAWorkspaceMax(len(x_idx), len(y_idx), self.E.shape[1])
        via_take = bma.compute_bma_fast_ws_view(
            self.E, x_idx, y_idx, ws.views(len(x_idx), len(y_idx))
        )
        blocks = bma.precompute_term_embedding_blocks(
            self.E, {"x": x_idx, "y": y_idx}
        )
        via_blocks = bma.compute_bma_blocks_ws(
            blocks["x"], blocks["y"], ws.views(len(x_idx), len(y_idx))
        )
        self.assertAlmostEqual(via_blocks, via_take, places=6)

    def test_bma_null_uses_separate_backgrounds(self):
        E = np.eye(4, dtype=np.float32)
        pop1 = np.array([0, 1], dtype=np.int32)
        pop2 = np.array([2, 3], dtype=np.int32)
        cache = bma.NullCacheBMA()
        cache.precompute(E, pop1, {(2, 2)}, ite=2, seed=7, verbose=False, population_idx2=pop2)
        mean, std = cache.cache[(2, 2)]
        self.assertEqual(mean, 0.0)
        self.assertEqual(std, 0.0)

    def test_bma_cache_metadata_roundtrip_and_rejects_wrong_background(self):
        E = np.eye(5, dtype=np.float32)
        pop1 = np.array([0, 1, 2], dtype=np.int32)
        pop2 = np.array([2, 3, 4], dtype=np.int32)
        cache = bma.NullCacheBMA()
        cache.precompute(E, pop1, {(2, 2)}, ite=2, seed=11, verbose=False, population_idx2=pop2)

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        try:
            cache.save(path)
            loaded = bma.NullCacheBMA()
            loaded.load(path)
        finally:
            os.unlink(path)

        expected = bma.NullCacheBMA.build_metadata(E, pop1, pop2, ite=2, seed=11)
        self.assertTrue(loaded.metadata_matches(expected)[0])
        wrong = bma.NullCacheBMA.build_metadata(E, pop2, pop1, ite=2, seed=11)
        self.assertFalse(loaded.metadata_matches(wrong)[0])

    def test_cost_chunking_preserves_all_pairs(self):
        pairs = {(10, 10), (20, 200), (200, 20), (300, 300), (50, 50)}
        chunks = bma._chunked_by_cost(pairs, ite=100, n_workers=2)
        flattened = [pair for chunk in chunks for pair in chunk]
        self.assertEqual(set(flattened), pairs)
        self.assertEqual(len(flattened), len(pairs))

    def test_threaded_matrix_scoring_matches_single_worker(self):
        terms = ["a", "b", "c"]
        indices = {
            "a": np.array([0, 1, 2], dtype=np.int32),
            "b": np.array([3, 4, 5], dtype=np.int32),
            "c": np.array([6, 7, 8], dtype=np.int32),
        }
        blocks = bma.precompute_term_embedding_blocks(self.E, indices)
        cache = bma.NullCacheBMA()
        for m in {len(v) for v in indices.values()}:
            cache.cache[(m, m)] = (0.0, 1.0)
        one = bma.score_bma_zscore_matrix(
            self.E,
            terms,
            terms,
            indices,
            indices,
            cache,
            blocks,
            blocks,
            symmetric=True,
            n_workers=1,
        )
        threaded = bma.score_bma_zscore_matrix(
            self.E,
            terms,
            terms,
            indices,
            indices,
            cache,
            blocks,
            blocks,
            symmetric=True,
            n_workers=2,
        )
        np.testing.assert_allclose(threaded, one, rtol=1e-6, atol=1e-6)

    def test_batched_matrix_scoring_matches_pairwise(self):
        terms = ["a", "b", "c"]
        indices = {
            "a": np.array([0, 1, 2], dtype=np.int32),
            "b": np.array([3, 4, 5, 6], dtype=np.int32),
            "c": np.array([7, 8, 9], dtype=np.int32),
        }
        blocks = bma.precompute_term_embedding_blocks(self.E, indices)
        cache = bma.NullCacheBMA()
        sizes = {len(v) for v in indices.values()}
        for m in sizes:
            for k in sizes:
                cache.cache[(m, k)] = (0.0, 1.0)
        pairwise = bma.score_bma_zscore_matrix(
            self.E,
            terms,
            terms,
            indices,
            indices,
            cache,
            blocks,
            blocks,
            symmetric=True,
            n_workers=1,
            numba_threshold=0,
        )
        batched, _, _ = bma.score_bma_zscore_matrix_batched(
            terms,
            terms,
            cache,
            blocks,
            blocks,
            symmetric=True,
            n_workers=1,
        )
        np.testing.assert_allclose(batched, pairwise, rtol=1e-5, atol=1e-5)


class GSEACorrectnessTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(456)
        raw = rng.normal(size=(40, 10)).astype(np.float32)
        self.E = np.ascontiguousarray(bma.l2_normalize_rows(raw), dtype=np.float32)

    def test_es_score_matches_matrix_reference(self):
        gene_set = np.array([0, 3, 5, 9], dtype=np.int32)
        ranked = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
        ranked_emb = compute_ranked_emb(self.E, ranked)
        expected = es_from_matrix(self.E[gene_set] @ ranked_emb.T)
        observed = compute_es_score(self.E, gene_set, ranked_emb)
        self.assertAlmostEqual(observed, expected, places=5)

    def test_es_trace_matches_es_score(self):
        gene_set = np.array([0, 3, 5, 9], dtype=np.int32)
        ranked = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
        ranked_emb = compute_ranked_emb(self.E, ranked)
        trace = compute_es_trace(self.E, gene_set, ranked_emb)
        observed = compute_es_score(self.E, gene_set, ranked_emb)

        self.assertAlmostEqual(trace["es"], observed, places=5)
        self.assertEqual(trace["running_es"].shape[0], ranked.shape[0])
        self.assertEqual(trace["best_match_score"].shape[0], ranked.shape[0])
        self.assertTrue(np.all(trace["best_gene_set_position"] < len(gene_set)))

    def test_es_parallel_matches_serial(self):
        pop = np.arange(30, dtype=np.int32)
        ranked = np.arange(15, dtype=np.int32)
        ranked_emb = compute_ranked_emb(self.E, ranked)
        sizes = {3, 5, 7}

        serial = NullCacheES()
        serial.precompute(self.E, pop, sizes, ranked_emb, ite=20, seed=99, verbose=False)

        parallel = NullCacheES()
        parallel.precompute_parallel(self.E, pop, sizes, ranked_emb,
                                     ite=20, seed=99, verbose=False, n_workers=2)

        for m in sizes:
            s_mu, s_std = serial.cache[m]
            p_mu, p_std = parallel.cache[m]
            self.assertAlmostEqual(s_mu, p_mu, places=4,
                                   msg=f"mu mismatch at size {m}")
            self.assertAlmostEqual(s_std, p_std, places=4,
                                   msg=f"std mismatch at size {m}")

    def test_es_cache_metadata_roundtrip(self):
        pop = np.arange(20, dtype=np.int32)
        ranked = np.arange(10, dtype=np.int32)
        ranked_emb = compute_ranked_emb(self.E, ranked)
        cache = NullCacheES()
        cache.precompute(self.E, pop, {3}, ranked_emb, ite=2, seed=5, verbose=False)

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        try:
            cache.save(path)
            loaded = NullCacheES.load(path)
        finally:
            os.unlink(path)

        expected = NullCacheES.build_metadata(self.E, pop, ranked_emb, ite=2, seed=5)
        self.assertTrue(loaded.metadata_matches(expected)[0])
        wrong = NullCacheES.build_metadata(self.E, pop, ranked_emb, ite=3, seed=5)
        self.assertFalse(loaded.metadata_matches(wrong)[0])


if __name__ == "__main__":
    unittest.main()
