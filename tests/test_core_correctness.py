import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import func_optimized as bma
import andes_index
import precompute_cache
from func_gsea import (
    NullCacheESBetter as NullCacheES,
    compute_es_score,
    compute_es_trace,
    compute_ranked_emb,
    score_terms_batched,
    score_terms_bestmatch,
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

    def test_bma_rebuild_clears_stale_entries_on_metadata_mismatch(self):
        E = np.eye(8, dtype=np.float32)
        pop = np.arange(8, dtype=np.int32)
        cache = bma.NullCacheBMA()
        cache.metadata = bma.NullCacheBMA.build_metadata(E, pop, pop, ite=2, seed=1)
        cache.cache[(6, 6)] = (123.0, 456.0)

        cache.precompute(E, pop, {(2, 3)}, ite=2, seed=2, verbose=False)

        self.assertNotIn((6, 6), cache.cache)
        self.assertEqual(set(cache.cache), {(2, 3)})
        expected = bma.NullCacheBMA.build_metadata(E, pop, pop, ite=2, seed=2)
        self.assertTrue(cache.metadata_matches(expected)[0])

    def test_bma_serial_precompute_matches_parallel(self):
        pop = np.arange(24, dtype=np.int32)
        size_pairs = {(2, 3), (4, 2), (5, 5)}

        serial = bma.NullCacheBMA()
        serial.precompute(self.E, pop, size_pairs, ite=8, seed=77, verbose=False)

        parallel = bma.NullCacheBMA()
        parallel.precompute_parallel(
            self.E, pop, size_pairs, ite=8, seed=77, verbose=False, n_workers=2
        )

        for pair in size_pairs:
            np.testing.assert_allclose(
                parallel.cache[pair], serial.cache[pair], rtol=1e-6, atol=1e-6
            )

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

        chunked, _, _ = bma.score_bma_zscore_matrix_batched(
            terms,
            terms,
            cache,
            blocks,
            blocks,
            symmetric=True,
            n_workers=1,
            max_workspace_mb=0.000001,
        )
        np.testing.assert_allclose(chunked, pairwise, rtol=1e-5, atol=1e-5)

    def test_vectorized_zscore_lookup_matches_scalar_lookup(self):
        true_scores = np.array(
            [
                [0.20, 0.40, 0.60],
                [0.10, 0.30, 0.50],
            ],
            dtype=np.float32,
        )
        sizes1 = np.array([2, 3], dtype=np.int32)
        sizes2 = np.array([4, 5, 4], dtype=np.int32)
        cache = {
            (2, 4): (0.10, 0.05),
            (2, 5): (0.20, 0.10),
            (3, 4): (0.10, 0.00),
            (3, 5): (0.25, 0.05),
        }

        expected = np.empty_like(true_scores)
        for i, m in enumerate(sizes1):
            for j, k in enumerate(sizes2):
                expected[i, j] = bma._zscore_from_cache(
                    cache, float(true_scores[i, j]), int(m), int(k)
                )

        observed = bma.zscore_matrix_from_cache(
            true_scores, sizes1, sizes2, cache
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

        with self.assertRaises(KeyError):
            bma.zscore_matrix_from_cache(
                true_scores, sizes1, sizes2, {(2, 4): (0.0, 1.0)}
            )

    def test_bestmatch_matrix_scoring_matches_pairwise(self):
        terms1 = ["a", "b"]
        terms2 = ["x", "y", "z"]
        indices1 = {
            "a": np.array([0, 1, 2], dtype=np.int32),
            "b": np.array([3, 4, 5, 6], dtype=np.int32),
        }
        indices2 = {
            "x": np.array([7, 8, 9], dtype=np.int32),
            "y": np.array([10, 11, 12, 13], dtype=np.int32),
            "z": np.array([14, 15, 16], dtype=np.int32),
        }
        blocks1 = bma.precompute_term_embedding_blocks(self.E, indices1)
        blocks2 = bma.precompute_term_embedding_blocks(self.E, indices2)
        cache = bma.NullCacheBMA()
        for m in {len(v) for v in indices1.values()}:
            for k in {len(v) for v in indices2.values()}:
                cache.cache[(m, k)] = (0.0, 1.0)

        pairwise = bma.score_bma_zscore_matrix(
            self.E,
            terms1,
            terms2,
            indices1,
            indices2,
            cache,
            blocks1,
            blocks2,
            symmetric=False,
            n_workers=1,
            numba_threshold=0,
        )
        bestmatch, stats = bma.score_bma_zscore_matrix_bestmatch(
            self.E,
            terms1,
            terms2,
            indices1,
            indices2,
            cache,
            blocks1,
            blocks2,
            max_workspace_mb=0.00001,
        )

        np.testing.assert_allclose(bestmatch, pairwise, rtol=1e-5, atol=1e-5)
        self.assertGreaterEqual(stats["workspace_mb"], 0.0)

    def test_symmetric_bestmatch_reuse_matches_two_direction_path(self):
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

        two_direction, two_stats = bma.score_bma_zscore_matrix_bestmatch(
            self.E,
            terms,
            terms,
            indices,
            indices,
            cache,
            blocks,
            blocks,
            symmetric=False,
            max_workspace_mb=0.00001,
        )
        symmetric, sym_stats = bma.score_bma_zscore_matrix_bestmatch(
            self.E,
            terms,
            terms,
            indices,
            indices,
            cache,
            blocks,
            blocks,
            symmetric=True,
            max_workspace_mb=0.00001,
        )

        np.testing.assert_allclose(symmetric, two_direction, rtol=1e-5, atol=1e-5)
        self.assertFalse(two_stats["symmetric_reuse"])
        self.assertTrue(sym_stats["symmetric_reuse"])

    def test_prefix_coupled_null_matches_reference_permutations(self):
        E = self.E[:10]
        pop = np.arange(10, dtype=np.int32)
        size_pairs = {(2, 3), (4, 2)}
        ite = 5
        seed = 101

        cache = bma.NullCacheBMA()
        cache.precompute_prefix(E, pop, size_pairs, ite=ite, seed=seed, verbose=False)

        rng = np.random.default_rng(seed)
        max_m = 4
        max_k = 3
        scores = {pair: [] for pair in size_pairs}
        for _ in range(ite):
            X = E[pop[rng.permutation(len(pop))[:max_m]]]
            Y = E[pop[rng.permutation(len(pop))[:max_k]]]
            A = X @ Y.T
            row_best = np.maximum.accumulate(A, axis=1)
            row_sum = np.cumsum(row_best, axis=0)
            col_best = np.maximum.accumulate(A, axis=0)
            col_sum = np.cumsum(col_best, axis=1)
            for m, k in size_pairs:
                scores[(m, k)].append(
                    (row_sum[m - 1, k - 1] + col_sum[m - 1, k - 1]) / (m + k)
                )

        for pair, values in scores.items():
            values = np.asarray(values, dtype=np.float64)
            expected = (float(values.mean()), float(values.std(ddof=1)))
            np.testing.assert_allclose(
                cache.cache[pair], expected, rtol=1e-6, atol=1e-6
            )

        default_metadata = bma.NullCacheBMA.build_metadata(
            E, pop, pop, ite=ite, seed=seed
        )
        self.assertFalse(cache.metadata_matches(default_metadata)[0])

    def test_prefix_cache_roundtrip_can_score_without_rebuild(self):
        terms1 = ["a", "b"]
        terms2 = ["x", "y"]
        indices1 = {
            "a": np.array([0, 1, 2], dtype=np.int32),
            "b": np.array([3, 4, 5, 6], dtype=np.int32),
        }
        indices2 = {
            "x": np.array([7, 8, 9], dtype=np.int32),
            "y": np.array([10, 11, 12, 13], dtype=np.int32),
        }
        pop = np.arange(self.E.shape[0], dtype=np.int32)
        size_pairs = {
            (len(indices1[t1]), len(indices2[t2]))
            for t1 in terms1
            for t2 in terms2
        }

        cache = bma.NullCacheBMA()
        cache.precompute_prefix(
            self.E, pop, size_pairs, ite=12, seed=202, verbose=False
        )
        self.assertEqual(set(cache.cache), size_pairs)

        prefix_metadata = bma.NullCacheBMA.build_metadata(
            self.E,
            pop,
            pop,
            ite=12,
            seed=202,
            null_sampling="prefix_coupled",
        )
        self.assertTrue(cache.metadata_matches(prefix_metadata)[0])

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        try:
            cache.save(path)
            loaded = bma.NullCacheBMA()
            loaded.load(path)
        finally:
            os.unlink(path)

        self.assertTrue(loaded.metadata_matches(prefix_metadata)[0])
        self.assertEqual(set(loaded.cache), size_pairs)

        blocks1 = bma.precompute_term_embedding_blocks(self.E, indices1)
        blocks2 = bma.precompute_term_embedding_blocks(self.E, indices2)
        pairwise = bma.score_bma_zscore_matrix(
            self.E,
            terms1,
            terms2,
            indices1,
            indices2,
            loaded,
            blocks1,
            blocks2,
            symmetric=False,
            n_workers=1,
            numba_threshold=0,
        )
        bestmatch, _ = bma.score_bma_zscore_matrix_bestmatch(
            self.E,
            terms1,
            terms2,
            indices1,
            indices2,
            loaded,
            blocks1,
            blocks2,
            max_workspace_mb=0.00001,
        )

        self.assertTrue(np.all(np.isfinite(pairwise)))
        np.testing.assert_allclose(bestmatch, pairwise, rtol=1e-5, atol=1e-5)

    def test_indexed_query_matches_pairwise_scoring(self):
        gene_list = [f"g{i}" for i in range(self.E.shape[0])]
        terms = ["x", "y", "z"]
        target_indices = {
            "x": np.array([7, 8, 9], dtype=np.int32),
            "y": np.array([10, 11, 12, 13], dtype=np.int32),
            "z": np.array([14, 15, 16], dtype=np.int32),
        }
        query_indices = {"query": np.array([0, 1, 2, 3], dtype=np.int32)}
        background = np.arange(self.E.shape[0], dtype=np.int32)
        cache = bma.NullCacheBMA()
        for term in terms:
            cache.cache[(4, len(target_indices[term]))] = (0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            andes_index.build_andes_index(
                self.E,
                gene_list,
                terms,
                target_indices,
                background,
                tmp,
                max_workspace_mb=0.00001,
            )
            index = andes_index.load_andes_index(tmp)
            true_scores, zscores = index.score_query(
                query_indices["query"], null_cache=cache
            )

        blocks_query = bma.precompute_term_embedding_blocks(self.E, query_indices)
        blocks_target = bma.precompute_term_embedding_blocks(self.E, target_indices)
        expected = bma.score_bma_zscore_matrix(
            self.E,
            ["query"],
            terms,
            query_indices,
            target_indices,
            cache,
            blocks_query,
            blocks_target,
            symmetric=False,
            n_workers=1,
            numba_threshold=0,
        )[0]

        np.testing.assert_allclose(zscores, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(true_scores, expected, rtol=1e-5, atol=1e-5)

    def test_index_maps_query_genes_and_writes_top_k(self):
        gene_list = [f"g{i}" for i in range(self.E.shape[0])]
        terms = ["x", "y"]
        target_indices = {
            "x": np.array([4, 5, 6], dtype=np.int32),
            "y": np.array([7, 8, 9], dtype=np.int32),
        }
        background = np.arange(self.E.shape[0], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp:
            andes_index.build_andes_index(
                self.E,
                gene_list,
                terms,
                target_indices,
                background,
                tmp,
                max_workspace_mb=0.00001,
            )
            index = andes_index.load_andes_index(tmp, mmap=True)
            query_idx, missing = index.map_genes(["g0", "g1", "absent", "g0"])
            self.assertEqual(missing, ["absent"])
            np.testing.assert_array_equal(query_idx, np.array([0, 1], dtype=np.int32))

            true_scores, _ = index.score_query(query_idx)
            out = os.path.join(tmp, "query.csv")
            df = andes_index.write_query_results(
                out, index, true_scores, zscores=None, top_k=1
            )
            self.assertEqual(len(df), 1)
            self.assertTrue(os.path.exists(out))

    def test_indexed_query_cache_builds_needed_size_pairs(self):
        gene_list = [f"g{i}" for i in range(self.E.shape[0])]
        terms = ["x", "y"]
        target_indices = {
            "x": np.array([4, 5, 6], dtype=np.int32),
            "y": np.array([7, 8, 9, 10], dtype=np.int32),
        }
        background = np.arange(self.E.shape[0], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp:
            andes_index.build_andes_index(
                self.E,
                gene_list,
                terms,
                target_indices,
                background,
                tmp,
                max_workspace_mb=0.00001,
            )
            index = andes_index.load_andes_index(tmp)
            cache_path = os.path.join(tmp, "query_null.pkl")
            cache = andes_index.load_or_build_query_cache(
                index,
                query_size=2,
                cache_path=cache_path,
                ite=4,
                seed=303,
                null_mode="prefix",
                verbose=False,
            )

            self.assertEqual(set(cache.cache), {(2, 3), (2, 4)})
            self.assertTrue(os.path.exists(cache_path))
            true_scores, zscores = index.score_query(
                np.array([0, 1], dtype=np.int32), null_cache=cache
            )
            self.assertTrue(np.all(np.isfinite(true_scores)))
            self.assertTrue(np.all(np.isfinite(zscores)))

    def test_indexed_zscore_rejects_query_outside_background(self):
        gene_list = [f"g{i}" for i in range(self.E.shape[0])]
        terms = ["x"]
        target_indices = {"x": np.array([7, 8, 9], dtype=np.int32)}
        background = np.arange(1, self.E.shape[0], dtype=np.int32)
        cache = bma.NullCacheBMA()
        cache.cache[(2, 3)] = (0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            andes_index.build_andes_index(
                self.E,
                gene_list,
                terms,
                target_indices,
                background,
                tmp,
                max_workspace_mb=0.00001,
            )
            index = andes_index.load_andes_index(tmp)

            true_scores, zscores = index.score_query(np.array([0, 1], dtype=np.int32))
            self.assertIsNone(zscores)
            self.assertTrue(np.all(np.isfinite(true_scores)))
            with self.assertRaisesRegex(ValueError, "outside-background genes"):
                index.score_query(np.array([0, 1], dtype=np.int32), null_cache=cache)

    def test_precompute_bma_rebuilds_stale_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            E = self.E[:8]
            gene_list = [f"g{i}" for i in range(E.shape[0])]
            emb_path = os.path.join(tmp, "emb.csv")
            genes_path = os.path.join(tmp, "genes.txt")
            gmt_path = os.path.join(tmp, "sets.gmt")
            np.savetxt(emb_path, E, delimiter=",")
            with open(genes_path, "w") as fh:
                fh.write("\n".join(gene_list) + "\n")
            with open(gmt_path, "w") as fh:
                fh.write("t1\tname\tg0\tg1\n")
                fh.write("t2\tname\tg2\tg3\tg4\n")

            previous_root = precompute_cache.CACHE_ROOT
            precompute_cache.CACHE_ROOT = precompute_cache.Path(tmp) / "cache"
            try:
                args = SimpleNamespace(
                    emb=emb_path,
                    genelist=genes_path,
                    gmt=[gmt_path],
                    min=1,
                    max=10,
                    ite=2,
                    seed=11,
                    workers=1,
                    chunk_size=0,
                    rebuild=False,
                )
                E_unit, genes = precompute_cache.load_embedding(emb_path, genes_path)
                raw, _, _ = precompute_cache.load_gmt_to_indices(
                    [gmt_path], genes, args.min, args.max
                )
                pop = precompute_cache.background_pop(raw, genes, set(genes))
                eid = precompute_cache.emb_id(E_unit, genes)
                pid = precompute_cache.pop_id(pop)
                cache_dir = os.path.join(precompute_cache.CACHE_ROOT, "bma")
                os.makedirs(cache_dir)
                cache_path = os.path.join(
                    cache_dir, f"{eid[:16]}__{pid[:16]}__ite{args.ite}__seed{args.seed}.pkl"
                )
                stale = bma.NullCacheBMA()
                stale.metadata = bma.NullCacheBMA.build_metadata(
                    E_unit, pop, pop, ite=args.ite, seed=999
                )
                for pair in {(2, 2), (2, 3), (3, 2), (3, 3)}:
                    stale.cache[pair] = (123.0, 456.0)
                stale.save(cache_path)

                precompute_cache.cmd_bma(args)

                loaded = bma.NullCacheBMA()
                loaded.load(cache_path)
                expected, _ = precompute_cache._bma_metadata(
                    E_unit, pop, args.ite, args.seed
                )
                self.assertTrue(loaded.metadata_matches(expected)[0])
                self.assertNotEqual(loaded.cache[(2, 2)], (123.0, 456.0))
            finally:
                precompute_cache.CACHE_ROOT = previous_root


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

    def test_ranked_bestmatch_scoring_matches_batched(self):
        terms = ["a", "b", "c"]
        indices = {
            "a": np.array([0, 1, 2], dtype=np.int32),
            "b": np.array([3, 4, 5, 6], dtype=np.int32),
            "c": np.array([7, 8, 9], dtype=np.int32),
        }
        ranked = np.arange(18, dtype=np.int32)
        ranked_emb = compute_ranked_emb(self.E, ranked)
        ranked_emb_T = np.ascontiguousarray(ranked_emb.T, dtype=np.float32)
        cache = NullCacheES()
        for m in {len(v) for v in indices.values()}:
            cache.cache[m] = (0.0, 1.0)

        true_batched, z_batched = score_terms_batched(
            self.E,
            indices,
            terms,
            ranked_emb_T,
            cache,
            batch_bytes=1024 * 1024,
        )
        true_best, z_best, stats = score_terms_bestmatch(
            self.E,
            indices,
            terms,
            ranked_emb,
            cache,
            max_workspace_mb=0.00001,
        )

        self.assertGreaterEqual(stats["workspace_mb"], 0.0)
        for term in terms:
            self.assertAlmostEqual(true_best[term], true_batched[term], places=5)
            self.assertAlmostEqual(z_best[term], z_batched[term], places=5)

    def test_ranked_prefix_cache_roundtrip_can_score_bestmatch(self):
        terms = ["a", "b"]
        indices = {
            "a": np.array([0, 1, 2], dtype=np.int32),
            "b": np.array([3, 4, 5, 6], dtype=np.int32),
        }
        pop = np.arange(30, dtype=np.int32)
        ranked = np.arange(15, dtype=np.int32)
        ranked_emb = compute_ranked_emb(self.E, ranked)
        sizes = {len(v) for v in indices.values()}

        cache = NullCacheES()
        cache.precompute(self.E, pop, sizes, ranked_emb, ite=12, seed=202, verbose=False)

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        try:
            cache.save(path)
            loaded = NullCacheES.load(path)
        finally:
            os.unlink(path)

        expected = NullCacheES.build_metadata(
            self.E, pop, ranked_emb, ite=12, seed=202
        )
        self.assertTrue(loaded.metadata_matches(expected)[0])
        self.assertEqual(set(loaded.cache), sizes)

        true_best, z_best, _ = score_terms_bestmatch(
            self.E,
            indices,
            terms,
            ranked_emb,
            loaded,
            max_workspace_mb=0.00001,
        )

        self.assertEqual(set(true_best), set(terms))
        self.assertTrue(np.all(np.isfinite([z_best[t] for t in terms])))


if __name__ == "__main__":
    unittest.main()
