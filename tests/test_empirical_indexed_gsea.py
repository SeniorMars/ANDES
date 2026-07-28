import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from andes import enrich as andes_gsea
from andes import index as andes_index
from andes import expression
from andes import bma
from andes.ranked import (
    compute_es_score,
    compute_ranked_emb,
    score_terms_indexed,
)


class VectorizedExpressionRankingTests(unittest.TestCase):
    def test_batched_t_statistics_match_statsmodels_ols(self):
        rng = np.random.default_rng(481)
        values = rng.normal(size=(31, 10))
        base = np.array([0] * 4 + [1] * 6, dtype=np.float64)
        conditions = np.column_stack([rng.permutation(base) for _ in range(7)])

        observed = expression.binary_ols_t_statistics(values, conditions)
        expected = np.empty_like(observed)
        for permutation in range(conditions.shape[1]):
            design = sm.add_constant(conditions[:, permutation])
            for gene in range(values.shape[0]):
                expected[gene, permutation] = (
                    sm.OLS(values[gene], design).fit().tvalues[1]
                )

        np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-12)

    def test_degenerate_rows_and_stable_ties_are_explicit(self):
        condition = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        values = np.array(
            [
                [3, 3, 3, 3, 3, 3],
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 3, 2, 1],
            ],
            dtype=np.float64,
        )
        statistics = expression.binary_ols_t_statistics(values, condition)
        self.assertEqual(statistics[0], 0.0)
        self.assertTrue(np.isposinf(statistics[1]))
        self.assertEqual(statistics[2], 0.0)
        np.testing.assert_array_equal(
            expression.stable_rank_orders(np.array([2.0, 2.0, 1.0, 2.0])),
            np.array([0, 1, 3, 2]),
        )

    def test_invalid_design_and_nonfinite_expression_are_rejected(self):
        values = np.ones((3, 4), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "exactly two groups"):
            expression.encode_two_group_condition([0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            expression.encode_two_group_condition([0, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "binary"):
            expression.binary_ols_t_statistics(
                values, np.array([0, 0, 1, 2], dtype=np.float64)
            )
        values[1, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            expression.binary_ols_t_statistics(
                values, np.array([0, 0, 1, 1], dtype=np.float64)
            )

        table = pd.DataFrame(
            np.ones((2, 5), dtype=np.float64),
            index=["g1", "g2"],
        )
        with self.assertRaisesRegex(ValueError, "sample_columns is required"):
            expression.prepare_expression_data(
                table,
                [0, 0, 1, 1],
            )

    def test_shuffle_uses_generator_without_global_rng_mutation(self):
        condition = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        np.random.seed(991)
        state_before = np.random.get_state()

        observed = expression.permuted_condition_matrix(condition, [7, 8, 9])

        state_after = np.random.get_state()
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])

        expected = np.random.default_rng(7).permutation(condition)
        np.testing.assert_array_equal(observed[:, 0], expected)
        np.testing.assert_array_equal(observed.sum(axis=0), np.array([3.0, 3.0, 3.0]))

    def test_permutation_batching_does_not_change_rank_orders(self):
        rng = np.random.default_rng(812)
        values = rng.normal(size=(13, 8))
        condition = np.array([0, 0, 0, 1, 1, 1, 1, 1])

        batches = list(
            expression.iter_label_shuffled_rank_orders(
                values,
                condition,
                7,
                seed=31,
                batch_size=3,
            )
        )
        observed = np.column_stack([orders for _, orders in batches])

        expected_columns = []
        for i in range(7):
            shuffled = expression.permuted_condition_matrix(condition, [31 + i])[:, 0]
            stats = expression.binary_ols_t_statistics(values, shuffled)
            expected_columns.append(expression.stable_rank_orders(stats))
        expected = np.column_stack(expected_columns)
        np.testing.assert_array_equal(observed, expected)

    def test_expression_loader_precomputes_embedding_row_mapping(self):
        condition = [0, 0, 1, 1]
        data = pd.DataFrame(
            [
                [0.0, 0.1, 2.0, 2.1, 10.0, 0.1, 4.0],
                [3.0, 3.2, 1.0, 1.1, 20.0, 0.2, 5.0],
                [1.0, 1.1, 1.0, 1.1, 30.0, 0.3, 6.0],
            ],
            index=["known_a", "unknown", "known_b"],
        )
        with self.subTest("context"):
            # Exercise the file-independent core used by the CLI loader.
            values, genes, encoded = expression.prepare_expression_data(
                data,
                condition,
                sample_columns=data.columns[:4],
            )
            mapping = np.fromiter(
                ({"known_a": 4, "known_b": 2}.get(g, -1) for g in genes),
                dtype=np.int32,
            )
            known = mapping >= 0
            order = expression.stable_rank_orders(
                expression.binary_ols_t_statistics(values[known], encoded)
            )
            np.testing.assert_array_equal(
                mapping[known][order], np.array([4, 2], dtype=np.int32)
            )


class IndexedRankedScoringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(923)
        raw = rng.normal(size=(29, 9)).astype(np.float32)
        self.E = np.ascontiguousarray(bma.l2_normalize_rows(raw), dtype=np.float32)
        self.terms = ["a", "b", "c", "d"]
        self.indices = {
            "a": np.array([0, 3, 8], dtype=np.int32),
            "b": np.array([1, 5, 7, 11], dtype=np.int32),
            "c": np.array([2, 4, 6, 9, 12], dtype=np.int32),
            "d": np.array([10, 13], dtype=np.int32),
        }
        self.sizes = np.array(
            [len(self.indices[t]) for t in self.terms], dtype=np.int32
        )
        self.bestmatch = np.column_stack(
            [(self.E @ self.E[self.indices[t]].T).max(axis=1) for t in self.terms]
        ).astype(np.float32)
        self.ranked_idx = np.array(
            [15, 2, 21, 4, 8, 1, 18, 11, 7, 6, 20], dtype=np.int32
        )

    def test_indexed_scores_match_direct_and_bestmatch_reference(self):
        cache = SimpleNamespace(
            cache={
                2: (0.25, 1.5),
                3: (-0.5, 2.0),
                4: (0.1, 0.75),
                5: (0.0, 1.0),
            }
        )
        observed, zscores, stats = score_terms_indexed(
            self.bestmatch,
            self.ranked_idx,
            self.sizes,
            cache,
            max_workspace_mb=0.00002,
        )

        ranked_emb = compute_ranked_emb(self.E, self.ranked_idx)
        expected = np.array(
            [compute_es_score(self.E, self.indices[t], ranked_emb) for t in self.terms],
            dtype=np.float32,
        )
        expected_z = np.array(
            [
                (score - cache.cache[int(size)][0]) / cache.cache[int(size)][1]
                for score, size in zip(expected, self.sizes)
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(zscores, expected_z, rtol=2e-6, atol=2e-6)
        self.assertLess(stats["term_chunk_size"], len(self.terms))

    def test_indexed_scoring_preserves_first_maximum_and_validates_inputs(self):
        bestmatch = np.array(
            [
                [2.0, 1.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scores, _, _ = score_terms_indexed(
            bestmatch,
            np.arange(4, dtype=np.int32),
            np.array([1, 1], dtype=np.int32),
        )
        # First column trace is [1, 0, 1, 0], so the first |max| is retained.
        np.testing.assert_array_equal(scores, np.array([1.0, 0.0]))

        with self.assertRaisesRegex(IndexError, "out-of-range"):
            score_terms_indexed(
                bestmatch,
                np.array([0, 5], dtype=np.int32),
                np.array([1, 1], dtype=np.int32),
            )
        with self.assertRaisesRegex(TypeError, "integer embedding rows"):
            score_terms_indexed(
                bestmatch,
                np.array([0.0, 1.0]),
                np.array([1, 1], dtype=np.int32),
            )
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            score_terms_indexed(
                bestmatch,
                np.array([0, 2**32], dtype=np.uint64),
                np.array([1, 1], dtype=np.int32),
            )
        with self.assertRaisesRegex(KeyError, "sizes not in null cache"):
            score_terms_indexed(
                bestmatch,
                np.arange(4, dtype=np.int32),
                np.array([1, 2], dtype=np.int32),
                {1: (0.0, 1.0)},
            )

    def test_incremental_empirical_counts_match_brute_force(self):
        rng = np.random.default_rng(122)
        values = rng.normal(size=(8, 6))
        condition = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        row_to_embedding = np.array([14, 2, 18, 5, 1, 20, 11, 7], dtype=np.int32)
        observed_order = expression.stable_rank_orders(
            expression.binary_ols_t_statistics(values, condition)
        )
        observed, _, _ = score_terms_indexed(
            self.bestmatch,
            row_to_embedding[observed_order],
            self.sizes,
        )

        counts = andes_gsea.empirical_exceedance_counts(
            self.bestmatch,
            self.sizes,
            observed,
            values,
            condition,
            row_to_embedding,
            n_permutations=7,
            seed=19,
            permutation_batch_size=3,
            max_workspace_mb=0.00002,
        )

        expected = np.zeros(len(self.terms), dtype=np.int64)
        for i in range(7):
            shuffled = expression.permuted_condition_matrix(condition, [19 + i])[:, 0]
            order = expression.stable_rank_orders(
                expression.binary_ols_t_statistics(values, shuffled)
            )
            scores, _, _ = score_terms_indexed(
                self.bestmatch,
                row_to_embedding[order],
                self.sizes,
            )
            expected += np.abs(scores) >= np.abs(observed)
        np.testing.assert_array_equal(counts, expected)

    def test_external_index_compatibility_checks_gene_order_and_embedding(self):
        index = SimpleNamespace(
            gene_list=["g0", "g1", "g2"],
            E_unit=self.E[:3],
            terms=["a"],
            term_indices={"a": np.array([0, 2], dtype=np.int32)},
            background=np.array([0, 1, 2], dtype=np.int32),
            metadata={"embedding_hash": andes_index._hash_array(self.E[:3])},
        )
        andes_gsea.validate_index_compatibility(
            index,
            E_unit=self.E[:3],
            gene_list=["g0", "g1", "g2"],
            terms=["a"],
            term_indices={"a": np.array([2, 0], dtype=np.int32)},
            background=np.array([2, 1, 0], dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "gene order"):
            andes_gsea.validate_index_compatibility(index, gene_list=["g1", "g0", "g2"])
        changed = self.E[:3].copy()
        changed[0, 0] += np.float32(0.01)
        with self.assertRaisesRegex(ValueError, "embedding"):
            andes_gsea.validate_index_compatibility(index, E_unit=changed)


if __name__ == "__main__":
    unittest.main()
