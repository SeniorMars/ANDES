import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray

from andes import data, expression
from andes import enrich as andes_gsea
from andes import index as andes_index
from andes.ranked import score_terms_indexed
from tests.reference import reference_ranked_es


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
        state_before = cast(
            tuple[str, NDArray[np.uint32], int, int, float],
            np.random.get_state(),
        )

        observed = expression.permuted_condition_matrix(condition, [7, 8, 9])

        state_after = cast(
            tuple[str, NDArray[np.uint32], int, int, float],
            np.random.get_state(),
        )
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
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "expression.tsv"
            path.write_text(
                "0\t0\t1\t1\n"
                "gene\ts1\ts2\ts3\ts4\tmetadata\n"
                "known_a\t0.0\t0.1\t2.0\t2.1\t10\n"
                "unknown\t3.0\t3.2\t1.0\t1.1\t20\n"
                "known_b\t1.0\t1.1\t1.0\t1.1\t30\n",
                encoding="utf-8",
            )
            ranked_indices, context = andes_gsea.ranked_list_from_expression(
                path,
                {"known_a": 4, "known_b": 2},
            )

        np.testing.assert_array_equal(
            ranked_indices,
            np.array([4, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            context["row_to_embedding"],
            np.array([4, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(context["condition"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(context["values"].shape, (2, 4))


class IndexedRankedScoringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(923)
        raw = rng.normal(size=(29, 9)).astype(np.float32)
        self.E = data.EmbeddingSpace.from_arrays(
            raw,
            [f"g{i}" for i in range(raw.shape[0])],
        ).vectors
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
        observed, stats = score_terms_indexed(
            self.bestmatch,
            self.ranked_idx,
            max_workspace_mb=0.00002,
        )

        similarity = self.E @ self.E.T
        expected = np.empty(len(self.terms), dtype=np.float32)
        for i, term in enumerate(self.terms):
            expected[i] = reference_ranked_es(
                similarity,
                self.indices[term],
                self.ranked_idx,
            )
        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)
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
        scores, _ = score_terms_indexed(
            bestmatch,
            np.arange(4, dtype=np.int32),
        )
        # First column trace is [1, 0, 1, 0], so the first |max| is retained.
        np.testing.assert_array_equal(scores, np.array([1.0, 0.0]))

        with self.assertRaisesRegex(IndexError, "out-of-range"):
            score_terms_indexed(
                bestmatch,
                np.array([0, 5], dtype=np.int32),
            )
        with self.assertRaisesRegex(TypeError, "integer embedding rows"):
            score_terms_indexed(
                bestmatch,
                np.array([0.0, 1.0]),
            )
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            score_terms_indexed(
                bestmatch,
                np.array([0, 2**32], dtype=np.uint64),
            )

    def test_indexed_scoring_has_an_explicit_first_near_tie_policy(self):
        within_tolerance = np.float32(1.0e-6)
        beyond_tolerance = np.float32(4.0e-6)
        bestmatch = np.array(
            [
                [1.0, 1.0, 1.0],
                [
                    -2.0,
                    -2.0 - within_tolerance,
                    -2.0 - beyond_tolerance,
                ],
                [
                    1.0,
                    1.0 + within_tolerance,
                    1.0 + beyond_tolerance,
                ],
            ],
            dtype=np.float32,
        )

        scores, _ = score_terms_indexed(
            bestmatch,
            np.arange(3, dtype=np.int32),
        )

        self.assertGreater(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)
        self.assertLess(scores[2], 0.0)
        np.testing.assert_allclose(
            np.abs(scores),
            np.array([1.0, 1.0, 1.0 + beyond_tolerance]),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_incremental_empirical_counts_match_brute_force(self):
        rng = np.random.default_rng(122)
        values = rng.normal(size=(8, 6))
        condition = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        row_to_embedding = np.array([14, 2, 18, 5, 1, 20, 11, 7], dtype=np.int32)
        observed_order = expression.stable_rank_orders(
            expression.binary_ols_t_statistics(values, condition)
        )
        observed, _ = score_terms_indexed(
            self.bestmatch,
            row_to_embedding[observed_order],
        )

        result = andes_gsea.empirical_exceedance_counts(
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
            scores, _ = score_terms_indexed(
                self.bestmatch,
                row_to_embedding[order],
            )
            expected += np.abs(scores) >= np.abs(observed)
        np.testing.assert_array_equal(result.counts, expected)
        self.assertGreater(result.stats.indexed_workspace_bytes, 0)
        self.assertEqual(result.stats.batches, 3)

    def test_empirical_counting_rejects_nonpermutation_orders(self):
        row_to_embedding = np.array([0, 1, 2, 3], dtype=np.int32)
        orders = np.column_stack(
            [
                np.arange(4, dtype=np.int64),
                np.arange(3, -1, -1, dtype=np.int64),
            ]
        )
        observed = np.zeros(len(self.terms), dtype=np.float64)

        counts, _ = andes_gsea.count_indexed_ranked_exceedances(
            self.bestmatch,
            row_to_embedding,
            orders,
            observed,
            max_workspace_mb=0.00001,
        )

        np.testing.assert_array_equal(
            counts,
            np.full(len(self.terms), orders.shape[1], dtype=np.int64),
        )

        duplicate = orders.copy()
        duplicate[-1, 0] = duplicate[0, 0]
        with self.assertRaisesRegex(ValueError, "must be a permutation"):
            andes_gsea.count_indexed_ranked_exceedances(
                self.bestmatch,
                row_to_embedding,
                duplicate,
                observed,
            )

    def test_monte_carlo_pvalues_include_the_observed_labeling(self):
        counts = np.array([0, 3, 10], dtype=np.int64)
        original = counts.copy()

        observed = andes_gsea.monte_carlo_pvalues(counts, 10)

        np.testing.assert_allclose(observed, np.array([1.0, 4.0, 11.0]) / 11.0)
        np.testing.assert_array_equal(counts, original)
        self.assertEqual(observed.dtype, np.float64)

        for invalid in (
            np.array([-1], dtype=np.int64),
            np.array([11], dtype=np.int64),
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "between zero"),
            ):
                andes_gsea.monte_carlo_pvalues(invalid, 10)

        with self.assertRaisesRegex(TypeError, "contain integers"):
            andes_gsea.monte_carlo_pvalues(np.array([0.0]), 10)
        with self.assertRaisesRegex(ValueError, "positive"):
            andes_gsea.monte_carlo_pvalues(np.array([0]), 0)
        for invalid_n in (10.0, True):
            with (
                self.subTest(invalid_n=invalid_n),
                self.assertRaisesRegex(TypeError, "must be an integer"),
            ):
                andes_gsea.monte_carlo_pvalues(np.array([0]), invalid_n)

    def test_external_index_compatibility_requires_exact_typed_identity(self):
        embedding = data.EmbeddingSpace.from_arrays(
            self.E[:3],
            ["g0", "g1", "g2"],
            normalize=False,
        )
        database = data.GeneSetDatabase.from_index_mapping(
            {"a": np.array([2, 0], dtype=np.int32)},
            n_genes=3,
            embedding_fingerprint=embedding.fingerprint,
            background=np.array([2, 1, 0], dtype=np.int32),
            background_policy="explicit_test_background",
        )

        with tempfile.TemporaryDirectory() as root:
            index_path = Path(root) / "index"
            andes_index.build_andes_index(
                embedding,
                database,
                index_path,
                max_workspace_mb=0.001,
            )
            index = andes_index.load_andes_index(index_path, mmap=True)
            andes_gsea.validate_index_compatibility(
                index,
                E_unit=embedding.vectors,
                gene_list=embedding.genes,
                database=database,
            )

            with self.assertRaisesRegex(ValueError, "gene order"):
                andes_gsea.validate_index_compatibility(
                    index,
                    gene_list=["g1", "g0", "g2"],
                )

            changed = embedding.vectors.copy()
            changed[0, 0] += np.float32(0.01)
            with self.assertRaisesRegex(ValueError, "embedding"):
                andes_gsea.validate_index_compatibility(index, E_unit=changed)

            different_embedding = data.EmbeddingSpace.from_arrays(
                embedding.vectors,
                ["g1", "g0", "g2"],
                normalize=False,
            )
            incompatible_database = data.GeneSetDatabase.from_index_mapping(
                {"a": np.array([2, 0], dtype=np.int32)},
                n_genes=3,
                embedding_fingerprint=different_embedding.fingerprint,
                background=np.array([2, 1, 0], dtype=np.int32),
                background_policy="explicit_test_background",
            )
            with self.assertRaisesRegex(ValueError, "embedding identity"):
                andes_gsea.validate_index_compatibility(
                    index,
                    database=incompatible_database,
                )
