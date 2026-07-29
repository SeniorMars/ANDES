import json
import tempfile
import unittest
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from andes import artifacts, data, index_cli
from andes import index as andes_index
from andes.nulls import (
    BmaNullModel,
    NullSpec,
    null_cache_dir,
    resolve_bma_null,
)


def _bma_reference(E_unit, left, right):
    similarities = E_unit[left] @ E_unit[right].T
    return np.float32(
        (
            similarities.max(axis=1).sum(dtype=np.float32)
            + similarities.max(axis=0).sum(dtype=np.float32)
        )
        / (len(left) + len(right))
    )


def _null_model(index1, index2, values):
    spec = NullSpec(
        kind="bma",
        iterations=2,
        seed=7,
        sampling="prefix_coupled",
        ddof=1,
        embedding_hash=str(index1.metadata["embedding_hash"]),
        population_hashes=(
            artifacts.hash_array(index1.background),
            artifacts.hash_array(index2.background),
        ),
    )
    return BmaNullModel.from_mapping(values, spec)


class AndesIndexConsumerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260728)
        raw = rng.normal(size=(18, 7)).astype(np.float32)
        self.E = data.EmbeddingSpace.from_arrays(
            raw,
            [f"g{i}" for i in range(raw.shape[0])],
        ).vectors
        self.genes = [f"g{i}" for i in range(self.E.shape[0])]
        self.terms1 = ["a2", "a3", "a4"]
        self.indices1 = {
            "a2": np.array([0, 2], dtype=np.int32),
            "a3": np.array([1, 3, 5], dtype=np.int32),
            "a4": np.array([4, 6, 8, 10], dtype=np.int32),
        }
        self.terms2 = ["b3", "b5"]
        self.indices2 = {
            "b3": np.array([7, 9, 11], dtype=np.int32),
            "b5": np.array([2, 6, 12, 14, 16], dtype=np.int32),
        }

    def build(
        self,
        path,
        terms=None,
        indices=None,
        E=None,
        genes=None,
        background=None,
    ):
        E = self.E if E is None else E
        genes = self.genes if genes is None else genes
        terms = self.terms1 if terms is None else terms
        indices = self.indices1 if indices is None else indices
        background = (
            np.arange(E.shape[0], dtype=np.int32) if background is None else background
        )
        embedding = data.EmbeddingSpace.from_arrays(E, genes, normalize=False)
        database = data.GeneSetDatabase.from_index_mapping(
            indices,
            n_genes=len(embedding.genes),
            embedding_fingerprint=embedding.fingerprint,
            terms=terms,
            background=background,
            background_policy="explicit_test_background",
        )
        return andes_index.build_andes_index(
            embedding,
            database,
            path,
            max_workspace_mb=0.0001,
        )

    def test_memmapped_build_is_exact_and_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            indices = {
                "dupes": np.array([5, 1, 5, 3], dtype=np.int64),
                "other": np.array([8, 7], dtype=np.int32),
            }
            metadata = self.build(
                index_dir,
                terms=["dupes", "other"],
                indices=indices,
                background=np.array(
                    [17, 0, 17, 4, 5, 1, 3, 8, 7],
                    dtype=np.int64,
                ),
            )
            index = andes_index.load_andes_index(index_dir, mmap=True)

            self.assertEqual(metadata["index_version"], andes_index.INDEX_VERSION)
            self.assertEqual(
                metadata["background_policy"],
                "explicit_test_background",
            )
            self.assertIsInstance(index.E_unit, np.memmap)
            self.assertIsInstance(index.bestmatch, np.memmap)
            np.testing.assert_array_equal(
                index.members_for("dupes"),
                np.array([1, 3, 5], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                index.background,
                np.array([0, 1, 3, 4, 5, 7, 8, 17], dtype=np.int32),
            )
            expected = np.column_stack(
                [
                    (self.E @ self.E[index.members_for(term)].T).max(axis=1)
                    for term in index.terms
                ]
            ).astype(np.float32)
            np.testing.assert_allclose(index.bestmatch, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(
                index.embedding_space().fingerprint,
                metadata["embedding_fingerprint"],
            )
            self.assertEqual(
                index.gene_set_database().fingerprint,
                metadata["database_fingerprint"],
            )

    def test_loaded_index_cannot_be_mutated_after_validation(self):
        with tempfile.TemporaryDirectory() as root:
            index_dir = Path(root) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=False)

            with self.assertRaises(ValueError):
                index.bestmatch[0, 0] = 0.0
            with self.assertRaises(ValueError):
                index.members_at(0)[0] = 0
            with self.assertRaises(TypeError):
                cast(MutableMapping[str, object], index.metadata)["embedding_hash"] = (
                    "changed"
                )
            with self.assertRaises(TypeError):
                cast(MutableMapping[str, int], index.term_to_index)["new"] = 0

    def test_load_requires_the_published_index_lock(self):
        with tempfile.TemporaryDirectory() as root:
            index_dir = Path(root) / "index"
            self.build(index_dir)
            lock_path = index_dir.with_name(f".{index_dir.name}.lock")
            self.assertTrue(lock_path.is_file())
            lock_path.unlink()

            with self.assertRaisesRegex(
                FileNotFoundError,
                "republish or adopt the artifact",
            ):
                andes_index.load_andes_index(index_dir, mmap=True)

            self.assertFalse(lock_path.exists())

    def test_build_requires_aligned_canonical_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TypeError, "EmbeddingSpace"):
                andes_index.build_andes_index(
                    cast(Any, self.E),
                    cast(Any, object()),
                    Path(tmp) / "raw",
                )

            embedding = data.EmbeddingSpace.from_arrays(
                self.E,
                self.genes,
                normalize=False,
            )
            other_embedding = data.EmbeddingSpace.from_arrays(
                self.E[::-1],
                list(reversed(self.genes)),
                normalize=False,
            )
            database = data.GeneSetDatabase.from_index_mapping(
                self.indices1,
                n_genes=len(self.genes),
                embedding_fingerprint=other_embedding.fingerprint,
                terms=self.terms1,
                background=np.arange(len(self.genes), dtype=np.int32),
                background_policy="explicit_test_background",
            )
            with self.assertRaisesRegex(ValueError, "different embedding"):
                andes_index.build_andes_index(
                    embedding,
                    database,
                    Path(tmp) / "misaligned",
                )

    def test_load_rejects_structural_and_identity_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            missing_dir = Path(root) / "missing"
            self.build(missing_dir)
            (missing_dir / "membership.npz").unlink()
            with self.assertRaisesRegex(
                ValueError, "missing required files.*membership"
            ):
                andes_index.load_andes_index(missing_dir)

            version_dir = Path(root) / "version"
            self.build(version_dir)
            metadata_path = version_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["index_version"] = andes_index.INDEX_VERSION + 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported index_version"):
                andes_index.load_andes_index(version_dir)

            shape_dir = Path(root) / "shape"
            self.build(shape_dir)
            np.save(
                shape_dir / "sizes.npy",
                np.zeros(len(self.terms1) + 1, dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "sizes.npy has shape"):
                andes_index.load_andes_index(shape_dir)

            genes_dir = Path(root) / "genes"
            self.build(genes_dir)
            (genes_dir / "genes.json").write_text(
                json.dumps(list(reversed(self.genes))),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "gene_list_hash mismatch"):
                andes_index.load_andes_index(genes_dir)

            membership_dir = Path(root) / "membership"
            self.build(membership_dir)
            membership = sparse.load_npz(membership_dir / "membership.npz").tocsr()
            membership.indices[0] = 1
            sparse.save_npz(
                membership_dir / "membership.npz",
                membership,
            )
            with self.assertRaisesRegex(ValueError, "members_hash mismatch"):
                andes_index.load_andes_index(membership_dir)

    def test_metadata_load_is_lazy_and_full_audit_rejects_dense_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            index_dir = Path(root) / "index"
            self.build(index_dir)
            bestmatch_path = index_dir / "bestmatch.npy"
            bestmatch = np.load(bestmatch_path, allow_pickle=False)
            bestmatch[0, 0] += np.float32(0.25)
            np.save(bestmatch_path, bestmatch, allow_pickle=False)

            metadata_verified = andes_index.load_andes_index(
                index_dir,
                mmap=True,
            )
            self.assertEqual(metadata_verified.bestmatch.shape, bestmatch.shape)

            with self.assertRaisesRegex(ValueError, "bestmatch_hash mismatch"):
                andes_index.load_andes_index(
                    index_dir,
                    mmap=True,
                    verify="full",
                )

    def test_index_comparison_matches_scalar_reference_and_axis_zscores(self):
        with tempfile.TemporaryDirectory() as root:
            dir1 = Path(root) / "one"
            dir2 = Path(root) / "two"
            self.build(dir1)
            self.build(dir2, terms=self.terms2, indices=self.indices2)
            index1 = andes_index.load_andes_index(dir1, mmap=True)
            index2 = andes_index.load_andes_index(dir2, mmap=True)

            exact = index1.compare(index2)
            true_scores = exact.scores
            expected = np.empty_like(true_scores)
            for i, term1 in enumerate(self.terms1):
                for j, term2 in enumerate(self.terms2):
                    expected[i, j] = _bma_reference(
                        self.E,
                        self.indices1[term1],
                        self.indices2[term2],
                    )
            np.testing.assert_allclose(true_scores, expected, rtol=1e-6, atol=1e-6)

            row_positions = np.asarray([2, 0], dtype=np.int32)
            column_positions = np.asarray([1, 0], dtype=np.int32)
            selected = index1.compare(
                index2,
                row_positions=row_positions,
                column_positions=column_positions,
                max_workspace_mb=0.0001,
            )
            selected_expected = np.empty((2, 2), dtype=np.float32)
            for output_row, term_position in enumerate(row_positions):
                for output_column, other_position in enumerate(column_positions):
                    selected_expected[output_row, output_column] = _bma_reference(
                        self.E,
                        self.indices1[self.terms1[int(term_position)]],
                        self.indices2[self.terms2[int(other_position)]],
                    )
            np.testing.assert_allclose(
                selected.scores,
                selected_expected,
                rtol=1e-6,
                atol=1e-6,
            )
            with self.assertRaisesRegex(ValueError, "duplicate positions"):
                index1.compare(
                    index2,
                    row_positions=[0, 0],
                )

            self_result = index1.compare(index1)
            self_scores = self_result.scores
            self.assertTrue(self_result.stats.symmetric_reuse)
            expected_self = np.empty_like(self_scores)
            for i, term1 in enumerate(self.terms1):
                for j, term2 in enumerate(self.terms1):
                    expected_self[i, j] = _bma_reference(
                        self.E,
                        self.indices1[term1],
                        self.indices1[term2],
                    )
            np.testing.assert_allclose(self_scores, expected_self, rtol=1e-6, atol=1e-6)

            values = {}
            for m in index1.sizes:
                for k in index2.sizes:
                    values[(int(m), int(k))] = (
                        float(10 * int(m) + int(k)),
                        2.0,
                    )
            model = _null_model(index1, index2, values)
            zscores = andes_index.calibrate_index_scores(
                exact,
                index1.sizes,
                index1,
                index2,
                model,
            ).scores
            expected_zscores = np.empty_like(true_scores)
            for i, m in enumerate(index1.sizes):
                for j, k in enumerate(index2.sizes):
                    expected_zscores[i, j] = (
                        true_scores[i, j] - (10 * int(m) + int(k))
                    ) / 2.0
            np.testing.assert_allclose(zscores, expected_zscores, rtol=0, atol=0)

    def test_comparison_rejects_embedding_gene_order_and_null_axis_mismatches(self):
        with tempfile.TemporaryDirectory() as root:
            base_dir = Path(root) / "base"
            embedding_dir = Path(root) / "embedding"
            order_dir = Path(root) / "order"
            axis_dir = Path(root) / "axis"
            self.build(
                base_dir,
                background=np.arange(0, 15, dtype=np.int32),
            )

            changed = self.E.copy()
            changed[0] *= np.float32(-1)
            self.build(embedding_dir, E=changed)
            base = andes_index.load_andes_index(base_dir)
            changed_index = andes_index.load_andes_index(embedding_dir)
            with self.assertRaisesRegex(ValueError, "embedding mismatch"):
                base.compare(changed_index)

            self.build(order_dir, genes=list(reversed(self.genes)))
            reordered = andes_index.load_andes_index(order_dir)
            with self.assertRaisesRegex(ValueError, "gene order mismatch"):
                base.compare(reordered)

            self.build(
                axis_dir,
                terms=self.terms2,
                indices=self.indices2,
                background=np.arange(2, 18, dtype=np.int32),
            )
            axis = andes_index.load_andes_index(axis_dir)
            values = {}
            for m in base.sizes:
                for k in axis.sizes:
                    values[(int(m), int(k))] = (0.0, 1.0)
            wrong_axis_model = _null_model(axis, base, values)
            with self.assertRaisesRegex(ValueError, "row/background-1"):
                andes_index.calibrate_index_scores(
                    base.compare(axis),
                    base.sizes,
                    base,
                    axis,
                    wrong_axis_model,
                )

            with self.assertRaisesRegex(TypeError, "BmaNullModel"):
                andes_index.calibrate_index_scores(
                    base.compare(axis),
                    base.sizes,
                    base,
                    axis,
                    cast(Any, values),
                )

    def test_batch_queries_match_scalar_reference_under_tiny_workspace(self):
        queries = [
            np.array([0, 1], dtype=np.int32),
            np.array([2, 4, 6], dtype=np.int32),
            np.array([3, 5, 7, 9], dtype=np.int32),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            batch_result = index.score_queries(
                queries,
                max_workspace_mb=0.00001,
            )
            batch_true = batch_result.scores
            expected = np.asarray(
                [
                    [
                        _bma_reference(
                            self.E,
                            query,
                            self.indices1[term],
                        )
                        for term in self.terms1
                    ]
                    for query in queries
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(batch_true, expected, rtol=1e-6, atol=1e-6)
            for workspace_mb in (0.00001, 128):
                one_query_batch = index.score_queries(
                    [queries[0]],
                    max_workspace_mb=workspace_mb,
                )
                single_query = index.score_query(
                    queries[0],
                    max_workspace_mb=workspace_mb,
                )
                np.testing.assert_array_equal(
                    one_query_batch.scores[0],
                    single_query.scores,
                )

            values = {}
            for m in {len(query) for query in queries}:
                for k in index.sizes:
                    values[(int(m), int(k))] = (0.25, 0.5)
            model = _null_model(index, index, values)
            batch_z = andes_index.calibrate_index_scores(
                batch_result,
                np.asarray([len(query) for query in queries], dtype=np.int32),
                index,
                index,
                model,
            ).scores
            np.testing.assert_allclose(
                batch_z,
                (expected - np.float32(0.25)) / np.float32(0.5),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_query_scores_are_invariant_to_workspace_chunking(self):
        query = np.arange(12, dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)

            one_gene_chunks = index.score_query(
                query,
                max_workspace_mb=0.000001,
            )
            small_chunks = index.score_query(
                query,
                max_workspace_mb=0.0005,
            )
            unlimited = index.score_query(query, max_workspace_mb=None)

            self.assertEqual(one_gene_chunks.stats.details["query_chunk_size"], 1)
            np.testing.assert_array_equal(
                one_gene_chunks.scores,
                small_chunks.scores,
            )
            np.testing.assert_array_equal(
                one_gene_chunks.scores,
                unlimited.scores,
            )

    def test_query_top_k_preserves_database_order_across_score_ties(self):
        terms = ["first", "second", "third"]
        identical = {term: np.asarray([0, 2, 4], dtype=np.int32) for term in terms}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "index"
            output = root / "query.csv"
            self.build(index_dir, terms=terms, indices=identical)

            self.assertEqual(
                index_cli.main(
                    [
                        "query",
                        "--index",
                        str(index_dir),
                        "--term",
                        "first",
                        "--no-zscore",
                        "--top-k",
                        "2",
                        "--out",
                        str(output),
                    ]
                ),
                0,
            )
            frame = pd.read_csv(output)
            self.assertEqual(frame["term"].tolist(), ["first", "second"])

    def test_prebuilt_query_null_is_content_addressed_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            cache_dir = null_cache_dir("bma", root / "cache")
            query_sizes = [1, 2, 3]

            built = resolve_bma_null(
                index.embedding_space(),
                index.background,
                index.background,
                query_sizes,
                index.sizes,
                base_dir=cache_dir,
                iterations=3,
                seed=31,
            )
            reused = resolve_bma_null(
                index.embedding_space(),
                index.background,
                index.background,
                query_sizes,
                index.sizes,
                base_dir=cache_dir,
                iterations=3,
                seed=31,
                no_build=True,
            )

            self.assertTrue(built.built)
            self.assertFalse(reused.built)
            self.assertEqual(reused.path, built.path)
            self.assertEqual(reused.model.spec, built.model.spec)
            for query_size in query_sizes:
                for term_size in index.sizes:
                    self.assertTrue(reused.model.present[query_size, int(term_size)])

    def test_index_calibration_preserves_float64_null_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            query = np.asarray([0], dtype=np.int32)
            exact = index.score_query(query)
            true_scores = exact.scores

            first_score = float(true_scores[0])
            delta = abs(float(np.spacing(np.float32(first_score)))) / 4.0
            values = {
                (1, int(size)): (first_score + delta, delta)
                for size in np.unique(index.sizes)
            }
            model = _null_model(index, index, values)
            expected = model.standardize_matrix(
                true_scores[None, :],
                np.asarray([1], dtype=np.int32),
                index.sizes,
            )[0]
            observed = andes_index.calibrate_index_scores(
                exact,
                np.asarray([len(query)], dtype=np.int32),
                index,
                index,
                model,
            ).scores

            self.assertEqual(expected[0], np.float32(-1.0))
            np.testing.assert_array_equal(observed, expected)

    def test_indexed_term_fast_path_matches_scalar_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            for term in index.terms:
                observed = index.score_indexed_term(term).scores
                expected = np.asarray(
                    [
                        _bma_reference(
                            self.E,
                            self.indices1[term],
                            self.indices1[target],
                        )
                        for target in self.terms1
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

    def test_indexed_term_accepts_name_or_position_and_rejects_invalid_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            by_position = index.score_indexed_term(1).scores
            by_name = index.score_indexed_term(index.terms[1]).scores
            np.testing.assert_array_equal(by_position, by_name)
            with self.assertRaisesRegex(KeyError, "not present"):
                index.score_indexed_term("absent")
            for invalid in (True, 1.5):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaisesRegex(TypeError, "string identifier"),
                ):
                    index.score_indexed_term(cast(Any, invalid))

    def test_loaded_index_shares_large_canonical_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            embedding = index.embedding_space()
            database = index.gene_set_database()
            artifact = index.ranked_bestmatch_artifact()
            self.assertTrue(np.shares_memory(embedding.vectors, index.E_unit))
            self.assertTrue(
                np.shares_memory(database.members, index.membership.indices)
            )
            self.assertTrue(np.shares_memory(database.sizes, index.sizes))
            self.assertEqual(
                artifact.embedding_fingerprint,
                embedding.fingerprint,
            )
            self.assertEqual(
                artifact.database_fingerprint,
                database.fingerprint,
            )
