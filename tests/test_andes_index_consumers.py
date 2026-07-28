import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from andes import index as andes_index
from andes import bma
from andes import data as andes_data


def _bma_reference(E_unit, left, right):
    similarities = E_unit[left] @ E_unit[right].T
    return np.float32(
        (
            similarities.max(axis=1).sum(dtype=np.float32)
            + similarities.max(axis=0).sum(dtype=np.float32)
        )
        / (len(left) + len(right))
    )


class AndesIndexConsumerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260728)
        raw = rng.normal(size=(18, 7)).astype(np.float32)
        self.E = np.ascontiguousarray(bma.l2_normalize_rows(raw), dtype=np.float32)
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
        return andes_index.build_andes_index(
            E,
            genes,
            terms,
            indices,
            background,
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
            self.assertGreaterEqual(
                metadata["chunk_workspace_mb"],
                metadata["bestmatch_write_chunk_mb"],
            )
            self.assertIsInstance(index.E_unit, np.memmap)
            self.assertIsInstance(index.bestmatch, np.memmap)
            np.testing.assert_array_equal(
                index.term_indices["dupes"],
                np.array([1, 3, 5], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                index.background,
                np.array([0, 1, 3, 4, 5, 7, 8, 17], dtype=np.int32),
            )
            expected = np.column_stack(
                [
                    (self.E @ self.E[index.term_indices[term]].T).max(axis=1)
                    for term in index.terms
                ]
            ).astype(np.float32)
            np.testing.assert_allclose(index.bestmatch, expected, rtol=1e-6, atol=1e-6)

    def test_build_rejects_bad_dtype_duplicate_genes_and_out_of_range_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "dtype float32"):
                self.build(Path(tmp) / "float64", E=self.E.astype(np.float64))

            unnormalized = self.E.copy()
            unnormalized[0] *= np.float32(2.0)
            with self.assertRaisesRegex(ValueError, "unit-normalized"):
                self.build(Path(tmp) / "unnormalized", E=unnormalized)

            duplicate_genes = list(self.genes)
            duplicate_genes[-1] = duplicate_genes[0]
            with self.assertRaisesRegex(ValueError, "duplicate names"):
                self.build(Path(tmp) / "genes", genes=duplicate_genes)

            bad_indices = dict(self.indices1)
            bad_indices["a2"] = np.array([0, self.E.shape[0]], dtype=np.int32)
            with self.assertRaisesRegex(ValueError, "outside"):
                self.build(Path(tmp) / "indices", indices=bad_indices)

            wrapped_indices = dict(self.indices1)
            wrapped_indices["a2"] = np.array([0, 2**32], dtype=np.uint64)
            with self.assertRaisesRegex(ValueError, "outside"):
                self.build(Path(tmp) / "wrapped", indices=wrapped_indices)

    def test_load_rejects_missing_file_bad_version_shape_and_gene_hash(self):
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

    def test_default_index_load_does_not_hash_large_matrix_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            index_dir = Path(root) / "index"
            self.build(index_dir)

            with mock.patch.object(
                andes_index,
                "_hash_array",
                wraps=andes_index._hash_array,
            ) as hash_array:
                andes_index.load_andes_index(index_dir, mmap=True)
            default_shapes = [
                np.asanyarray(call.args[0]).shape for call in hash_array.call_args_list
            ]
            self.assertNotIn(self.E.shape, default_shapes)
            self.assertNotIn(
                (self.E.shape[0], len(self.terms1)),
                default_shapes,
            )

            with mock.patch.object(
                andes_index,
                "_hash_array",
                wraps=andes_index._hash_array,
            ) as hash_array:
                andes_index.load_andes_index(
                    index_dir,
                    mmap=True,
                    verify_hashes=True,
                )
            full_shapes = [
                np.asanyarray(call.args[0]).shape for call in hash_array.call_args_list
            ]
            self.assertIn(self.E.shape, full_shapes)
            self.assertIn(
                (self.E.shape[0], len(self.terms1)),
                full_shapes,
            )

    def test_index_comparison_matches_pairwise_reference_and_axis_zscores(self):
        with tempfile.TemporaryDirectory() as root:
            dir1 = Path(root) / "one"
            dir2 = Path(root) / "two"
            self.build(dir1)
            self.build(dir2, terms=self.terms2, indices=self.indices2)
            index1 = andes_index.load_andes_index(dir1, mmap=True)
            index2 = andes_index.load_andes_index(dir2, mmap=True)

            true_scores, no_zscores = andes_index.score_index_to_index(index1, index2)
            self.assertIsNone(no_zscores)
            expected = np.empty_like(true_scores)
            for i, term1 in enumerate(self.terms1):
                for j, term2 in enumerate(self.terms2):
                    expected[i, j] = _bma_reference(
                        self.E,
                        self.indices1[term1],
                        self.indices2[term2],
                    )
            np.testing.assert_allclose(true_scores, expected, rtol=1e-6, atol=1e-6)

            self_scores, _ = index1.compare(index1)
            expected_self = np.empty_like(self_scores)
            for i, term1 in enumerate(self.terms1):
                for j, term2 in enumerate(self.terms1):
                    expected_self[i, j] = _bma_reference(
                        self.E,
                        self.indices1[term1],
                        self.indices1[term2],
                    )
            np.testing.assert_allclose(self_scores, expected_self, rtol=1e-6, atol=1e-6)

            cache = bma.BmaNullBuilder()
            for m in index1.sizes:
                for k in index2.sizes:
                    cache.cache[(int(m), int(k))] = (
                        float(10 * int(m) + int(k)),
                        2.0,
                    )
            _, zscores = index1.compare(index2, null_cache=cache)
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
            cache = bma.BmaNullBuilder()
            cache.metadata = bma.BmaNullBuilder.build_metadata(
                self.E,
                axis.background,
                base.background,
                ite=2,
                seed=7,
            )
            for m in base.sizes:
                for k in axis.sizes:
                    cache.cache[(int(m), int(k))] = (0.0, 1.0)
            with self.assertRaisesRegex(ValueError, "row/background-1"):
                base.compare(axis, null_cache=cache)

    def test_batch_queries_match_repeated_single_queries_across_small_chunks(self):
        queries = [
            np.array([0, 1], dtype=np.int32),
            np.array([2, 4, 6], dtype=np.int32),
            np.array([3, 5, 7, 9], dtype=np.int32),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            batch_true, batch_z = index.score_queries(
                queries,
                max_workspace_mb=0.00001,
            )
            self.assertIsNone(batch_z)
            repeated = np.vstack([index.score_query(query)[0] for query in queries])
            np.testing.assert_allclose(batch_true, repeated, rtol=1e-6, atol=1e-6)

            cache = bma.BmaNullBuilder()
            for m in {len(query) for query in queries}:
                for k in index.sizes:
                    cache.cache[(int(m), int(k))] = (0.25, 0.5)
            _, batch_z = index.score_queries(
                queries,
                null_cache=cache,
                max_workspace_mb=0.00001,
            )
            repeated_z = np.vstack(
                [index.score_query(query, null_cache=cache)[1] for query in queries]
            )
            np.testing.assert_allclose(batch_z, repeated_z, rtol=1e-6, atol=1e-6)

    def test_indexed_term_fast_path_matches_regular_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            self.build(index_dir)
            index = andes_index.load_andes_index(index_dir, mmap=True)
            for term in index.terms:
                expected, _ = index.score_query(index.term_indices[term])
                observed, _ = index.score_indexed_term(term)
                np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

            by_position, _ = index.score_indexed_term(1)
            by_name, _ = index.score_indexed_term(index.terms[1])
            np.testing.assert_array_equal(by_position, by_name)
            with self.assertRaisesRegex(KeyError, "not present"):
                index.score_indexed_term("absent")

            embedding = andes_data.EmbeddingSpace.from_arrays(
                index.E_unit,
                index.gene_list,
                normalize=False,
            )
            database = andes_data.GeneSetDatabase.from_index_mapping(
                index.term_indices,
                n_genes=len(index.gene_list),
                embedding_fingerprint=embedding.fingerprint,
                terms=index.terms,
                background=index.background,
            )
            artifact = index.ranked_bestmatch_artifact()
            self.assertEqual(
                artifact.embedding_fingerprint,
                embedding.fingerprint,
            )
            self.assertEqual(
                artifact.database_fingerprint,
                database.fingerprint,
            )

    def test_parser_exposes_compare_batch_and_indexed_term_modes(self):
        query = andes_index.parse_args(
            [
                "query",
                "--index",
                "index",
                "--term",
                "a2",
                "--out",
                "scores.csv",
                "--no-zscore",
            ]
        )
        self.assertEqual(query.cmd, "query")
        self.assertEqual(query.term, "a2")
        self.assertIsNone(query.genes)

        batch = andes_index.parse_args(
            [
                "batch-query",
                "--index",
                "index",
                "--queries",
                "queries.gmt",
                "--out",
                "scores.csv",
            ]
        )
        self.assertEqual(batch.cmd, "batch-query")

        compare = andes_index.parse_args(
            [
                "compare",
                "--index1",
                "one",
                "--index2",
                "two",
                "--out",
                "matrix.npy",
            ]
        )
        self.assertEqual(compare.cmd, "compare")


if __name__ == "__main__":
    unittest.main()
