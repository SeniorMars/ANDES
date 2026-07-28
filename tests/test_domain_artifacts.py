import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from andes import artifacts
from andes import nulls as andes_nulls
from andes import ranked
from andes import bma
from andes import data as load_data


class EmbeddingSpaceTests(unittest.TestCase):
    def test_normalizes_validates_and_fingerprints_gene_order(self):
        raw = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
        space = load_data.EmbeddingSpace.from_arrays(raw, ["a", "b"])

        np.testing.assert_allclose(
            np.linalg.norm(space.vectors, axis=1),
            np.ones(2, dtype=np.float32),
            rtol=1e-6,
        )
        self.assertEqual(space.gene_to_index["b"], 1)
        self.assertFalse(space.vectors.flags.writeable)

        same = load_data.EmbeddingSpace.from_arrays(raw.copy(), ["a", "b"])
        reordered = load_data.EmbeddingSpace.from_arrays(raw[::-1], ["b", "a"])
        self.assertEqual(space.fingerprint, same.fingerprint)
        self.assertNotEqual(space.fingerprint, reordered.fingerprint)

    def test_rejects_duplicate_genes_and_zero_vectors(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            load_data.EmbeddingSpace.from_arrays(np.eye(2), ["a", "a"])
        with self.assertRaisesRegex(ValueError, "zero-length"):
            load_data.EmbeddingSpace.from_arrays(
                np.asarray([[1.0, 0.0], [0.0, 0.0]]),
                ["a", "b"],
            )
        with self.assertRaisesRegex(ValueError, "unit-normalized"):
            load_data.EmbeddingSpace.from_arrays(
                np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                ["a", "b"],
                normalize=False,
            )


class GeneSetDatabaseTests(unittest.TestCase):
    def test_canonicalizes_before_filtering_and_preserves_term_order(self):
        database = load_data.GeneSetDatabase.from_index_mapping(
            {
                "second": np.asarray([4, 2, 4, 3], dtype=np.int64),
                "filtered": np.asarray([1, 1, 1], dtype=np.int64),
                "first": np.asarray([1, 0, 1], dtype=np.int32),
            },
            n_genes=6,
            embedding_fingerprint="embedding-a",
            min_size=2,
            max_size=3,
        )

        self.assertEqual(database.terms, ("second", "first"))
        np.testing.assert_array_equal(database.indices("second"), [2, 3, 4])
        np.testing.assert_array_equal(database.indices(1), [0, 1])
        np.testing.assert_array_equal(database.background, [0, 1, 2, 3, 4])
        self.assertFalse(database.members.flags.writeable)

        rebuilt = load_data.GeneSetDatabase.from_index_mapping(
            database.as_index_mapping(),
            n_genes=6,
            embedding_fingerprint="embedding-a",
            terms=database.terms,
            background=database.background,
            min_size=1,
        )
        self.assertEqual(database.fingerprint, rebuilt.fingerprint)

    def test_rejects_bad_memberships(self):
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"bad": np.asarray([0, 3], dtype=np.int32)},
                n_genes=3,
                embedding_fingerprint="embedding-a",
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"a": np.asarray([0]), "b": np.asarray([1])},
                n_genes=2,
                embedding_fingerprint="embedding-a",
                terms=["a", "a"],
            )
        with self.assertRaisesRegex(ValueError, "background omits"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"a": np.asarray([0, 1], dtype=np.int32)},
                n_genes=3,
                embedding_fingerprint="embedding-a",
                background=np.asarray([0, 2], dtype=np.int32),
            )

    def test_gmt_rejects_malformed_and_duplicate_terms(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sets.gmt"
            path.write_text("term\tdescription\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires at least"):
                load_data.load_gmt(path)

            path.write_text(
                "term\tdescription\tg1\nterm\tother\tg2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate GMT term"):
                load_data.load_gmt(path)


class ArtifactHelperTests(unittest.TestCase):
    def test_atomic_json_and_directory_publish(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            metadata_path = root_path / "metadata.json"
            artifacts.write_json_atomic(metadata_path, {"schema": 2})
            self.assertEqual(json.loads(metadata_path.read_text()), {"schema": 2})

            target = root_path / "artifact"
            with artifacts.atomic_artifact_directory(target) as building:
                (building / "payload.txt").write_text("complete")
                self.assertFalse(target.exists())
            self.assertEqual((target / "payload.txt").read_text(), "complete")

    def test_shared_array_hash_matches_existing_contract(self):
        value = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(
            artifacts.hash_array(value),
            artifacts.hash_array(np.asfortranarray(value)),
        )

    def test_atomic_directory_requires_explicit_overwrite_even_when_empty(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "artifact"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                with artifacts.atomic_artifact_directory(target):
                    pass


class NullArtifactTests(unittest.TestCase):
    def test_bma_round_trip_and_in_place_standardization(self):
        cache = bma.BmaNullBuilder()
        cache.metadata = {
            "kind": "andes_bma_null",
            "version": 2,
            "embedding_hash": "embedding",
            "population1_hash": "left",
            "population2_hash": "right",
            "ite": 25,
            "seed": 7,
            "std_ddof": 1,
            "null_sampling": "prefix_coupled",
        }
        cache.cache = {
            (1, 2): (0.25, 0.5),
            (2, 2): (-0.5, 2.0),
        }

        model = andes_nulls.BmaNullModel.from_builder(cache)
        scores = np.asarray([[0.75], [1.5]], dtype=np.float32)
        calibrated = model.standardize_matrix(
            scores,
            np.asarray([1, 2]),
            np.asarray([2]),
            out=scores,
        )
        self.assertIs(calibrated, scores)
        np.testing.assert_allclose(calibrated[:, 0], [1.0, 1.0])

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bma.null"
            model.save(path)
            loaded = andes_nulls.load_null_model(path)
            self.assertEqual(loaded.spec, model.spec)
            self.assertEqual(loaded.to_mapping(), model.to_mapping())

            restored = bma.BmaNullBuilder()
            restored.load_artifact(path)
            self.assertEqual(restored.cache, cache.cache)

    def test_ranked_round_trip_and_corruption_rejection(self):
        cache = ranked.RankedNullBuilder()
        cache.metadata = {
            "kind": "andes_gsea_es_null",
            "version": 4,
            "embedding_hash": "embedding",
            "population_hash": "population",
            "ranked_emb_hash": "ranked",
            "ite": 50,
            "seed": 11,
        }
        cache.cache = {2: (1.0, 2.0), 4: (-2.0, 0.0)}
        model = andes_nulls.RankedNullModel.from_builder(cache)
        np.testing.assert_allclose(
            model.standardize(
                np.asarray([5.0, 10.0], dtype=np.float32),
                np.asarray([2, 4], dtype=np.int32),
            ),
            [2.0, 0.0],
        )

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ranked.null"
            model.save(path)
            loaded = andes_nulls.RankedNullModel.load(path)
            self.assertEqual(loaded.to_mapping(), model.to_mapping())

            means = np.load(path / "means.npy", allow_pickle=False)
            means[2] += 1.0
            np.save(path / "means.npy", means, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                andes_nulls.RankedNullModel.load(path)


if __name__ == "__main__":
    unittest.main()
