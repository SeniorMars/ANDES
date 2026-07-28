import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from andes import artifacts
from andes import nulls as andes_nulls
from andes import scoring as andes_scoring
from andes import bma
from andes import data as load_data


class TypedScoringApiTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        raw = rng.normal(size=(9, 6)).astype(np.float32)
        self.embedding = load_data.EmbeddingSpace.from_arrays(
            raw, [f"g{i}" for i in range(raw.shape[0])]
        )
        self.left = load_data.GeneSetDatabase.from_index_mapping(
            {
                "left_a": np.asarray([0, 1, 2], dtype=np.int32),
                "left_b": np.asarray([3, 4], dtype=np.int32),
            },
            n_genes=9,
            embedding_fingerprint=self.embedding.fingerprint,
        )
        self.right = load_data.GeneSetDatabase.from_index_mapping(
            {
                "right_a": np.asarray([1, 3], dtype=np.int32),
                "right_b": np.asarray([5, 6, 7], dtype=np.int32),
            },
            n_genes=9,
            embedding_fingerprint=self.embedding.fingerprint,
        )

    def test_bma_engines_return_the_same_exact_matrix(self):
        results = {
            engine: andes_scoring.score_bma_matrix(
                self.embedding,
                self.left,
                self.right,
                engine=engine,
                workers=1,
                workspace_mb=0.01,
            )
            for engine in ("pairwise", "batched", "bestmatch")
        }
        np.testing.assert_allclose(
            results["batched"].scores,
            results["pairwise"].scores,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            results["bestmatch"].scores,
            results["pairwise"].scores,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertEqual(results["bestmatch"].stats.engine, "bestmatch")

    def test_bma_calibration_can_reuse_exact_output(self):
        result = andes_scoring.score_bma_matrix(
            self.embedding, self.left, self.right, engine="bestmatch"
        )
        original = result.scores.copy()
        spec = andes_nulls.NullSpec(
            kind="bma",
            iterations=10,
            seed=1,
            sampling="prefix_coupled",
            ddof=1,
            embedding_hash=artifacts.hash_array(self.embedding.vectors),
            population_hashes=(
                artifacts.hash_array(self.left.background),
                artifacts.hash_array(self.right.background),
            ),
        )
        cache = {
            (int(left), int(right)): (0.0, 1.0)
            for left in np.unique(self.left.sizes)
            for right in np.unique(self.right.sizes)
        }
        model = andes_nulls.BmaNullModel.from_mapping(cache, spec)
        calibrated = andes_scoring.calibrate_bma(
            result,
            model,
            self.embedding,
            self.left,
            self.right,
            in_place=True,
        )
        self.assertIs(calibrated.scores, result.scores)
        np.testing.assert_allclose(calibrated.scores, original)

        incompatible_spec = andes_nulls.NullSpec(
            kind="bma",
            iterations=10,
            seed=1,
            sampling="prefix_coupled",
            ddof=1,
            embedding_hash=artifacts.hash_array(self.embedding.vectors),
            population_hashes=("wrong-left", "wrong-right"),
        )
        incompatible = andes_nulls.BmaNullModel.from_mapping(
            cache,
            incompatible_spec,
        )
        with self.assertRaisesRegex(ValueError, "background is incompatible"):
            andes_scoring.calibrate_bma(
                result,
                incompatible,
                self.embedding,
                self.left,
                self.right,
            )

    def test_ranked_engines_return_aligned_exact_vectors(self):
        ranked = np.asarray([8, 0, 2, 5, 4, 1, 7, 3, 6], dtype=np.int32)
        batched = andes_scoring.score_ranked(
            self.embedding,
            self.right,
            ranked,
            engine="batched",
            workspace_mb=0.01,
        )
        bestmatch = andes_scoring.score_ranked(
            self.embedding,
            self.right,
            ranked,
            engine="bestmatch",
            workspace_mb=0.01,
        )

        term_indices = self.right.as_index_mapping()
        blocks = bma.precompute_term_embedding_blocks(
            self.embedding.vectors, term_indices
        )
        persistent, _ = bma.gene_to_term_best_match_matrix(
            self.embedding.vectors,
            self.right.terms,
            blocks,
            max_workspace_mb=0.01,
        )
        indexed = andes_scoring.score_ranked(
            self.embedding,
            self.right,
            ranked,
            engine="indexed",
            bestmatch=andes_scoring.IndexedBestMatch(
                persistent,
                self.embedding.fingerprint,
                self.right.fingerprint,
            ),
            workspace_mb=0.01,
        )

        np.testing.assert_allclose(
            bestmatch.scores, batched.scores, rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(indexed.scores, batched.scores, rtol=2e-5, atol=2e-5)

        with self.assertRaisesRegex(TypeError, "IndexedBestMatch"):
            andes_scoring.score_ranked(
                self.embedding,
                self.right,
                ranked,
                engine="indexed",
                bestmatch=persistent,
            )

    def test_database_identity_rejects_same_sized_different_embedding(self):
        changed = load_data.EmbeddingSpace.from_arrays(
            self.embedding.vectors[::-1],
            tuple(reversed(self.embedding.genes)),
            normalize=False,
        )
        with self.assertRaisesRegex(ValueError, "different embedding"):
            andes_scoring.score_bma_matrix(
                changed,
                self.left,
                self.right,
            )


if __name__ == "__main__":
    unittest.main()
