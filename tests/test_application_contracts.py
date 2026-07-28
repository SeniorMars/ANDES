import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from andes import application as andes_application
from andes import artifacts
from andes import nulls as andes_nulls
from andes import ranked
from andes import data as load_data
from andes import cli


class ApplicationContractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(818)
        self.embedding = load_data.EmbeddingSpace.from_arrays(
            rng.normal(size=(8, 5)),
            [f"g{i}" for i in range(8)],
        )
        self.database = load_data.GeneSetDatabase.from_index_mapping(
            {
                "a": np.asarray([0, 1], dtype=np.int32),
                "b": np.asarray([2, 3, 4], dtype=np.int32),
            },
            n_genes=8,
            embedding_fingerprint=self.embedding.fingerprint,
        )

    def test_compare_service_and_writer_return_structured_provenance(self):
        result = andes_application.run_compare(
            andes_application.CompareRequest(
                embedding=self.embedding,
                left=self.database,
                right=self.database,
                engine="bestmatch",
                workspace_mb=0.01,
                runtime={"query_blas_threads": 2},
            )
        )
        self.assertEqual(result.score_kind, "true_score")
        self.assertTrue(result.provenance.symmetric_reuse)
        self.assertEqual(result.scores.scores.shape, (2, 2))

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "scores.csv"
            sidecar = andes_application.write_compare_result(result, output)
            frame = pd.read_csv(output, index_col=0)
            self.assertEqual(frame.shape, (2, 2))
            metadata = json.loads(sidecar.read_text())
            self.assertEqual(metadata["method"], "andes_bma")
            self.assertEqual(metadata["output_file"], "scores.csv")
            self.assertEqual(len(metadata["output_hash"]), 32)

            binary = Path(root) / "scores.npy"
            andes_application.write_compare_result(result, binary)
            self.assertEqual(
                json.loads(binary.with_suffix(".rows.json").read_text()),
                list(result.row_terms),
            )
            self.assertEqual(
                json.loads(binary.with_suffix(".columns.json").read_text()),
                list(result.column_terms),
            )

    def test_ranked_service_calibrates_in_place(self):
        spec = andes_nulls.NullSpec(
            kind="ranked",
            iterations=20,
            seed=3,
            sampling="prefix_coupled",
            ddof=1,
            embedding_hash=artifacts.hash_array(self.embedding.vectors),
            population_hashes=(artifacts.hash_array(self.database.background),),
            ranked_hash=artifacts.hash_array(
                ranked.compute_ranked_emb(
                    self.embedding.vectors,
                    np.arange(8, dtype=np.int32),
                )
            ),
        )
        model = andes_nulls.RankedNullModel.from_mapping(
            {2: (0.0, 1.0), 3: (0.0, 1.0)}, spec
        )
        result = andes_application.run_ranked(
            andes_application.RankedRequest(
                embedding=self.embedding,
                database=self.database,
                ranked_indices=np.arange(8, dtype=np.int32),
                engine="batched",
                workspace_mb=0.01,
                null_model=model,
            )
        )
        self.assertEqual(result.score_kind, "z_score")
        self.assertEqual(result.provenance.null_spec["iterations"], 20)
        self.assertEqual(result.scores.scores.shape, (2,))

        with self.assertRaisesRegex(ValueError, "ranking is incompatible"):
            andes_application.run_ranked(
                andes_application.RankedRequest(
                    embedding=self.embedding,
                    database=self.database,
                    ranked_indices=np.arange(7, -1, -1, dtype=np.int32),
                    engine="batched",
                    workspace_mb=0.01,
                    null_model=model,
                )
            )

    def test_unified_cli_dispatch_is_directly_testable(self):
        received = []

        def handler(arguments):
            received.append(list(arguments))
            return 0

        original = cli.COMMANDS["compare"]
        cli.COMMANDS["compare"] = handler
        try:
            self.assertEqual(cli.main(["compare", "--example", "value"]), 0)
        finally:
            cli.COMMANDS["compare"] = original
        self.assertEqual(received, [["--example", "value"]])


if __name__ == "__main__":
    unittest.main()
