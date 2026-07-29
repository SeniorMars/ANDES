import io
import json
import tempfile
import unittest
from argparse import Namespace
from collections.abc import MutableMapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np
import pandas as pd

from andes import application, artifacts, null_cli, ranked
from andes import data as load_data
from andes import nulls as andes_nulls
from andes.provenance import RunProvenance
from andes.scoring import ScoreResult, ScoreStats


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

    def test_ranked_list_parser_rejects_duplicate_mapped_genes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ranking.tsv"
            path.write_text(
                '"known_a"\t3.0\nunknown\t2.0\nknown_b\t1.0\n',
                encoding="utf-8",
            )
            observed = load_data.load_ranked_indices(
                path,
                {"known_a": 4, "known_b": 2},
            )
            np.testing.assert_array_equal(observed, [4, 2])

            path.write_text("known_a\nunknown\nknown_a\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"ranking\.tsv:3: duplicate ranked gene.*line 1",
            ):
                load_data.load_ranked_indices(path, {"known_a": 4})

    def test_ranked_list_parser_rejects_blank_and_unmapped_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ranking.tsv"
            path.write_text("known\n\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"ranking\.tsv:2: empty"):
                load_data.load_ranked_indices(path, {"known": 0})

            path.write_text("unknown\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no genes in the embedding"):
                load_data.load_ranked_indices(path, {"known": 0})


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
            background_policy="retained_members",
            min_size=2,
            max_size=3,
        )

        self.assertEqual(database.terms, ("second", "first"))
        np.testing.assert_array_equal(database.members_for("second"), [2, 3, 4])
        np.testing.assert_array_equal(database.members_at(1), [0, 1])
        np.testing.assert_array_equal(database.background, [0, 1, 2, 3, 4])
        self.assertFalse(database.members.flags.writeable)

        rebuilt = load_data.GeneSetDatabase.from_index_mapping(
            {
                term: database.members_at(position)
                for position, term in enumerate(database.terms)
            },
            n_genes=6,
            embedding_fingerprint="embedding-a",
            terms=database.terms,
            background=database.background,
            background_policy="reconstructed",
            min_size=1,
        )
        self.assertEqual(database.fingerprint, rebuilt.fingerprint)

        subset = database.select_terms(["first"])
        self.assertEqual(subset.terms, ("first",))
        np.testing.assert_array_equal(subset.members_at(0), [0, 1])
        np.testing.assert_array_equal(subset.background, database.background)
        self.assertEqual(subset.n_genes, database.n_genes)

    def test_rejects_bad_memberships(self):
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"bad": np.asarray([0, 3], dtype=np.int32)},
                n_genes=3,
                embedding_fingerprint="embedding-a",
                background_policy="retained_members",
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"a": np.asarray([0]), "b": np.asarray([1])},
                n_genes=2,
                embedding_fingerprint="embedding-a",
                terms=["a", "a"],
                background_policy="retained_members",
            )
        with self.assertRaisesRegex(ValueError, "background omits"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"a": np.asarray([0, 1], dtype=np.int32)},
                n_genes=3,
                embedding_fingerprint="embedding-a",
                background=np.asarray([0, 2], dtype=np.int32),
                background_policy="explicit_test_background",
            )

    def test_positional_and_named_membership_apis_cannot_be_confused(self):
        database = load_data.GeneSetDatabase.from_index_mapping(
            {
                "1": np.asarray([0], dtype=np.int32),
                "alpha": np.asarray([1], dtype=np.int32),
            },
            n_genes=2,
            embedding_fingerprint="embedding-a",
            background_policy="retained_members",
        )

        np.testing.assert_array_equal(database.members_for("1"), [0])
        np.testing.assert_array_equal(database.members_at(np.int64(1)), [1])
        with self.assertRaisesRegex(TypeError, "position must be an integer"):
            database.members_at(cast(Any, "1"))
        with self.assertRaisesRegex(TypeError, "position must be an integer"):
            database.members_at(cast(Any, True))
        with self.assertRaisesRegex(TypeError, "position must be an integer"):
            database.packed_axis.members_at(cast(Any, "1"))

    def test_background_policy_is_explicit_and_includes_filtered_gmt_genes(self):
        with self.assertRaisesRegex(ValueError, "background_policy"):
            load_data.GeneSetDatabase.from_index_mapping(
                {"retained": np.asarray([0, 1], dtype=np.int32)},
                n_genes=4,
                embedding_fingerprint="embedding-a",
            )

        embedding = load_data.EmbeddingSpace.from_arrays(
            np.eye(4, dtype=np.float32),
            ["g0", "g1", "g2", "g3"],
        )
        with tempfile.TemporaryDirectory() as root:
            gmt = Path(root) / "sets.gmt"
            gmt.write_text(
                "retained\tdescription\tg0\tg1\nfiltered\tdescription\tg3\n",
                encoding="utf-8",
            )
            database = load_data.load_gene_set_database(
                gmt,
                embedding,
                min_size=2,
                max_size=3,
            )

        self.assertEqual(database.terms, ("retained",))
        self.assertEqual(database.background_policy, "all_mapped_genes")
        np.testing.assert_array_equal(database.background, [0, 1, 3])

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

            path.write_text(
                "empty\tdescription\t\ntrailing\tdescription\tg1\tg2\t\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_data.load_gmt(path),
                {"empty": [], "trailing": ["g1", "g2"]},
            )

            path.write_text(
                "internal\tdescription\tg1\t\tg2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "empty GMT gene identifier"):
                load_data.load_gmt(path)

    def test_multiple_gmts_share_one_canonical_background_and_term_namespace(self):
        embedding = load_data.EmbeddingSpace.from_arrays(
            np.eye(5, dtype=np.float32),
            [f"g{i}" for i in range(5)],
        )
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "first.gmt"
            second = Path(root) / "second.gmt"
            first.write_text(
                "first\tdescription\tg0\tg1\nfiltered\tdescription\tg4\n",
                encoding="utf-8",
            )
            second.write_text(
                "second\tdescription\tg2\tg3\n",
                encoding="utf-8",
            )
            database = load_data.load_gene_set_databases(
                [first, second],
                embedding,
                min_size=2,
                max_size=2,
                sort_terms=True,
            )
            self.assertEqual(database.terms, ("first", "second"))
            np.testing.assert_array_equal(database.background, [0, 1, 2, 3, 4])

            second.write_text(
                "first\tanother description\tg2\tg3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate term across GMT"):
                load_data.load_gene_set_databases(
                    [first, second],
                    embedding,
                    min_size=1,
                    max_size=2,
                )


class ArtifactHelperTests(unittest.TestCase):
    def test_atomic_directory_is_hidden_until_complete(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            target = root_path / "artifact"
            with artifacts.atomic_artifact_directory(target) as building:
                (building / "payload.txt").write_text("complete")
                self.assertFalse(target.exists())
            self.assertEqual((target / "payload.txt").read_text(), "complete")

    def test_failed_json_replacement_preserves_published_file(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "metadata.json"
            target.write_text('{"schema": 1}\n', encoding="utf-8")

            with (
                mock.patch.object(
                    artifacts.os,
                    "replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                artifacts.write_json_atomic(target, {"schema": 2})

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"schema": 1},
            )
            self.assertEqual(
                sorted(path.name for path in Path(root).iterdir()),
                ["metadata.json"],
            )

    def test_artifact_lock_excludes_a_second_writer(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "shared.null"
            with (
                artifacts.artifact_lock(target),
                self.assertRaisesRegex(TimeoutError, "artifact lock"),
                artifacts.artifact_lock(target, timeout_seconds=0),
            ):
                pass

            lock_path = Path(root) / ".shared.null.lock"
            writer_record = lock_path.read_text(encoding="utf-8")
            writer_mtime = lock_path.stat().st_mtime_ns
            with artifacts.artifact_lock(target, shared=True):
                self.assertEqual(
                    lock_path.read_text(encoding="utf-8"),
                    writer_record,
                )
            self.assertEqual(lock_path.stat().st_mtime_ns, writer_mtime)

    def test_noncreating_reader_rejects_missing_lock_without_adopting_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "published.null"
            target.mkdir()
            lock_path = Path(root) / ".published.null.lock"

            with (
                self.assertRaisesRegex(FileNotFoundError, "republish or adopt"),
                artifacts.artifact_lock(
                    target,
                    shared=True,
                    create=False,
                ),
            ):
                pass

            self.assertFalse(lock_path.exists())

    def test_array_hash_is_independent_of_layout_and_chunk_size(self):
        value = np.arange(12, dtype=np.float32).reshape(3, 4)
        expected = artifacts.hash_array(value)
        self.assertEqual(
            expected,
            artifacts.hash_array(np.asfortranarray(value)),
        )
        self.assertEqual(
            expected,
            artifacts.hash_array(value, chunk_bytes=7),
        )

    def test_ranked_table_preserves_alignment_and_numerical_provenance(self):
        provenance = RunProvenance(
            method="andes_ranked",
            score_engine="indexed_ranked",
            score_kind="z_score",
            similarity_dtype="float32",
            score_accumulator_dtype="float64",
            null_accumulator_dtype="float64",
            output_dtype="float32",
            tie_policy=ranked.RANKED_ES_TIE_POLICY,
            embedding_fingerprint="embedding",
            left_database_fingerprint="database",
        )
        result = application.RankedResult(
            scores=ScoreResult(
                np.asarray([1.25, -0.5], dtype=np.float32),
                ScoreStats.create("indexed_ranked"),
            ),
            true_scores=np.asarray([0.25, -0.1], dtype=np.float32),
            terms=("first", "second"),
            sizes=np.asarray([2, 3], dtype=np.int32),
            null_means=np.asarray([0.0, 0.1], dtype=np.float64),
            null_stds=np.asarray([0.2, 0.4], dtype=np.float64),
            score_kind="z_score",
            provenance=provenance,
        )

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "ranked.csv"
            sidecar = application.write_ranked_result(
                result,
                output,
                provenance_extra={"expression_derived": False},
            )

            frame = pd.read_csv(output, index_col=0)
            self.assertEqual(list(frame.index), ["first", "second"])
            self.assertEqual(
                list(frame.columns),
                ["size", "true_score", "null_mu", "null_sigma", "z_score"],
            )
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(metadata["tie_policy"], ranked.RANKED_ES_TIE_POLICY)
            self.assertEqual(metadata["output_dtype"], "mixed")
            self.assertFalse(metadata["extra"]["expression_derived"])

    def test_provenance_copies_and_freezes_caller_owned_mappings(self):
        runtime = {"query_seconds": 1.0}
        provenance = RunProvenance(
            method="andes_ranked",
            score_engine="indexed",
            score_kind="z_score",
            similarity_dtype="float32",
            score_accumulator_dtype="float64",
            null_accumulator_dtype="float64",
            output_dtype="float32",
            tie_policy=ranked.RANKED_ES_TIE_POLICY,
            embedding_fingerprint="embedding",
            left_database_fingerprint="database",
            runtime=runtime,
        )

        runtime["query_seconds"] = 2.0
        self.assertEqual(provenance.runtime["query_seconds"], 1.0)
        with self.assertRaises(TypeError):
            cast(MutableMapping[str, object], provenance.runtime)["query_seconds"] = 3.0

    def test_matrix_manifest_binds_scores_and_axis_labels(self):
        provenance = RunProvenance(
            method="andes_bma",
            score_engine="bestmatch",
            score_kind="true_score",
            similarity_dtype="float32",
            score_accumulator_dtype="float32",
            null_accumulator_dtype="not_applicable",
            output_dtype="float32",
            tie_policy="not_applicable",
            embedding_fingerprint="embedding",
            left_database_fingerprint="left",
            right_database_fingerprint="right",
        )
        result = application.CompareResult(
            scores=ScoreResult(
                np.asarray([[0.25, 0.5]], dtype=np.float32),
                ScoreStats.create("bestmatch"),
            ),
            row_terms=("row",),
            column_terms=("first", "second"),
            score_kind="true_score",
            provenance=provenance,
        )

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "scores.npy"
            sidecar = application.write_compare_result(result, output)
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(metadata["manifest_version"], 1)
            expected_paths = (
                output,
                output.with_suffix(".rows.json"),
                output.with_suffix(".columns.json"),
            )
            self.assertEqual(
                set(metadata["files"]),
                {path.name for path in expected_paths},
            )
            for path in expected_paths:
                self.assertEqual(
                    metadata["files"][path.name],
                    {
                        "size_bytes": path.stat().st_size,
                        "hash": artifacts.hash_file(path),
                    },
                )

            rows_path = output.with_suffix(".rows.json")
            recorded_hash = metadata["files"][rows_path.name]["hash"]
            artifacts.write_json_atomic(rows_path, ["different"])
            self.assertNotEqual(recorded_hash, artifacts.hash_file(rows_path))

    def test_atomic_directory_requires_explicit_overwrite_even_when_empty(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "artifact"
            target.mkdir()
            with (
                self.assertRaises(FileExistsError),
                artifacts.atomic_artifact_directory(target),
            ):
                pass

    def test_failed_overwrite_preserves_the_published_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "artifact"
            target.mkdir()
            (target / "payload.txt").write_text("published", encoding="utf-8")

            with (
                self.assertRaisesRegex(RuntimeError, "build failed"),
                artifacts.atomic_artifact_directory(
                    target,
                    overwrite=True,
                ) as building,
            ):
                (building / "payload.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("build failed")

            self.assertEqual(
                (target / "payload.txt").read_text(encoding="utf-8"),
                "published",
            )
            self.assertEqual(
                sorted(path.name for path in Path(root).iterdir()),
                ["artifact"],
            )


class NullArtifactTests(unittest.TestCase):
    def test_null_spec_requires_more_samples_than_ddof(self):
        with self.assertRaisesRegex(ValueError, "greater than ddof"):
            andes_nulls.NullSpec(
                kind="bma",
                iterations=1,
                seed=1,
                sampling="prefix_coupled",
                ddof=1,
                embedding_hash="embedding",
                population_hashes=("left", "right"),
            )

    def test_null_spec_rejects_non_prefix_sampling(self):
        with self.assertRaisesRegex(ValueError, "prefix-coupled"):
            andes_nulls.NullSpec(
                kind="bma",
                iterations=10,
                seed=1,
                sampling="per_size_pair",
                ddof=1,
                embedding_hash="embedding",
                population_hashes=("left", "right"),
            )

    def test_null_listing_reports_a_malformed_artifact_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as root:
            cache_root = Path(root)
            (cache_root / "bma").mkdir()
            (cache_root / "ranked").mkdir()
            (cache_root / "bma" / "broken.null").write_text(
                "not a directory",
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                null_cli.cmd_list(Namespace(cache_root=str(cache_root)))

            listing = output.getvalue()
            self.assertIn("broken.null", listing)
            self.assertIn("INVALID", listing)

    def test_cache_root_has_one_explicit_environment_policy(self):
        with tempfile.TemporaryDirectory() as root:
            environment_root = Path(root) / "environment"
            explicit_root = Path(root) / "explicit"
            with mock.patch.dict(
                "os.environ",
                {"ANDES_CACHE_ROOT": str(environment_root)},
            ):
                self.assertEqual(
                    andes_nulls.resolve_cache_root(),
                    environment_root,
                )
                self.assertEqual(
                    andes_nulls.resolve_cache_root(explicit_root),
                    explicit_root,
                )
                self.assertEqual(
                    andes_nulls.null_cache_dir("bma"),
                    environment_root / "bma",
                )
                self.assertEqual(
                    andes_nulls.null_cache_dir("ranked"),
                    environment_root / "ranked",
                )

    def test_query_size_ranges_are_inclusive_and_canonical(self):
        self.assertEqual(
            null_cli.parse_size_spec("3,1:3,5"),
            [1, 2, 3, 5],
        )
        for invalid in ("", "0", "3:2", "1::3", "two"):
            with (
                self.subTest(value=invalid),
                self.assertRaises(ValueError),
            ):
                null_cli.parse_size_spec(invalid)

    def test_bma_round_trip_and_in_place_standardization(self):
        spec = andes_nulls.NullSpec(
            kind="bma",
            iterations=25,
            seed=7,
            sampling="prefix_coupled",
            ddof=1,
            embedding_hash="embedding",
            population_hashes=("left", "right"),
        )
        model = andes_nulls.BmaNullModel.from_mapping(
            {
                (1, 2): (0.25, 0.5),
                (2, 2): (-0.5, 2.0),
            },
            spec,
        )
        scores = np.asarray([[0.75], [1.5]], dtype=np.float32)
        calibrated = model.standardize_matrix(
            scores,
            np.asarray([1, 2]),
            np.asarray([2]),
            out=scores,
        )
        self.assertIs(calibrated, scores)
        np.testing.assert_allclose(calibrated[:, 0], [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "int32"):
            model.standardize_matrix(
                np.asarray([[1.0]], dtype=np.float32),
                np.asarray([2**32 + 1], dtype=np.uint64),
                np.asarray([2], dtype=np.int32),
            )

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bma.null"
            model.save(path)
            loaded = andes_nulls.load_null_model(path)
            self.assertEqual(loaded.spec, model.spec)
            self.assertEqual(loaded.to_mapping(), model.to_mapping())

            metadata = json.loads((path / "metadata.json").read_text())
            del metadata["spec"]["sampling"]
            artifacts.write_json_atomic(path / "metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "sampling"):
                andes_nulls.BmaNullModel.load(path)

    def test_ranked_round_trip_and_corruption_rejection(self):
        spec = andes_nulls.NullSpec(
            kind="ranked",
            iterations=50,
            seed=11,
            sampling="prefix_coupled",
            ddof=1,
            embedding_hash="embedding",
            population_hashes=("population",),
            ranked_hash="ranked",
            tie_policy=ranked.RANKED_ES_TIE_POLICY,
        )
        model = andes_nulls.RankedNullModel.from_mapping(
            {2: (1.0, 2.0), 4: (-2.0, 0.0)},
            spec,
        )
        np.testing.assert_allclose(
            model.standardize(
                np.asarray([5.0, 10.0], dtype=np.float32),
                np.asarray([2, 4], dtype=np.int32),
            ),
            [2.0, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "int32"):
            model.standardize(
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2**32 + 1], dtype=np.uint64),
            )

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ranked.null"
            model.save(path)
            loaded = andes_nulls.RankedNullModel.load(path)
            self.assertEqual(loaded.to_mapping(), model.to_mapping())

            metadata = json.loads((path / "metadata.json").read_text())
            metadata["artifact_version"] = 1
            artifacts.write_json_atomic(path / "metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "rebuild"):
                andes_nulls.RankedNullModel.load(path)
            metadata["artifact_version"] = andes_nulls.NULL_ARTIFACT_VERSION
            artifacts.write_json_atomic(path / "metadata.json", metadata)

            means = np.load(path / "means.npy", allow_pickle=False)
            means[2] += 1.0
            np.save(path / "means.npy", means, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                andes_nulls.RankedNullModel.load(path)
