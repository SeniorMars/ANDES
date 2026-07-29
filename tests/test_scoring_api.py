import numpy as np
import pytest

from andes import application, artifacts, bma, ranked, scoring
from andes import data as load_data
from andes import nulls as andes_nulls
from tests.reference import reference_bma, reference_ranked_es


def _scoring_problem(seed):
    rng = np.random.default_rng(seed)
    embedding = load_data.EmbeddingSpace.from_arrays(
        rng.normal(size=(17, 7)).astype(np.float32),
        [f"g{i}" for i in range(17)],
    )
    left = load_data.GeneSetDatabase.from_index_mapping(
        {
            "left_a": np.asarray([0, 1, 2], dtype=np.int32),
            "left_b": np.asarray([3, 4], dtype=np.int32),
            "left_c": np.asarray([5, 7, 9, 11], dtype=np.int32),
        },
        n_genes=len(embedding.genes),
        embedding_fingerprint=embedding.fingerprint,
        background_policy="retained_members",
    )
    right = load_data.GeneSetDatabase.from_index_mapping(
        {
            "right_a": np.asarray([1, 3], dtype=np.int32),
            "right_b": np.asarray([5, 6, 7], dtype=np.int32),
            "right_c": np.asarray([2, 8, 12, 14, 16], dtype=np.int32),
        },
        n_genes=len(embedding.genes),
        embedding_fingerprint=embedding.fingerprint,
        background_policy="retained_members",
    )
    ranking = rng.permutation(len(embedding.genes)).astype(np.int32)
    return embedding, left, right, ranking


@pytest.mark.parametrize("seed", [3, 33, 303])
def test_streamed_bma_matches_scalar_reference_for_symmetric_and_asymmetric_axes(
    seed,
):
    embedding, left, right, _ = _scoring_problem(seed)
    similarity = embedding.vectors @ embedding.vectors.T

    for row_database, column_database in ((left, right), (left, left)):
        result = scoring.score_bma_matrix(
            embedding,
            row_database,
            column_database,
            workspace_mb=0.001,
        )
        expected = np.empty_like(result.scores)
        for i in range(len(row_database.terms)):
            for j in range(len(column_database.terms)):
                expected[i, j] = reference_bma(
                    similarity,
                    row_database.members_at(i),
                    column_database.members_at(j),
                )

        np.testing.assert_allclose(result.scores, expected, rtol=2e-6, atol=2e-6)
        assert result.stats.engine == "bestmatch"
        assert result.stats.symmetric_reuse is (
            row_database.fingerprint == column_database.fingerprint
        )


@pytest.mark.parametrize("seed", [7, 71, 701])
def test_standalone_and_indexed_ranked_scoring_match_scalar_reference(seed):
    embedding, _, database, ranking = _scoring_problem(seed)
    similarity = embedding.vectors @ embedding.vectors.T
    expected = np.asarray(
        [
            reference_ranked_es(
                similarity,
                database.members_at(position),
                ranking,
            )
            for position in range(len(database.terms))
        ],
        dtype=np.float32,
    )

    standalone_run = application.run_ranked(
        embedding=embedding,
        database=database,
        ranked_indices=ranking,
        workspace_mb=0.002,
    )
    persistent, _ = bma.gene_to_term_best_match_matrix(
        embedding.vectors,
        database.packed_axis,
        max_workspace_mb=0.001,
    )
    indexed_run = application.run_ranked(
        embedding=embedding,
        database=database,
        ranked_indices=ranking,
        bestmatch=scoring.IndexedBestMatch(
            persistent,
            embedding.fingerprint,
            database.fingerprint,
        ),
        workspace_mb=0.002,
    )
    standalone = standalone_run.scores
    indexed = indexed_run.scores

    np.testing.assert_allclose(standalone.scores, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(indexed.scores, expected, rtol=2e-5, atol=2e-5)
    assert standalone.stats.engine == "bestmatch"
    assert indexed.stats.engine == "indexed"
    assert standalone_run.provenance.extra["bestmatch_source"] == "transient_stream"
    assert indexed_run.provenance.extra["bestmatch_source"] == "persistent_index"


def test_ranked_trace_uses_the_same_score_and_tie_policy_as_production():
    embedding, _, database, ranking = _scoring_problem(707)
    result = scoring.score_ranked(
        embedding,
        database,
        ranking,
        workspace_mb=0.002,
    )

    for position in range(len(database.terms)):
        trace = ranked.ranked_term_trace(
            embedding.vectors,
            database.members_at(position),
            ranking,
            max_workspace_mb=0.002,
        )
        assert trace.score == pytest.approx(
            float(result.scores[position]),
            rel=2e-5,
            abs=2e-5,
        )
        assert trace.running_scores[trace.score_index] == trace.score


def test_bma_calibration_reuses_exact_output_only_when_requested():
    embedding, left, right, _ = _scoring_problem(44)
    result = scoring.score_bma_matrix(embedding, left, right)
    original = result.scores.copy()
    spec = andes_nulls.NullSpec(
        kind="bma",
        iterations=10,
        seed=1,
        sampling="prefix_coupled",
        ddof=1,
        embedding_hash=artifacts.hash_array(embedding.vectors),
        population_hashes=(
            artifacts.hash_array(left.background),
            artifacts.hash_array(right.background),
        ),
    )
    model = andes_nulls.BmaNullModel.from_mapping(
        {
            (int(left_size), int(right_size)): (0.0, 1.0)
            for left_size in np.unique(left.sizes)
            for right_size in np.unique(right.sizes)
        },
        spec,
    )

    copied = scoring.calibrate_bma(
        result,
        model,
        embedding,
        left,
        right,
    )
    assert copied.scores is not result.scores
    np.testing.assert_array_equal(result.scores, original)

    in_place = scoring.calibrate_bma(
        result,
        model,
        embedding,
        left,
        right,
        in_place=True,
    )
    assert in_place.scores is result.scores
    np.testing.assert_allclose(in_place.scores, original)


def test_bma_calibration_rejects_incompatible_background_identity():
    embedding, left, right, _ = _scoring_problem(45)
    exact = scoring.score_bma_matrix(embedding, left, right)
    spec = andes_nulls.NullSpec(
        kind="bma",
        iterations=10,
        seed=1,
        sampling="prefix_coupled",
        ddof=1,
        embedding_hash=embedding.vector_hash,
        population_hashes=(left.background_hash, right.background_hash),
    )
    model = andes_nulls.BmaNullModel.from_mapping(
        {
            (int(left_size), int(right_size)): (0.0, 1.0)
            for left_size in np.unique(left.sizes)
            for right_size in np.unique(right.sizes)
        },
        spec,
    )
    changed_left = load_data.GeneSetDatabase.from_index_mapping(
        {term: left.members_at(position) for position, term in enumerate(left.terms)},
        n_genes=len(embedding.genes),
        embedding_fingerprint=embedding.fingerprint,
        terms=left.terms,
        background=np.union1d(left.background, np.asarray([16], dtype=np.int32)),
        background_policy="explicit_test_background",
    )

    with pytest.raises(ValueError, match="left background is incompatible"):
        scoring.calibrate_bma(
            exact,
            model,
            embedding,
            changed_left,
            right,
        )


def test_ranked_scoring_rejects_an_untyped_bestmatch_matrix():
    embedding, _, database, ranking = _scoring_problem(55)
    bestmatch = np.empty(
        (len(embedding.genes), len(database.terms)),
        dtype=np.float32,
    )
    with pytest.raises(TypeError, match="IndexedBestMatch"):
        scoring.score_ranked(
            embedding,
            database,
            ranking,
            bestmatch=bestmatch,
        )


def test_ranked_scoring_rejects_indices_that_overflow_int32():
    embedding, _, database, _ = _scoring_problem(56)
    wrapped_zero = np.asarray([2**32], dtype=np.uint64)

    with pytest.raises(IndexError, match="out-of-range"):
        scoring.score_ranked(
            embedding,
            database,
            wrapped_zero,
        )


def test_database_identity_rejects_a_different_embedding():
    embedding, left, right, _ = _scoring_problem(66)
    changed = load_data.EmbeddingSpace.from_arrays(
        embedding.vectors[::-1],
        tuple(reversed(embedding.genes)),
        normalize=False,
    )
    with pytest.raises(ValueError, match="different embedding"):
        scoring.score_bma_matrix(changed, left, right)
