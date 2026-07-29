import numpy as np

from andes import bma, data


def _normalized_embedding(seed=7, n_genes=24, dim=8):
    rng = np.random.default_rng(seed)
    return data.EmbeddingSpace.from_arrays(
        rng.normal(size=(n_genes, dim)).astype(np.float32),
        [f"g{i}" for i in range(n_genes)],
    ).vectors


def _packed_axis(terms, indices, n_genes):
    return data.GeneSetDatabase.from_index_mapping(
        indices,
        n_genes=n_genes,
        embedding_fingerprint="test_embedding",
        terms=terms,
        background=np.arange(n_genes, dtype=np.int32),
        background_policy="explicit_test_background",
    ).packed_axis


def test_materialized_bestmatch_matches_per_term_reference_under_tiny_budget():
    E = _normalized_embedding()
    terms = ["a", "b", "c"]
    indices = {
        "a": np.array([1, 2, 3], dtype=np.int32),
        "b": np.array([3, 5, 7, 9], dtype=np.int32),
        "c": np.array([11, 12], dtype=np.int32),
    }
    packed = _packed_axis(terms, indices, E.shape[0])
    observed, _ = bma.gene_to_term_best_match_matrix(
        E,
        packed,
        max_workspace_mb=0.0001,
    )

    expected = np.column_stack([(E @ E[indices[term]].T).max(axis=1) for term in terms])
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_reusable_workspace_respects_budget_unless_one_term_is_too_large():
    lengths = np.asarray([50, *([1] * 80)], dtype=np.int32)
    budget_mb = 0.03

    ranges, workspace_mb = bma.bestmatch_term_ranges(
        lengths,
        source_rows=100,
        embedding_dim=8,
        max_workspace_mb=budget_mb,
        aggregate_rows=10,
    )

    assert len(ranges) > 1
    assert workspace_mb <= budget_mb

    _, irreducible_mb = bma.bestmatch_term_ranges(
        np.asarray([50], dtype=np.int32),
        source_rows=100,
        embedding_dim=8,
        max_workspace_mb=0.001,
        aggregate_rows=10,
    )
    assert irreducible_mb > 0.001


def test_combination_matches_explicit_true_score_normalization_in_place():
    directed12 = np.array([[2.0, 3.0], [1.0, 5.0]], dtype=np.float32)
    directed21 = np.array([[4.0, 2.0], [6.0, 3.0]], dtype=np.float32)
    sizes1 = np.array([2, 3], dtype=np.int32)
    sizes2 = np.array([4, 5], dtype=np.int32)

    true_scores = (directed12 + directed21.T) / (sizes1[:, None] + sizes2[None, :])
    output = np.empty_like(directed12)
    observed = bma.combine_directed_scores(
        directed12,
        directed21,
        sizes1,
        sizes2,
        out=output,
    )

    assert observed is output
    np.testing.assert_allclose(observed, true_scores, rtol=1e-6, atol=1e-6)
