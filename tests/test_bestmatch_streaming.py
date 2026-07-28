import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from andes import bma


def _normalized_embedding(seed=7, n_genes=24, dim=8):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        bma.l2_normalize_rows(rng.normal(size=(n_genes, dim)).astype(np.float32))
    )


def test_pack_term_indices_is_sorted_unique_and_offset_addressable():
    packed = bma.pack_term_indices(
        ["a", "b"],
        {
            "a": np.array([4, 1, 4, 3], dtype=np.int32),
            "b": np.array([9, 2], dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(packed.sizes, [3, 2])
    np.testing.assert_array_equal(packed.offsets, [0, 3, 5])
    np.testing.assert_array_equal(packed.flat_indices, [1, 3, 4, 2, 9])


def test_streamed_bestmatch_chunks_match_materialized_reference():
    E = _normalized_embedding()
    terms = ["a", "b", "c"]
    indices = {
        "a": np.array([1, 2, 3], dtype=np.int32),
        "b": np.array([3, 5, 7, 9], dtype=np.int32),
        "c": np.array([11, 12], dtype=np.int32),
    }
    packed = bma.pack_term_indices(terms, indices)
    observed = np.empty((E.shape[0], len(terms)), dtype=np.float32)
    for start, end, chunk in bma.iter_gene_to_term_best_match_chunks(
        E,
        E,
        packed,
        max_workspace_mb=0.0001,
    ):
        observed[:, start:end] = chunk

    blocks = bma.precompute_term_embedding_blocks(E, indices)
    expected, _ = bma.gene_to_term_best_match_matrix(
        E,
        terms,
        blocks,
        max_workspace_mb=0.0001,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_one_shot_bestmatch_restricts_rows_to_source_union():
    E = _normalized_embedding(n_genes=40)
    terms = ["a", "b"]
    indices = {
        "a": np.array([1, 2, 3], dtype=np.int32),
        "b": np.array([3, 4, 5], dtype=np.int32),
    }
    cache = bma.BmaNullBuilder()
    cache.cache[(3, 3)] = (0.0, 1.0)

    observed, stats = bma.score_bma_zscore_matrix_bestmatch(
        E,
        terms,
        terms,
        indices,
        indices,
        cache,
        symmetric=True,
        max_workspace_mb=0.0001,
    )
    expected = bma.score_bma_zscore_matrix(
        E,
        terms,
        terms,
        indices,
        indices,
        cache,
        symmetric=True,
        n_workers=1,
        numba_threshold=0,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)
    assert stats["source_genes1"] == 5
    assert stats["source_genes1"] < E.shape[0]
    assert stats["bestmatch_materialized_mb"] == 0.0
    assert stats["chunk_workspace_mb"] > 0.0
    assert stats["estimated_peak_mb"] >= (
        stats["directed_matrix_mb"] + stats["output_matrix_mb"]
    )
    assert stats["workspace_mb"] == stats["estimated_peak_mb"]


def test_fused_combination_matches_explicit_true_score_normalization():
    directed12 = np.array([[2.0, 3.0], [1.0, 5.0]], dtype=np.float32)
    directed21 = np.array([[4.0, 2.0], [6.0, 3.0]], dtype=np.float32)
    sizes1 = np.array([2, 3], dtype=np.int32)
    sizes2 = np.array([4, 5], dtype=np.int32)
    cache = {
        (2, 4): (0.1, 0.2),
        (2, 5): (0.2, 0.3),
        (3, 4): (0.3, 0.0),
        (3, 5): (0.4, 0.5),
    }

    true_scores = (directed12 + directed21.T) / (
        sizes1[:, None] + sizes2[None, :]
    )
    expected = bma.zscore_matrix_from_cache(true_scores, sizes1, sizes2, cache)
    observed = bma.combine_directed_scores_and_zscore(
        directed12,
        directed21,
        sizes1,
        sizes2,
        cache,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
