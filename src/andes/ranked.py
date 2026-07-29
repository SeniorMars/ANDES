"""Exact ranked enrichment scoring and prefix-coupled ranked nulls.

For each term, ranked ANDES centers the term's best-match affinity along the
ranked list, cumulatively sums the centered values, and returns the signed
deviation with the largest absolute magnitude. Standalone scoring streams
only ranked rows; indexed scoring gathers those rows from persistent B[g,t].
Both representations use the same reduction and tie policy.

Affinities use float32. Means and cumulative sums use float64.
Absolute extrema within the documented float32-scale tolerance are treated as
tied and the first is retained, preserving the biological sign deterministically.

Ranked nulls draw deterministic partial Fisher-Yates prefixes and reuse each
maximum-size sample for every requested term size. Individual size marginals
remain valid; cross-size estimates are correlated by design. Parallel workers
receive iteration-derived seeds, so worker count does not change sampled
permutations. Float64 Welford combination may differ only at final rounding.
"""

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from numba import jit
from numpy.typing import ArrayLike, NDArray

from . import bma
from .data import PackedTermAxis

RANKED_ES_TIE_RTOL = 8.0 * np.finfo(np.float32).eps
RANKED_ES_TIE_ATOL = 8.0 * np.finfo(np.float32).eps
RANKED_ES_TIE_POLICY = "first_max_abs_with_float32_tolerance"

# Numba kernels


@jit(nopython=True, nogil=True, cache=True)
def _fys_sample_single_into(perm, js, out):
    """Partial Fisher-Yates sample into a pre-allocated buffer.

    perm: int32 (N,), identity [0..N-1] on entry and exit.
    js:   int64 (m,), swap targets; js[j] must lie in [j, N-1].
    out:  int32 (>=m,), receives the sampled local indices in out[:m].
    """
    m = js.shape[0]
    for j in range(m):
        t = js[j]
        tmp = perm[j]
        perm[j] = perm[t]
        perm[t] = tmp
        out[j] = perm[j]
    for j in range(m - 1, -1, -1):
        t = js[j]
        tmp = perm[j]
        perm[j] = perm[t]
        perm[t] = tmp


@jit(nopython=True, nogil=True, cache=True)
def _prefix_es_welford(A3, sizes, means, M2s, counts, col_max_ws):
    """Streaming prefix-max + ES + Welford in one pass over a GEMM block.

    A3:         float32 (b, max_m, L), b iterations of (max_m, L) matmul.
    sizes:      int32   (S,), sorted ascending; max(sizes) <= max_m.
    means:      float64 (S,), Welford running mean, updated in place.
    M2s:        float64 (S,), Welford running M2, updated in place.
    counts:     int64   (S,), Welford running count, updated in place.
    col_max_ws: float32 (L,), scratch buffer for the prefix col-max.

    Maintains ``sum_col_max`` as col-max entries change. Each per-size mean is
    O(1), avoiding another O(L) scan. The ES sweep and Welford update both use
    strict float64 accumulation.
    """
    b = A3.shape[0]
    max_m = A3.shape[1]
    L = A3.shape[2]
    S = sizes.shape[0]

    for bi in range(b):
        # Initialize col-max from row 0 to avoid a -inf*L sentinel sum.
        sum_col_max = 0.0
        for j in range(L):
            v = A3[bi, 0, j]
            col_max_ws[j] = v
            sum_col_max += v
        s_pos = 0

        # Handle prefix length 1 if requested.
        if S > 0 and sizes[0] == 1:
            best_signed = _prefix_es_inner(col_max_ws, sum_col_max, L)
            _welford_update(means, M2s, counts, s_pos, np.float64(best_signed))
            s_pos += 1

        # Rows 1..max_m-1: update col-max and sum incrementally, emit on hit.
        for r in range(1, max_m):
            for j in range(L):
                v = A3[bi, r, j]
                old = col_max_ws[j]
                if v > old:
                    col_max_ws[j] = v
                    sum_col_max += v - old

            m_now = r + 1
            while s_pos < S and sizes[s_pos] == m_now:
                best_signed = _prefix_es_inner(col_max_ws, sum_col_max, L)
                _welford_update(means, M2s, counts, s_pos, np.float64(best_signed))
                s_pos += 1


@jit(nopython=True, nogil=True, cache=True)
def _prefix_es_inner(col_max_ws, sum_col_max, L):
    """One float64 ES sweep with the shared first-near-tie policy."""
    mean_val = sum_col_max / L
    running = 0.0
    max_abs = 0.0
    best_signed = 0.0
    for j in range(L):
        running += col_max_ws[j] - mean_val
        a = abs(running)
        if _ranked_abs_is_larger(a, max_abs):
            max_abs = a
            best_signed = running
    return best_signed


@jit(nopython=True, nogil=True, cache=True, inline="always")
def _ranked_abs_is_larger(candidate, current):
    """Return whether a deviation beats the first maximum beyond tolerance."""
    if current == 0.0:
        return candidate > 0.0
    scale = max(1.0, candidate, current)
    tolerance = RANKED_ES_TIE_ATOL + RANKED_ES_TIE_RTOL * scale
    return candidate > current + tolerance


@jit(nopython=True, nogil=True, cache=True)
def _welford_update(means, M2s, counts, i, x):
    """Single-sample Welford update in float64, no fastmath."""
    counts[i] += 1
    c = counts[i]
    delta = x - means[i]
    means[i] += delta / c
    delta2 = x - means[i]
    M2s[i] += delta * delta2


@jit(nopython=True, nogil=True, cache=True)
def _orders_are_permutations(orders, seen):
    """Validate permutation columns in O(rows * columns) time and O(rows) space."""
    n_rows = orders.shape[0]
    for column in range(orders.shape[1]):
        marker = column + 1
        for row in range(n_rows):
            value = orders[row, column]
            if value < 0 or value >= n_rows or seen[value] == marker:
                return False
            seen[value] = marker
    return True


# User-facing scoring functions


def compute_ranked_emb(E_unit, ranked_list_idx):
    """Extract the embedding rows for a ranked gene list.

    Returns a C-contiguous float32 array of shape (L, d) where L is the number
    of ranked genes and d is the embedding dimension. Call once per ranked
    list; pass the result (or its transpose) to the scoring functions.
    """
    return np.ascontiguousarray(
        E_unit[np.asarray(ranked_list_idx, dtype=np.int32)], dtype=np.float32
    )


@jit(nopython=True, nogil=True, cache=True)
def _es_scores_from_ranked_bestmatch(
    best_by_rank,
    scores_out,
    mean_ws,
    running_ws,
    max_abs_ws,
):
    """Score columns of an indexed ranked best-match block.

    ``best_by_rank`` is row-major ``(ranked_genes, terms)``. Traversing ranks
    outside and terms inside keeps both passes contiguous while the three
    term-sized workspaces avoid centered and cumulative matrix temporaries.
    """
    ranked_len = best_by_rank.shape[0]
    n_terms = best_by_rank.shape[1]

    for term_i in range(n_terms):
        mean_ws[term_i] = 0.0
        running_ws[term_i] = 0.0
        max_abs_ws[term_i] = 0.0
        scores_out[term_i] = np.float32(0.0)

    for rank_i in range(ranked_len):
        for term_i in range(n_terms):
            mean_ws[term_i] += best_by_rank[rank_i, term_i]
    for term_i in range(n_terms):
        mean_ws[term_i] /= ranked_len

    for rank_i in range(ranked_len):
        for term_i in range(n_terms):
            running_ws[term_i] += best_by_rank[rank_i, term_i] - mean_ws[term_i]
            absolute = abs(running_ws[term_i])
            if _ranked_abs_is_larger(absolute, max_abs_ws[term_i]):
                max_abs_ws[term_i] = absolute
                scores_out[term_i] = running_ws[term_i]


@jit(nopython=True, nogil=True, cache=True)
def _count_ranked_exceedances(
    best_by_expression,
    orders,
    observed_abs,
    counts,
    means,
    running,
    max_abs,
):
    """Count empirical exceedances for a block of indexed terms.

    Every order is a permutation of the same expression rows, so term means
    are invariant across phenotype permutations. Computing those means once
    removes a complete pass over the indexed values for every permutation.
    """
    n_rows = best_by_expression.shape[0]
    n_terms = best_by_expression.shape[1]
    n_permutations = orders.shape[1]

    for term_i in range(n_terms):
        total = 0.0
        for row_i in range(n_rows):
            total += best_by_expression[row_i, term_i]
        means[term_i] = total / n_rows

    for permutation_i in range(n_permutations):
        for term_i in range(n_terms):
            running[term_i] = 0.0
            max_abs[term_i] = 0.0

        for rank_i in range(n_rows):
            row_i = orders[rank_i, permutation_i]
            for term_i in range(n_terms):
                running[term_i] += best_by_expression[row_i, term_i] - means[term_i]
                absolute = abs(running[term_i])
                if _ranked_abs_is_larger(absolute, max_abs[term_i]):
                    max_abs[term_i] = absolute

        for term_i in range(n_terms):
            if max_abs[term_i] >= observed_abs[term_i]:
                counts[term_i] += 1


@dataclass(frozen=True, slots=True)
class RankedTrace:
    """Exact per-rank evidence for one ranked ANDES term."""

    best_match_scores: NDArray[np.float32]
    best_member_positions: NDArray[np.int32]
    centered_scores: NDArray[np.float64]
    running_scores: NDArray[np.float64]
    score_index: int
    score: float


def ranked_term_trace(
    embedding: ArrayLike,
    term_indices: ArrayLike,
    ranked_indices: ArrayLike,
    *,
    max_workspace_mb: float | None = 128,
):
    """Return the exact ranked trace using the shared accumulation policy."""
    vectors = np.asarray(embedding, dtype=np.float32)
    members = np.asarray(term_indices)
    ranking = np.asarray(ranked_indices)
    if vectors.ndim != 2:
        raise ValueError("embedding must be two-dimensional")
    if members.ndim != 1 or members.size == 0:
        raise ValueError("term_indices must be a non-empty vector")
    if ranking.ndim != 1 or ranking.size == 0:
        raise ValueError("ranked_indices must be a non-empty vector")
    if members.dtype.kind not in "iu" or ranking.dtype.kind not in "iu":
        raise TypeError("term_indices and ranked_indices must contain integers")
    members = members.astype(np.int32, copy=False)
    ranking = ranking.astype(np.int32, copy=False)
    if int(members.min()) < 0 or int(members.max()) >= vectors.shape[0]:
        raise IndexError("term_indices contains an out-of-range embedding row")
    if int(ranking.min()) < 0 or int(ranking.max()) >= vectors.shape[0]:
        raise IndexError("ranked_indices contains an out-of-range embedding row")

    ranked_vectors = np.ascontiguousarray(vectors[ranking], dtype=np.float32)
    ranked_count = int(ranking.size)
    budget_bytes = (
        0 if max_workspace_mb is None else max(0, int(float(max_workspace_mb) * 1e6))
    )
    fixed_bytes = ranked_count * (
        2 * np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize
    )
    available = max(0, budget_bytes - fixed_bytes)
    bytes_per_member = max(
        np.dtype(np.float32).itemsize,
        ranked_count * np.dtype(np.float32).itemsize,
    )
    members_per_block = (
        int(members.size)
        if budget_bytes == 0
        else min(int(members.size), max(1, available // bytes_per_member))
    )

    best_scores = np.full(ranked_count, -np.inf, dtype=np.float32)
    best_positions = np.zeros(ranked_count, dtype=np.int32)
    columns = np.arange(ranked_count)
    ranked_transpose = np.ascontiguousarray(ranked_vectors.T, dtype=np.float32)
    for start in range(0, int(members.size), members_per_block):
        end = min(int(members.size), start + members_per_block)
        similarities = vectors[members[start:end]] @ ranked_transpose
        local_positions = similarities.argmax(axis=0).astype(np.int32)
        local_scores = similarities[local_positions, columns]
        better = local_scores > best_scores
        best_scores[better] = local_scores[better]
        best_positions[better] = start + local_positions[better]

    centered = best_scores.astype(np.float64)
    centered -= centered.mean(dtype=np.float64)
    running = np.cumsum(centered, dtype=np.float64)
    score_index = 0
    max_abs = 0.0
    for position, value in enumerate(running):
        candidate = abs(float(value))
        if max_abs == 0.0:
            is_larger = candidate > 0.0
        else:
            scale = max(1.0, candidate, max_abs)
            tolerance = RANKED_ES_TIE_ATOL + RANKED_ES_TIE_RTOL * scale
            is_larger = candidate > max_abs + tolerance
        if is_larger:
            max_abs = candidate
            score_index = position
    return RankedTrace(
        best_match_scores=best_scores,
        best_member_positions=best_positions,
        centered_scores=centered,
        running_scores=running,
        score_index=score_index,
        score=float(running[score_index]),
    )


def score_terms_bestmatch_exact(
    E_unit: ArrayLike,
    packed: PackedTermAxis,
    ranked_emb: ArrayLike,
    max_workspace_mb: float | None = 128,
    *,
    return_stats=False,
):
    """Return aligned exact ranked scores from streamed best-match columns.

    Only rows belonging to the ranked list are constructed. Each term chunk
    is reduced to its gene-to-term best matches and consumed immediately by
    the ranked ES kernel, so the full gene-by-term index is never allocated.
    """
    embeddings = np.asarray(E_unit, dtype=np.float32)
    ranked_emb = np.ascontiguousarray(ranked_emb, dtype=np.float32)
    if embeddings.ndim != 2 or ranked_emb.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if embeddings.shape[1] != ranked_emb.shape[1]:
        raise ValueError("ranked and full embedding dimensions differ")
    if ranked_emb.shape[0] == 0:
        raise ValueError("ranked_emb must contain at least one gene")

    n_terms = len(packed.terms)
    true_scores = np.empty(n_terms, dtype=np.float32)
    requested_bytes = (
        0 if max_workspace_mb is None else max(0, int(float(max_workspace_mb) * 1e6))
    )
    if n_terms == 0:
        stats = {
            "workspace_bytes": 0,
            "requested_workspace_bytes": requested_bytes,
            "term_chunk_size": 0,
        }
        return (true_scores, stats) if return_stats else true_scores

    ranges, compute_peak_mb = bma.bestmatch_term_ranges(
        packed.sizes,
        ranked_emb.shape[0],
        ranked_emb.shape[1],
        max_workspace_mb,
        aggregate_rows=6,
    )
    peak_bytes = int(compute_peak_mb * 1e6) + int(true_scores.nbytes)
    largest_chunk = max(end - start for start, end in ranges)

    chunks = bma._iter_gene_to_term_best_match_chunks(
        ranked_emb,
        embeddings,
        packed,
        max_workspace_mb=max_workspace_mb,
        aggregate_rows=6,
    )
    for start, end, best_by_rank in chunks:
        width = end - start
        means = np.empty(width, dtype=np.float64)
        running = np.empty(width, dtype=np.float64)
        max_abs = np.empty(width, dtype=np.float64)
        _es_scores_from_ranked_bestmatch(
            best_by_rank,
            true_scores[start:end],
            means,
            running,
            max_abs,
        )

    if return_stats:
        return true_scores, {
            "workspace_bytes": peak_bytes,
            "requested_workspace_bytes": requested_bytes,
            "term_chunk_size": int(largest_chunk),
        }
    return true_scores


def score_terms_indexed(
    bestmatch: ArrayLike,
    ranked_idx: ArrayLike,
    max_workspace_mb: float | None = 128,
):
    """Return exact ranked scores from a persistent gene-to-term index.

    Parameters
    ----------
    bestmatch : array-like, shape (embedding_genes, terms), float32
        ``bestmatch[g, t] = max(sim(g, x) for x in term_t)``.
    ranked_idx : array-like, shape (ranked_genes,)
        Embedding row indices in ranked order.
    max_workspace_mb : float
        Target for all live scoring arrays. Terms are streamed in column
        chunks. One term is always processed even when it exceeds the target.

    Returns
    -------
    true_scores : ndarray, shape (terms,), float32
    stats : dict
        Planned peak workspace and chosen term chunk size.

    Each indexed column is centered over ranked positions, cumulatively
    summed, and reduced at the first maximum absolute deviation.
    """
    B = np.asarray(bestmatch)
    if B.ndim != 2:
        raise ValueError("bestmatch must have shape (embedding_genes, terms)")
    if B.dtype.kind != "f":
        raise TypeError("bestmatch must contain real floating-point values")

    ranked_idx = np.asarray(ranked_idx)
    if ranked_idx.ndim != 1 or ranked_idx.size == 0:
        raise ValueError("ranked_idx must be a non-empty one-dimensional array")
    if ranked_idx.dtype.kind not in "iu":
        raise TypeError("ranked_idx must contain integer embedding rows")
    if int(ranked_idx.min()) < 0 or int(ranked_idx.max()) >= B.shape[0]:
        raise IndexError("ranked_idx contains an out-of-range embedding row")
    ranked_idx = ranked_idx.astype(np.int32, copy=False)

    n_terms = B.shape[1]
    ranked_len = ranked_idx.size
    output_bytes = n_terms * np.dtype(np.float32).itemsize
    bytes_per_term = (ranked_len + 6) * np.dtype(np.float32).itemsize
    if max_workspace_mb is None or float(max_workspace_mb) <= 0.0:
        terms_per_chunk = max(1, n_terms)
        budget_bytes = 0
    else:
        budget_bytes = max(4, int(float(max_workspace_mb) * 1e6))
        available = max(0, budget_bytes - output_bytes)
        terms_per_chunk = max(1, available // bytes_per_term)
        terms_per_chunk = min(terms_per_chunk, max(1, n_terms))

    true_scores = np.empty(n_terms, dtype=np.float32)
    peak_bytes = output_bytes
    for start in range(0, n_terms, terms_per_chunk):
        end = min(start + terms_per_chunk, n_terms)
        width = end - start
        best_by_rank = np.empty(
            (ranked_len, width),
            dtype=np.float32,
            order="C",
        )
        for output_row, source_row in enumerate(ranked_idx):
            best_by_rank[output_row] = B[int(source_row), start:end]
        score_buf = true_scores[start:end]
        mean_ws = np.empty(width, dtype=np.float64)
        running_ws = np.empty(width, dtype=np.float64)
        max_abs_ws = np.empty(width, dtype=np.float64)
        peak_bytes = max(
            peak_bytes,
            output_bytes
            + int(best_by_rank.nbytes)
            + int(mean_ws.nbytes)
            + int(running_ws.nbytes)
            + int(max_abs_ws.nbytes),
        )
        _es_scores_from_ranked_bestmatch(
            best_by_rank,
            score_buf,
            mean_ws,
            running_ws,
            max_abs_ws,
        )

    return (
        true_scores,
        {
            "workspace_mb": peak_bytes / 1e6,
            "workspace_bytes": int(peak_bytes),
            "requested_workspace_bytes": int(budget_bytes),
            "term_chunk_size": int(terms_per_chunk),
        },
    )


def count_indexed_ranked_exceedances(
    bestmatch,
    row_to_embedding,
    orders,
    observed_scores,
    *,
    counts=None,
    max_workspace_mb: float | None = 128,
):
    """Accumulate empirical exceedances for batched phenotype rankings.

    ``orders[:, p]`` must be a permutation of all expression rows. Each term
    chunk is gathered once and reused across the permutation batch. The
    function retains only the exceedance counts.
    """
    bestmatch = np.asanyarray(bestmatch)
    if bestmatch.ndim != 2 or bestmatch.dtype.kind != "f":
        raise TypeError("bestmatch must be a two-dimensional floating array")

    row_to_embedding_input = np.asarray(row_to_embedding)
    if row_to_embedding_input.ndim != 1 or row_to_embedding_input.size == 0:
        raise ValueError("row_to_embedding must be a non-empty vector")
    if row_to_embedding_input.dtype.kind not in "iu":
        raise TypeError("row_to_embedding must contain integer rows")
    if (
        int(row_to_embedding_input.min()) < 0
        or int(row_to_embedding_input.max()) >= bestmatch.shape[0]
    ):
        raise IndexError("row_to_embedding contains an out-of-range row")
    row_to_embedding = row_to_embedding_input.astype(np.int32, copy=False)
    row_mapping_copy_bytes = (
        0
        if np.shares_memory(row_to_embedding, row_to_embedding_input)
        else int(row_to_embedding.nbytes)
    )

    orders_input = np.asarray(orders)
    expected_order_shape = (
        row_to_embedding.size,
        orders_input.shape[1] if orders_input.ndim == 2 else 0,
    )
    if orders_input.ndim != 2 or orders_input.shape != expected_order_shape:
        raise ValueError("orders must have one row per expression gene")
    if orders_input.dtype.kind not in "iu":
        raise TypeError("orders must contain integer row positions")
    if orders_input.size and (
        int(orders_input.min()) < 0 or int(orders_input.max()) >= row_to_embedding.size
    ):
        raise ValueError("every order column must be a permutation of expression rows")
    orders = np.ascontiguousarray(orders_input, dtype=np.int32)
    order_copy_bytes = (
        0 if np.shares_memory(orders, orders_input) else int(orders.nbytes)
    )
    validation_ws = np.zeros(row_to_embedding.size, dtype=np.int64)
    validation_peak_bytes = (
        row_mapping_copy_bytes + order_copy_bytes + int(validation_ws.nbytes)
    )
    if not _orders_are_permutations(orders, validation_ws):
        raise ValueError("every order column must be a permutation of expression rows")
    del validation_ws

    observed = np.asarray(observed_scores)
    if observed.ndim != 1 or observed.size != bestmatch.shape[1]:
        raise ValueError("observed_scores must contain one value per indexed term")
    if observed.dtype.kind != "f":
        raise TypeError("observed_scores must contain real floating-point values")
    observed_abs = np.empty(observed.size, dtype=np.float32)
    np.absolute(observed, out=observed_abs, casting="unsafe")

    if counts is None:
        counts = np.zeros(bestmatch.shape[1], dtype=np.int64)
    else:
        counts = np.asarray(counts)
        if counts.shape != observed.shape or counts.dtype != np.int64:
            raise TypeError("counts must be an int64 vector matching observed_scores")
        if np.any(counts < 0):
            raise ValueError("counts must not contain negative values")

    n_rows = row_to_embedding.size
    n_terms = bestmatch.shape[1]
    output_bytes = int(counts.nbytes)
    fixed_workspace_bytes = (
        output_bytes
        + row_mapping_copy_bytes
        + order_copy_bytes
        + int(observed_abs.nbytes)
    )
    if max_workspace_mb is None or float(max_workspace_mb) <= 0.0:
        budget_bytes = 0
    else:
        budget_bytes = max(4, int(float(max_workspace_mb) * 1e6))
    if n_terms == 0:
        return counts, {
            "workspace_bytes": max(validation_peak_bytes, fixed_workspace_bytes),
            "requested_workspace_bytes": budget_bytes,
            "term_chunk_size": 0,
            "fixed_workspace_bytes": fixed_workspace_bytes,
            "row_mapping_copy_bytes": row_mapping_copy_bytes,
            "order_copy_bytes": order_copy_bytes,
            "validation_workspace_bytes": int(row_to_embedding.size * 8),
        }
    bytes_per_term = int((n_rows + 6) * np.dtype(np.float32).itemsize)
    if budget_bytes == 0:
        terms_per_chunk = n_terms
    else:
        available = max(0, budget_bytes - fixed_workspace_bytes)
        terms_per_chunk = min(n_terms, max(1, available // bytes_per_term))
    peak_bytes = max(validation_peak_bytes, fixed_workspace_bytes)

    for start in range(0, n_terms, terms_per_chunk):
        end = min(start + terms_per_chunk, n_terms)
        width = end - start
        best_by_expression = np.empty(
            (n_rows, width),
            dtype=np.float32,
            order="C",
        )
        np.take(
            bestmatch[:, start:end],
            row_to_embedding,
            axis=0,
            out=best_by_expression,
        )
        means = np.empty(width, dtype=np.float64)
        running = np.empty(width, dtype=np.float64)
        max_abs = np.empty(width, dtype=np.float64)
        peak_bytes = max(
            peak_bytes,
            fixed_workspace_bytes
            + int(best_by_expression.nbytes)
            + int(means.nbytes)
            + int(running.nbytes)
            + int(max_abs.nbytes),
        )
        _count_ranked_exceedances(
            best_by_expression,
            orders,
            observed_abs[start:end],
            counts[start:end],
            means,
            running,
            max_abs,
        )

    return counts, {
        "workspace_bytes": int(peak_bytes),
        "requested_workspace_bytes": int(budget_bytes),
        "term_chunk_size": int(terms_per_chunk),
        "fixed_workspace_bytes": int(fixed_workspace_bytes),
        "row_mapping_copy_bytes": int(row_mapping_copy_bytes),
        "order_copy_bytes": int(order_copy_bytes),
        "validation_workspace_bytes": int(row_to_embedding.size * 8),
    }


@dataclass(frozen=True, slots=True)
class RankedNullPlan:
    """Resolved process and memory policy for ranked-null construction."""

    strategy: str
    workers: int
    workspace_bytes_per_worker: int
    shared_workspace_bytes: int
    total_workspace_bytes: int
    minimum_workspace_bytes_per_worker: int


def _ranked_null_workspace_bytes(
    batch_iterations,
    *,
    max_size,
    ranked_length,
    embedding_dimensions,
    population_size,
    n_sizes,
):
    """Estimate live numerical workspace for one ranked-null worker."""
    batch_iterations = int(batch_iterations)
    max_size = int(max_size)
    ranked_length = int(ranked_length)
    embedding_dimensions = int(embedding_dimensions)
    population_size = int(population_size)
    n_sizes = int(n_sizes)
    float32_bytes = np.dtype(np.float32).itemsize
    int32_bytes = np.dtype(np.int32).itemsize
    int64_bytes = np.dtype(np.int64).itemsize
    float64_bytes = np.dtype(np.float64).itemsize

    fixed = (
        ranked_length * float32_bytes
        + population_size * int32_bytes
        + 2 * max_size * int64_bytes
        + n_sizes * (2 * float64_bytes + int64_bytes)
    )
    per_iteration = max_size * (
        ranked_length * float32_bytes
        + embedding_dimensions * float32_bytes
        + int32_bytes
    )
    return int(fixed + batch_iterations * per_iteration)


def _ranked_null_lookup_workspace_bytes(
    batch_iterations,
    *,
    max_size,
    ranked_length,
    population_size,
    n_sizes,
):
    """Estimate private workspace when similarities are precomputed."""
    batch_iterations = int(batch_iterations)
    max_size = int(max_size)
    ranked_length = int(ranked_length)
    population_size = int(population_size)
    n_sizes = int(n_sizes)
    fixed = (
        ranked_length * np.dtype(np.float32).itemsize
        + population_size * np.dtype(np.int32).itemsize
        + 2 * max_size * np.dtype(np.int64).itemsize
        + n_sizes * (2 * np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize)
    )
    per_iteration = max_size * (
        ranked_length * np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize
    )
    return int(fixed + batch_iterations * per_iteration)


def plan_ranked_null_runtime(
    *,
    requested_workers,
    total_workspace_bytes,
    iterations,
    max_size,
    ranked_length,
    embedding_dimensions,
    population_size,
    n_sizes,
    cpu_count=None,
):
    """Choose a null worker count within one total memory budget.

    ``requested_workers=0`` selects up to eight workers, bounded by CPUs,
    iterations, and the memory needed for at least one iteration per worker.
    A positive request is capped by the iteration count and must fit the
    workspace budget.
    """
    requested_workers = int(requested_workers)
    total_workspace_bytes = int(total_workspace_bytes)
    iterations = int(iterations)
    if requested_workers < 0:
        raise ValueError("requested_workers must be non-negative")
    if total_workspace_bytes < 1:
        raise ValueError("total_workspace_bytes must be positive")
    if iterations < 2:
        raise ValueError("iterations must be at least two when ddof=1")
    dimensions = {
        "max_size": max_size,
        "ranked_length": ranked_length,
        "embedding_dimensions": embedding_dimensions,
        "population_size": population_size,
        "n_sizes": n_sizes,
    }
    invalid = [name for name, value in dimensions.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"{invalid[0]} must be positive")
    if cpu_count is not None and int(cpu_count) < 1:
        raise ValueError("cpu_count must be positive")

    direct_minimum = _ranked_null_workspace_bytes(
        1,
        max_size=max_size,
        ranked_length=ranked_length,
        embedding_dimensions=embedding_dimensions,
        population_size=population_size,
        n_sizes=n_sizes,
    )
    shared_similarity_bytes = (
        int(population_size) * int(ranked_length) * np.dtype(np.float32).itemsize
    )
    lookup_minimum = _ranked_null_lookup_workspace_bytes(
        1,
        max_size=max_size,
        ranked_length=ranked_length,
        population_size=population_size,
        n_sizes=n_sizes,
    )
    # The shared matrix has a fixed construction and mmap cost. Empirical
    # crossover testing on the reference workload favors it only after the
    # direct null would revisit the background by roughly an order of
    # magnitude; below that point, process startup and page traffic dominate.
    precompute_reuses_rows = int(iterations) * int(max_size) >= 12 * int(
        population_size
    )
    precomputed_worker_limit = (
        max(0, total_workspace_bytes - shared_similarity_bytes) // lookup_minimum
    )
    detected_cpus = os.cpu_count() or 1
    available_cpus = max(
        1,
        detected_cpus if cpu_count is None else int(cpu_count),
    )
    target_workers = (
        min(8, available_cpus, iterations)
        if requested_workers == 0
        else min(requested_workers, iterations)
    )
    use_precomputed = (
        precompute_reuses_rows and precomputed_worker_limit >= target_workers
    )
    strategy = "precomputed_similarity" if use_precomputed else "direct_gemm"
    shared_bytes = shared_similarity_bytes if use_precomputed else 0
    minimum = lookup_minimum if use_precomputed else direct_minimum
    private_total = total_workspace_bytes - shared_bytes
    memory_worker_limit = private_total // minimum
    if memory_worker_limit < 1:
        raise ValueError(
            "ranked-null workspace is too small for one iteration: "
            f"need at least {(shared_bytes + minimum) / 1e6:.1f} MB"
        )

    if requested_workers == 0:
        workers = min(target_workers, memory_worker_limit)
    else:
        workers = target_workers
        if workers > memory_worker_limit:
            raise ValueError(
                f"{workers} ranked-null workers need at least "
                f"{(shared_bytes + workers * minimum) / 1e6:.1f} MB "
                "total workspace"
            )

    per_worker = private_total // workers
    return RankedNullPlan(
        strategy=strategy,
        workers=int(workers),
        workspace_bytes_per_worker=int(per_worker),
        shared_workspace_bytes=int(shared_bytes),
        total_workspace_bytes=int(total_workspace_bytes),
        minimum_workspace_bytes_per_worker=int(minimum),
    )


# Worker

# Worker-process globals set by _init_worker via ProcessPoolExecutor.
_worker_e_pop: NDArray[np.float32] | None = None
_worker_ranked_emb_t: NDArray[np.float32] | None = None
_worker_workspace_bytes: int | None = None
_worker_similarities: NDArray[np.float32] | None = None
_worker_threadpool_limiter: object | None = None


def _init_worker(
    e_pop_path,
    ranked_path,
    similarity_path,
    blas_threads,
    worker_workspace_bytes,
):
    """Run once in each ProcessPoolExecutor worker.

    Loads E_pop and ranked_emb_T (stored as (d, L) for matmul without .T) from
    temporary ``.npy`` files written by ``build_ranked_null_parallel``.
    Uses mmap_mode='r' so the OS can share physical pages across workers when
    the arrays fit in the page cache. The threadpoolctl controller stays alive
    to enforce the worker's BLAS limit.
    """
    from threadpoolctl import threadpool_limits

    global _worker_e_pop, _worker_ranked_emb_t, _worker_workspace_bytes
    global _worker_similarities, _worker_threadpool_limiter
    _worker_threadpool_limiter = threadpool_limits(
        limits=int(blas_threads),
        user_api="blas",
    )
    _worker_e_pop = np.load(e_pop_path, mmap_mode="r")
    _worker_ranked_emb_t = np.load(ranked_path, mmap_mode="r")
    _worker_similarities = (
        None if not similarity_path else np.load(similarity_path, mmap_mode="r")
    )
    _worker_workspace_bytes = int(worker_workspace_bytes)


def _draw_ranked_null_samples(
    iter_indices,
    *,
    offset,
    batch_size,
    master_seed,
    population_size,
    lower_bounds,
    permutation,
    samples,
):
    """Fill a partial Fisher-Yates sample batch deterministically."""
    for batch_i in range(batch_size):
        iteration = int(iter_indices[offset + batch_i])
        rng = np.random.default_rng(
            np.random.SeedSequence([int(master_seed), iteration])
        )
        swaps = rng.integers(lower_bounds, population_size, dtype=np.int64)
        _fys_sample_single_into(permutation, swaps, samples[batch_i])


def _compute_mc_stats(
    iter_indices,
    sizes_arr,
    master_seed,
    e_pop,
    ranked_emb_T,
    worker_workspace_bytes,
):
    """Compute one deterministic prefix-null chunk from explicit arrays."""
    sizes = np.asarray(sizes_arr, dtype=np.int32)
    S = len(sizes)
    max_m = int(sizes[-1])

    N_pop = e_pop.shape[0]
    L = ranked_emb_T.shape[1]
    ite_chunk = len(iter_indices)

    fixed_bytes = _ranked_null_workspace_bytes(
        0,
        max_size=max_m,
        ranked_length=L,
        embedding_dimensions=e_pop.shape[1],
        population_size=N_pop,
        n_sizes=S,
    )
    per_iteration_bytes = (
        _ranked_null_workspace_bytes(
            1,
            max_size=max_m,
            ranked_length=L,
            embedding_dimensions=e_pop.shape[1],
            population_size=N_pop,
            n_sizes=S,
        )
        - fixed_bytes
    )
    minimum_bytes = fixed_bytes + per_iteration_bytes
    if int(worker_workspace_bytes) < minimum_bytes:
        raise ValueError(
            "ranked-null workspace is too small for one iteration: "
            f"need at least {minimum_bytes / 1e6:.1f} MB"
        )
    b = min(
        ite_chunk,
        (int(worker_workspace_bytes) - fixed_bytes) // per_iteration_bytes,
    )

    A_buf = np.empty((b * max_m, L), dtype=np.float32)
    X_buf = np.empty((b * max_m, e_pop.shape[1]), dtype=np.float32)
    xi = np.empty((b, max_m), dtype=np.int32)
    col_max_ws = np.empty(L, dtype=np.float32)

    means = np.zeros(S, dtype=np.float64)
    M2s = np.zeros(S, dtype=np.float64)
    counts = np.zeros(S, dtype=np.int64)

    perm = np.arange(N_pop, dtype=np.int32)
    m_low = np.arange(max_m, dtype=np.int64)

    done = 0
    while done < ite_chunk:
        b_act = min(b, ite_chunk - done)

        # Per-iteration deterministic seeds make results independent of worker
        # count and numerical batch size.
        _draw_ranked_null_samples(
            iter_indices,
            offset=done,
            batch_size=b_act,
            master_seed=master_seed,
            population_size=N_pop,
            lower_bounds=m_low,
            permutation=perm,
            samples=xi,
        )

        # Gather and multiply into persistent buffers. The workspace contract
        # includes the per-batch arrays.
        active_rows = b_act * max_m
        np.take(
            e_pop,
            xi[:b_act].ravel(),
            axis=0,
            out=X_buf[:active_rows],
        )
        np.matmul(
            X_buf[:active_rows],
            ranked_emb_T,
            out=A_buf[:active_rows],
        )

        A3 = A_buf[:active_rows].reshape(b_act, max_m, L)
        _prefix_es_welford(A3, sizes, means, M2s, counts, col_max_ws)

        done += b_act

    return means, M2s, counts


def _compute_mc_stats_precomputed(
    iter_indices,
    sizes_arr,
    master_seed,
    similarities,
    workspace_bytes,
):
    """Compute a prefix null from a shared background-to-ranking matrix."""
    sizes = np.asarray(sizes_arr, dtype=np.int32)
    n_sizes = len(sizes)
    max_m = int(sizes[-1])
    population_size, ranked_length = similarities.shape
    iteration_count = len(iter_indices)

    fixed_bytes = _ranked_null_lookup_workspace_bytes(
        0,
        max_size=max_m,
        ranked_length=ranked_length,
        population_size=population_size,
        n_sizes=n_sizes,
    )
    per_iteration_bytes = (
        _ranked_null_lookup_workspace_bytes(
            1,
            max_size=max_m,
            ranked_length=ranked_length,
            population_size=population_size,
            n_sizes=n_sizes,
        )
        - fixed_bytes
    )
    minimum_bytes = fixed_bytes + per_iteration_bytes
    if int(workspace_bytes) < minimum_bytes:
        raise ValueError(
            "ranked-null workspace is too small for one iteration: "
            f"need at least {minimum_bytes / 1e6:.1f} MB"
        )
    batch_size = min(
        iteration_count,
        (int(workspace_bytes) - fixed_bytes) // per_iteration_bytes,
    )

    score_buffer = np.empty(
        (batch_size * max_m, ranked_length),
        dtype=np.float32,
    )
    samples = np.empty((batch_size, max_m), dtype=np.int32)
    col_max = np.empty(ranked_length, dtype=np.float32)
    means = np.zeros(n_sizes, dtype=np.float64)
    M2s = np.zeros(n_sizes, dtype=np.float64)
    counts = np.zeros(n_sizes, dtype=np.int64)
    permutation = np.arange(population_size, dtype=np.int32)
    lower_bounds = np.arange(max_m, dtype=np.int64)

    done = 0
    while done < iteration_count:
        active_batch = min(batch_size, iteration_count - done)
        _draw_ranked_null_samples(
            iter_indices,
            offset=done,
            batch_size=active_batch,
            master_seed=master_seed,
            population_size=population_size,
            lower_bounds=lower_bounds,
            permutation=permutation,
            samples=samples,
        )
        active_rows = active_batch * max_m
        np.take(
            similarities,
            samples[:active_batch].ravel(),
            axis=0,
            out=score_buffer[:active_rows],
        )
        _prefix_es_welford(
            score_buffer[:active_rows].reshape(
                active_batch,
                max_m,
                ranked_length,
            ),
            sizes,
            means,
            M2s,
            counts,
            col_max,
        )
        done += active_batch

    return means, M2s, counts


def _compute_mc_chunk(args):
    """Worker adapter for deterministic prefix-null chunks."""
    if (
        _worker_e_pop is None
        or _worker_ranked_emb_t is None
        or _worker_workspace_bytes is None
    ):
        raise RuntimeError("ranked null worker was not initialized")
    iter_indices, sizes_arr, master_seed = args
    if _worker_similarities is not None:
        return _compute_mc_stats_precomputed(
            iter_indices,
            sizes_arr,
            master_seed,
            _worker_similarities,
            _worker_workspace_bytes,
        )
    return _compute_mc_stats(
        iter_indices,
        sizes_arr,
        master_seed,
        _worker_e_pop,
        _worker_ranked_emb_t,
        _worker_workspace_bytes,
    )


def _combine_welford(stats_list):
    """Merge per-worker Welford statistics into a single aggregate.

    stats_list : list of (means, M2s, counts)
        One tuple per worker.
    Each worker covers a disjoint range of iteration indices over the same
    size set S, so the parallel Welford combination formula applies exactly.
    Addition order may differ across runs with different worker counts, giving
    results that agree up to the last few ULPs of float64.
    """
    S = len(stats_list[0][0])
    agg_mean = np.zeros(S, dtype=np.float64)
    agg_m2 = np.zeros(S, dtype=np.float64)
    agg_count = np.zeros(S, dtype=np.int64)

    for means, m2s, counts in stats_list:
        for i in range(S):
            c_b = int(counts[i])
            if c_b == 0:
                continue
            c_a = int(agg_count[i])
            if c_a == 0:
                agg_mean[i] = means[i]
                agg_m2[i] = m2s[i]
                agg_count[i] = c_b
            else:
                c_ab = c_a + c_b
                delta = means[i] - agg_mean[i]
                agg_mean[i] += delta * c_b / c_ab
                agg_m2[i] += m2s[i] + (delta * delta) * c_a * c_b / c_ab
                agg_count[i] = c_ab

    return agg_mean, agg_m2, agg_count


# Numerical null builder


def _prepare_ranked_null_request(
    gene_set_sizes,
    existing,
    *,
    population_size,
    iterations,
    seed,
):
    """Validate one numerical request and return its missing sizes."""
    iterations = int(iterations)
    seed = int(seed)
    if iterations < 2:
        raise ValueError("iterations must be at least two when ddof=1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    sizes = sorted({int(size) for size in gene_set_sizes})
    cache = dict(existing or {})
    if not sizes:
        return cache, [], iterations, seed
    if sizes[0] < 1:
        raise ValueError(f"gene set sizes must be positive, got {sizes[0]}")
    if sizes[-1] > int(population_size):
        raise ValueError(
            f"max gene set size {sizes[-1]} exceeds "
            f"population size {int(population_size)}"
        )
    return (
        cache,
        [size for size in sizes if size not in cache],
        iterations,
        seed,
    )


def _store_ranked_null_statistics(cache, sizes, means, second_moments, counts):
    """Add completed float64 Welford statistics to a result mapping."""
    for position, size in enumerate(sizes):
        count = int(counts[position])
        if count < 2:
            raise RuntimeError(
                "ranked null construction produced fewer than two samples"
            )
        cache[int(size)] = (
            float(means[position]),
            float(np.sqrt(second_moments[position] / (count - 1))),
        )
    return cache


def build_ranked_null(
    E_unit,
    population,
    gene_set_sizes,
    ranked_embeddings,
    *,
    iterations=1000,
    seed=12345,
    worker_workspace_bytes=128 * 1024 * 1024,
    existing=None,
):
    """Return sequential prefix-coupled ranked null statistics.

    The null resolver validates ``existing`` before passing it here.
    """
    population = np.asarray(population, dtype=np.int32)
    cache, missing, iterations, seed = _prepare_ranked_null_request(
        gene_set_sizes,
        existing,
        population_size=population.size,
        iterations=iterations,
        seed=seed,
    )
    if not missing:
        return cache

    embeddings = np.asarray(E_unit, dtype=np.float32)
    ranked_embeddings = np.asarray(ranked_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or ranked_embeddings.ndim != 2:
        raise ValueError("embedding inputs must be two-dimensional")
    if embeddings.shape[1] != ranked_embeddings.shape[1]:
        raise ValueError("ranked and full embedding dimensions differ")

    population_embeddings = np.ascontiguousarray(
        embeddings[population],
        dtype=np.float32,
    )
    ranked_transpose = np.ascontiguousarray(
        ranked_embeddings.T,
        dtype=np.float32,
    )
    means, second_moments, counts = _compute_mc_stats(
        np.arange(iterations, dtype=np.int64),
        np.asarray(missing, dtype=np.int32),
        seed,
        population_embeddings,
        ranked_transpose,
        int(worker_workspace_bytes),
    )
    return _store_ranked_null_statistics(
        cache,
        missing,
        means,
        second_moments,
        counts,
    )


def build_ranked_null_parallel(
    E_unit,
    population,
    gene_set_sizes,
    ranked_embeddings,
    *,
    iterations=1000,
    seed=12345,
    workers=None,
    blas_threads_per_worker=1,
    worker_workspace_bytes=128 * 1024 * 1024,
    precompute_similarities=False,
    existing=None,
):
    """Return parallel prefix-coupled ranked null statistics.

    The null resolver validates ``existing`` before passing it here.
    """
    population = np.asarray(population, dtype=np.int32)
    cache, missing, iterations, seed = _prepare_ranked_null_request(
        gene_set_sizes,
        existing,
        population_size=population.size,
        iterations=iterations,
        seed=seed,
    )
    if not missing:
        return cache

    embeddings = np.asarray(E_unit, dtype=np.float32)
    ranked_embeddings = np.asarray(ranked_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or ranked_embeddings.ndim != 2:
        raise ValueError("embedding inputs must be two-dimensional")
    if embeddings.shape[1] != ranked_embeddings.shape[1]:
        raise ValueError("ranked and full embedding dimensions differ")

    worker_count = min(8, os.cpu_count() or 1) if workers is None else int(workers)
    worker_count = max(1, min(worker_count, iterations))
    iteration_chunks = [
        chunk
        for chunk in np.array_split(
            np.arange(iterations, dtype=np.int64),
            worker_count,
        )
        if chunk.size
    ]
    sizes = np.asarray(missing, dtype=np.int32)
    worker_arguments = [(chunk, sizes, seed) for chunk in iteration_chunks]

    with tempfile.TemporaryDirectory(prefix="andes_gsea_") as temporary_dir:
        population_path = os.path.join(temporary_dir, "population.npy")
        ranked_path = os.path.join(temporary_dir, "ranked.npy")
        population_embeddings = np.ascontiguousarray(
            embeddings[population],
            dtype=np.float32,
        )
        np.save(population_path, population_embeddings)
        np.save(
            ranked_path,
            np.ascontiguousarray(ranked_embeddings.T, dtype=np.float32),
        )

        similarity_path = ""
        if precompute_similarities:
            from threadpoolctl import threadpool_limits

            similarity_path = os.path.join(temporary_dir, "similarities.npy")
            similarities = np.lib.format.open_memmap(
                similarity_path,
                mode="w+",
                dtype=np.float32,
                shape=(population.size, ranked_embeddings.shape[0]),
            )
            ranked_transpose = np.load(
                ranked_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            with threadpool_limits(
                limits=int(blas_threads_per_worker),
                user_api="blas",
            ):
                np.matmul(
                    population_embeddings,
                    ranked_transpose,
                    out=similarities,
                )
            similarities.flush()
            del similarities, ranked_transpose

        del population_embeddings
        with ProcessPoolExecutor(
            max_workers=len(iteration_chunks),
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(
                population_path,
                ranked_path,
                similarity_path,
                int(blas_threads_per_worker),
                int(worker_workspace_bytes),
            ),
        ) as executor:
            partial_statistics = list(executor.map(_compute_mc_chunk, worker_arguments))

    means, second_moments, counts = _combine_welford(partial_statistics)
    return _store_ranked_null_statistics(
        cache,
        missing,
        means,
        second_moments,
        counts,
    )
