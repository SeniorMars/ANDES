"""
Exact ranked-ANDES scoring and numerical null construction.

Architecture
------------
A single Monte Carlo iteration draws one random permutation prefix of length
max_m and computes one matmul of shape (max_m, d) @ (d, L). The numba kernel
``_prefix_es_welford`` streams through the rows of the result, maintaining a
running column-wise max, and emits an ES score whenever the prefix length
matches a requested gene-set size. Welford accumulators are updated inline so
no per-iteration score array is allocated.

Per-iteration GEMM row work drops from sum(sizes) to max(sizes); the
1+2+...+max_m collapse is the dominant remaining optimization in the cache
build path.

Exact ranked best-match scoring
-------------------------------
For a ranked list R and terms X_t, score_terms_bestmatch computes
S[r,t] = max_{x in X_t} sim(R[r], x) in chunks, then runs the same column-wise
ES sweep as compute_es_score.  This is exact for the existing ranked ANDES
score.  For one ranked list the dot-product count is similar to the grouped
batched scorer, but the formulation exposes the reusable matrix
B[g,t] = max_{x in X_t} sim(g,x), which can be cached or sliced for many
ranked lists against the same gene-set database.

Reproducibility
---------------
Each iteration index k is seeded from ``SeedSequence([master_seed, k])``, so
the Monte Carlo sample for iteration k is independent of worker count or
batching. Final mu/sigma values are reproducible up to floating-point roundoff
because Welford merge order may differ across worker partitions (addition is
not associative). Differences are typically at the level of the last few ULPs
of float64 and well below Monte Carlo error.

Parallelism
-----------
Workers split the iteration index range, not the size set. Every worker
processes every size via prefix coupling and the results are merged with a
pairwise Welford combination at the end.

Memory
------
Per worker: O(b * max_m * (L + d) * 4) bytes for A_buf and the gathered X
block, where b is chosen so A_buf fits in es_batch_bytes (default 128 MB).
For max_m=300, L=18000, d=512, b=6: ~130 MB A_buf + ~4 MB X. Plus a shared
mmap of E_pop = E_unit[pop] (typically tens of MB).
"""

import os
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import jit
from tqdm import tqdm

from . import artifacts


# Hashing


def _hash_array(arr):
    """Return a 16-byte hex BLAKE2b digest of an array's shape, dtype, and data."""
    return artifacts.hash_array(arr)


# Numba kernels


@jit(nopython=True, nogil=True, cache=True)
def _fys_sample_single_into(perm, js, out):
    """Partial Fisher-Yates sample into a pre-allocated buffer.

    perm: int32 (N,) — identity [0..N-1] on entry and exit.
    js:   int64 (m,) — swap targets; js[j] must lie in [j, N-1].
    out:  int32 (>=m,) — receives the sampled local indices in out[:m].
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

    A3:         float32 (b, max_m, L) — b iterations of (max_m, L) matmul.
    sizes:      int32   (S,)          — sorted ascending; max(sizes) <= max_m.
    means:      float64 (S,)          — Welford running mean, updated in place.
    M2s:        float64 (S,)          — Welford running M2, updated in place.
    counts:     int64   (S,)          — Welford running count, updated in place.
    col_max_ws: float32 (L,)          — scratch buffer for the prefix col-max.

    Maintains ``sum_col_max`` incrementally as col-max entries change so the
    per-size mean is O(1) rather than another O(L) scan. ``fastmath`` is on
    for the ES sweep (Monte Carlo noise dominates any reassociation drift);
    the Welford block runs in float64 with strict ordering.
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


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _prefix_es_inner(col_max_ws, sum_col_max, L):
    """One ES sweep from a pre-summed col-max. fastmath OK; output is float32."""
    mean_val = sum_col_max / L
    running = 0.0
    max_abs = 0.0
    best_signed = 0.0
    for j in range(L):
        running += col_max_ws[j] - mean_val
        a = abs(running)
        if a > max_abs:
            max_abs = a
            best_signed = running
    return best_signed


@jit(nopython=True, nogil=True, cache=True)
def _welford_update(means, M2s, counts, i, x):
    """Single-sample Welford update in float64, no fastmath."""
    counts[i] += 1
    c = counts[i]
    delta = x - means[i]
    means[i] += delta / c
    delta2 = x - means[i]
    M2s[i] += delta * delta2


# User-facing scoring functions


def compute_ranked_emb(E_unit, ranked_list_idx):
    """Extract the embedding rows for a ranked gene list.

    Returns a C-contiguous float32 array of shape (L, d) where L is the number
    of ranked genes and d is the embedding dimension.  Call once per ranked
    list; pass the result (or its transpose) to the scoring functions.
    """
    return np.ascontiguousarray(
        E_unit[np.asarray(ranked_list_idx, dtype=np.int32)], dtype=np.float32
    )


def compute_es_score(E_unit, gene_set_idx, ranked_emb):
    """Compute the signed enrichment score for one gene set.

    col_max[i] = max cosine similarity between ranked gene i and any gene in
    the set.  The ES is the maximum signed cumulative sum of (col_max - mean).
    Allocates intermediate arrays; use compute_es_score_zero_alloc in hot loops.
    """
    X = E_unit[np.asarray(gene_set_idx, dtype=np.int32)]
    col_max = (X @ ranked_emb.T).max(axis=0)
    cs = np.cumsum(col_max - col_max.mean())
    return float(cs[np.abs(cs).argmax()])


def compute_es_score_with_buffers(
    E_unit, gene_set_idx, ranked_emb, A_buf, col_buf, cs_buf
):
    """Lower-allocation ES (NOT zero-allocation; see notes).

    Reuses caller-provided buffers for A, col_max, and the centered cumsum.
    Still allocates: the gathered X = E_unit[idx] fancy-index result (one
    (m, d) array). To make this truly allocation-free in a hot loop, pass an
    X_buf and use ``compute_es_score_zero_alloc`` below.
    """
    m = len(gene_set_idx)
    X = E_unit[np.asarray(gene_set_idx, dtype=np.int32)]
    np.matmul(X, ranked_emb.T, out=A_buf[:m])
    A_buf[:m].max(axis=0, out=col_buf)
    mean = col_buf.mean()
    np.subtract(col_buf, mean, out=cs_buf)
    np.cumsum(cs_buf, out=cs_buf)
    return float(cs_buf[_argmax_abs(cs_buf)])


def compute_es_score_zero_alloc(
    E_unit, gene_set_idx, ranked_emb_T, X_buf, A_buf, col_buf, cs_buf
):
    """Genuinely zero-allocation ES for tight inner loops.

    ranked_emb_T: float32 (d, L) — pre-transposed, C-contiguous (avoids .T overhead).
    Caller owns: X_buf (>= m, d), A_buf (>= m, L), col_buf (L,), cs_buf (L,).
    gene_set_idx must already be int32; no asarray is performed.
    """
    m = gene_set_idx.shape[0]
    np.take(E_unit, gene_set_idx, axis=0, out=X_buf[:m])
    np.matmul(X_buf[:m], ranked_emb_T, out=A_buf[:m])
    A_buf[:m].max(axis=0, out=col_buf)
    mean = col_buf.mean()
    np.subtract(col_buf, mean, out=cs_buf)
    np.cumsum(cs_buf, out=cs_buf)
    return float(cs_buf[_argmax_abs(cs_buf)])


@jit(nopython=True, nogil=True, cache=True)
def _argmax_abs(x):
    best_i = 0
    best_a = 0.0
    for i in range(x.shape[0]):
        a = abs(x[i])
        if a > best_a:
            best_a = a
            best_i = i
    return best_i


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _es_scores_from_col_max_batch(col_max_batch, scores_out):
    """Signed ES score for each row of col_max_batch.

    col_max_batch: float32 (B, L)
    scores_out:    float32 (B,)
    """
    B = col_max_batch.shape[0]
    L = col_max_batch.shape[1]
    for i in range(B):
        sum_val = 0.0
        for j in range(L):
            sum_val += col_max_batch[i, j]
        mean_val = sum_val / L
        running = 0.0
        max_abs = 0.0
        best_signed = 0.0
        for j in range(L):
            running += col_max_batch[i, j] - mean_val
            a = abs(running)
            if a > max_abs:
                max_abs = a
                best_signed = running
        scores_out[i] = best_signed


@jit(nopython=True, nogil=True, cache=True)
def _es_scores_from_ranked_bestmatch(
    best_by_rank,
    scores_out,
    mean_ws,
    running_ws,
    max_abs_ws,
):
    """Score columns of an indexed ranked best-match block.

    ``best_by_rank`` is row-major ``(ranked_genes, terms)``.  Traversing ranks
    outside and terms inside keeps both passes contiguous while the three
    term-sized workspaces avoid centered and cumulative matrix temporaries.
    """
    ranked_len = best_by_rank.shape[0]
    n_terms = best_by_rank.shape[1]

    for term_i in range(n_terms):
        mean_ws[term_i] = np.float32(0.0)
        running_ws[term_i] = np.float32(0.0)
        max_abs_ws[term_i] = np.float32(0.0)
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
            if absolute > max_abs_ws[term_i]:
                max_abs_ws[term_i] = absolute
                scores_out[term_i] = running_ws[term_i]


def standardize_ranked_scores(true_scores, term_sizes, cache, out=None):
    """Standardize aligned ranked-ES scores by gene-set size.

    ``cache`` may be a ``RankedNullBuilder`` instance or a mapping from size to
    ``(mean, standard_deviation)``.  Supplying ``out=true_scores`` performs the
    calibration in place.
    """
    scores = np.asarray(true_scores)
    sizes = np.asarray(term_sizes)
    if scores.ndim != 1 or sizes.ndim != 1 or scores.size != sizes.size:
        raise ValueError("true_scores and term_sizes must be same-length vectors")
    if sizes.dtype.kind not in "iu":
        raise TypeError("term_sizes must contain integers")
    sizes = sizes.astype(np.int32, copy=False)
    cache_values = cache.cache if hasattr(cache, "cache") else cache
    missing = sorted(
        set(int(size) for size in np.unique(sizes))
        - set(int(size) for size in cache_values)
    )
    if missing:
        raise KeyError(f"sizes not in null cache: {missing}")

    if out is None:
        out = np.empty(scores.shape, dtype=np.float32)
    else:
        out = np.asarray(out)
        if out.shape != scores.shape:
            raise ValueError("out must have the same shape as true_scores")

    for size in np.unique(sizes):
        mask = sizes == size
        mean, standard_deviation = cache_values[int(size)]
        if standard_deviation == 0.0:
            out[mask] = 0.0
        else:
            out[mask] = (scores[mask].astype(np.float64) - float(mean)) / float(
                standard_deviation
            )
    return np.asarray(out, dtype=np.float32)


def score_terms_batched_exact(
    E_unit,
    geneset_indices_np,
    geneset_terms,
    ranked_emb_T,
    batch_bytes=128 * 1024 * 1024,
):
    """Return aligned exact ranked scores using size-grouped GEMMs.

    Groups terms by gene-set size and scores K same-size terms at once via a
    single (K*m, d) @ (d, L) GEMM instead of K separate (m, d) @ (d, L) calls.

    ranked_emb_T: float32 (d, L) — pre-transposed, C-contiguous.
    Returns an array aligned with ``geneset_terms``.
    """
    d = E_unit.shape[1]
    L = ranked_emb_T.shape[1]

    terms_by_size = defaultdict(list)
    for term in geneset_terms:
        terms_by_size[len(geneset_indices_np[term])].append(term)

    if not terms_by_size:
        return np.empty(0, dtype=np.float32)

    max_m = max(terms_by_size)
    # Ensure A_buf can always hold at least one full gene set.
    row_budget = max(max_m, batch_bytes // (L * 4))

    X_buf = np.empty((row_budget, d), dtype=np.float32)
    A_buf = np.empty((row_budget, L), dtype=np.float32)
    flat_idx_buf = np.empty(row_budget, dtype=np.int32)

    term_positions = {term: i for i, term in enumerate(geneset_terms)}
    true_scores = np.empty(len(geneset_terms), dtype=np.float32)

    for m, terms_m in terms_by_size.items():
        terms_per_chunk = max(1, row_budget // m)
        # colmax_buf sized to actual chunk width, not max_rows.
        colmax_buf = np.empty((terms_per_chunk, L), dtype=np.float32)
        score_buf = np.empty(terms_per_chunk, dtype=np.float32)
        for start in range(0, len(terms_m), terms_per_chunk):
            chunk = terms_m[start : start + terms_per_chunk]
            k = len(chunk)
            rows = k * m

            pos = 0
            for term in chunk:
                flat_idx_buf[pos : pos + m] = geneset_indices_np[term]
                pos += m

            np.take(E_unit, flat_idx_buf[:rows], axis=0, out=X_buf[:rows])
            np.matmul(X_buf[:rows], ranked_emb_T, out=A_buf[:rows])
            A_buf[:rows].reshape(k, m, L).max(axis=1, out=colmax_buf[:k])
            _es_scores_from_col_max_batch(colmax_buf[:k], score_buf[:k])

            for i, term in enumerate(chunk):
                true_scores[term_positions[term]] = score_buf[i]

    return true_scores


def score_terms_batched(
    E_unit,
    geneset_indices_np,
    geneset_terms,
    ranked_emb_T,
    cache,
    batch_bytes=128 * 1024 * 1024,
):
    """Compatibility wrapper returning term-keyed true and calibrated scores."""
    aligned_true = score_terms_batched_exact(
        E_unit,
        geneset_indices_np,
        geneset_terms,
        ranked_emb_T,
        batch_bytes=batch_bytes,
    )
    sizes = np.asarray(
        [len(geneset_indices_np[term]) for term in geneset_terms],
        dtype=np.int32,
    )
    aligned_z = standardize_ranked_scores(aligned_true, sizes, cache)
    return (
        {term: float(aligned_true[i]) for i, term in enumerate(geneset_terms)},
        {term: float(aligned_z[i]) for i, term in enumerate(geneset_terms)},
    )


def _term_ranges_for_ranked_bestmatch(lengths, ranked_len, max_workspace_mb):
    """Split target terms so ranked-list x concatenated-genes workspace is bounded."""
    n_terms = len(lengths)
    if n_terms == 0:
        return [], 0.0

    if not max_workspace_mb or max_workspace_mb <= 0:
        ranges = [(0, n_terms)]
    else:
        budget_bytes = max_workspace_mb * 1e6
        ranges = []
        start = 0
        cols = 0
        for i, length in enumerate(lengths):
            next_cols = cols + int(length)
            next_bytes = ranked_len * next_cols * 4
            if cols and next_bytes > budget_bytes:
                ranges.append((start, i))
                start = i
                cols = int(length)
            else:
                cols = next_cols
        ranges.append((start, n_terms))

    max_bytes = 0
    for start, end in ranges:
        cols = int(lengths[start:end].sum())
        max_bytes = max(max_bytes, ranked_len * cols * 4)
    return ranges, max_bytes / 1e6


def score_terms_bestmatch_exact(
    E_unit,
    geneset_indices_np,
    geneset_terms,
    ranked_emb,
    max_workspace_mb=1024,
):
    """Return aligned exact ranked scores using best-match matrices.

    For ranked position r and term t, compute
    S[r,t] = max_{x in term_t} sim(ranked_gene_r, x).  The ES for each term is
    then the same column-wise signed max-absolute cumulative sum used by
    compute_es_score.  This is exact for the existing ranked ANDES score and is
    additive to the current size-grouped batched scorer.
    """
    lengths = np.asarray(
        [len(geneset_indices_np[t]) for t in geneset_terms], dtype=np.int32
    )
    ranges, workspace_mb = _term_ranges_for_ranked_bestmatch(
        lengths, ranked_emb.shape[0], max_workspace_mb
    )
    true_scores = np.empty(len(geneset_terms), dtype=np.float32)

    for start, end in ranges:
        terms_chunk = geneset_terms[start:end]
        lengths_chunk = lengths[start:end]
        offsets = np.empty(len(terms_chunk) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths_chunk, out=offsets[1:])

        concat_idx = np.concatenate([geneset_indices_np[t] for t in terms_chunk])
        concat_emb = np.ascontiguousarray(E_unit[concat_idx], dtype=np.float32)

        sims = ranked_emb @ concat_emb.T
        best_by_rank = np.maximum.reduceat(sims, offsets[:-1], axis=1)

        centered = best_by_rank - best_by_rank.mean(axis=0, keepdims=True)
        running = np.cumsum(centered, axis=0, dtype=np.float32)
        best_pos = np.abs(running).argmax(axis=0)
        scores = running[best_pos, np.arange(running.shape[1])]

        true_scores[start:end] = np.asarray(scores, dtype=np.float32)

    return true_scores, {"workspace_mb": workspace_mb}


def score_terms_bestmatch(
    E_unit,
    geneset_indices_np,
    geneset_terms,
    ranked_emb,
    cache,
    max_workspace_mb=1024,
):
    """Compatibility wrapper returning term-keyed true and calibrated scores."""
    aligned_true, stats = score_terms_bestmatch_exact(
        E_unit,
        geneset_indices_np,
        geneset_terms,
        ranked_emb,
        max_workspace_mb=max_workspace_mb,
    )
    sizes = np.asarray(
        [len(geneset_indices_np[term]) for term in geneset_terms],
        dtype=np.int32,
    )
    aligned_z = standardize_ranked_scores(aligned_true, sizes, cache)
    return (
        {term: float(aligned_true[i]) for i, term in enumerate(geneset_terms)},
        {term: float(aligned_z[i]) for i, term in enumerate(geneset_terms)},
        stats,
    )


def score_terms_indexed(
    bestmatch,
    ranked_idx,
    term_sizes,
    cache=None,
    max_workspace_mb=128,
):
    """Score a ranked list by slicing a persistent gene-to-term index.

    Parameters
    ----------
    bestmatch : array-like, shape (embedding_genes, terms), float32
        ``bestmatch[g, t] = max(sim(g, x) for x in term_t)``.
    ranked_idx : array-like, shape (ranked_genes,)
        Embedding row indices in ranked order.
    term_sizes : array-like, shape (terms,)
        Size of each indexed term, used only for optional z-score lookup.
    cache : RankedNullBuilder or mapping, optional
        ES null cache.  When omitted, the second return value is ``None``.
    max_workspace_mb : float
        Upper bound for the gathered ``bestmatch[ranked_idx, term_chunk]``
        matrix.  Terms are streamed in column chunks.

    Returns
    -------
    true_scores : ndarray, shape (terms,), float32
    z_scores : ndarray, shape (terms,), float32 or None
    stats : dict
        Peak gathered workspace and chosen term chunk size.

    This is exactly the existing ranked ANDES statistic: each indexed column
    is centered over ranked positions, cumulatively summed, and reduced at the
    first maximum absolute deviation.
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

    term_sizes = np.asarray(term_sizes)
    if term_sizes.ndim != 1 or term_sizes.size != B.shape[1]:
        raise ValueError("term_sizes length must equal bestmatch column count")
    if term_sizes.dtype.kind not in "iu":
        raise TypeError("term_sizes must contain integers")
    if np.any(term_sizes <= 0):
        raise ValueError("term_sizes must all be positive")
    if term_sizes.size and int(term_sizes.max()) > np.iinfo(np.int32).max:
        raise ValueError("term_sizes exceeds int32 range")
    term_sizes = term_sizes.astype(np.int32, copy=False)

    n_terms = B.shape[1]
    ranked_len = ranked_idx.size
    if max_workspace_mb is None or float(max_workspace_mb) <= 0.0:
        terms_per_chunk = max(1, n_terms)
    else:
        budget_bytes = max(4, int(float(max_workspace_mb) * 1e6))
        terms_per_chunk = max(1, budget_bytes // (ranked_len * 4))
        terms_per_chunk = min(terms_per_chunk, max(1, n_terms))

    true_scores = np.empty(n_terms, dtype=np.float32)
    peak_bytes = 0
    for start in range(0, n_terms, terms_per_chunk):
        end = min(start + terms_per_chunk, n_terms)
        width = end - start
        # Advanced indexing intentionally materializes just this bounded block.
        best_by_rank = np.array(
            B[ranked_idx, start:end],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        peak_bytes = max(peak_bytes, int(best_by_rank.nbytes))
        score_buf = true_scores[start:end]
        mean_ws = np.empty(width, dtype=np.float32)
        running_ws = np.empty(width, dtype=np.float32)
        max_abs_ws = np.empty(width, dtype=np.float32)
        _es_scores_from_ranked_bestmatch(
            best_by_rank,
            score_buf,
            mean_ws,
            running_ws,
            max_abs_ws,
        )

    z_scores = (
        None
        if cache is None
        else standardize_ranked_scores(true_scores, term_sizes, cache)
    )

    return (
        true_scores,
        z_scores,
        {
            "workspace_mb": peak_bytes / 1e6,
            "term_chunk_size": int(terms_per_chunk),
        },
    )


def compute_es_trace(E_unit, gene_set_idx, ranked_emb):
    """Compute the full ES trace for one gene set (for plotting).

    Returns a dict with:
      best_match_score    : float32 (L,) — col_max before centering
      best_gene_set_position : int32 (L,) — which gene-set row achieves col_max
      centered_score      : float32 (L,) — col_max - mean(col_max)
      running_es          : float32 (L,) — cumulative sum of centered_score
      es_index            : int — position of the maximum |running_es|
      es                  : float — ES value (matches compute_es_score)
    """
    gene_set_idx = np.asarray(gene_set_idx, dtype=np.int32)
    X = E_unit[gene_set_idx]
    A = X @ ranked_emb.T
    best_gene_set_position = A.argmax(axis=0).astype(np.int32)
    cols = np.arange(A.shape[1])
    best_match_score = A[best_gene_set_position, cols].astype(np.float32)
    centered_score = (best_match_score - best_match_score.mean()).astype(np.float32)
    running_es = np.cumsum(centered_score, dtype=np.float32)
    es_index = int(np.abs(running_es).argmax())
    return {
        "best_match_score": best_match_score,
        "best_gene_set_position": best_gene_set_position,
        "centered_score": centered_score,
        "running_es": running_es,
        "es_index": es_index,
        "es": float(running_es[es_index]),
    }


def warmup_numba_es():
    """Compile all Numba kernels in this module before timed code runs.

    Each kernel is called with minimal synthetic inputs so the JIT pass
    completes at warmup rather than on the first real iteration.  Call after
    ``bma.warmup_numba()`` if both modules are in use.
    """
    perm = np.arange(10, dtype=np.int32)
    js = np.array([0, 2, 3, 4], dtype=np.int64)
    out = np.empty(4, dtype=np.int32)
    _fys_sample_single_into(perm, js, out)

    A3 = np.zeros((2, 4, 6), dtype=np.float32)
    sizes = np.array([2, 4], dtype=np.int32)
    means = np.zeros(2, dtype=np.float64)
    M2s = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    col_max_ws = np.empty(6, dtype=np.float32)
    _prefix_es_welford(A3, sizes, means, M2s, counts, col_max_ws)
    _prefix_es_inner(col_max_ws, 0.0, 6)
    _welford_update(means, M2s, counts, 0, 0.0)
    _argmax_abs(np.zeros(4, dtype=np.float32))

    colmax = np.zeros((2, 6), dtype=np.float32)
    scores = np.zeros(2, dtype=np.float32)
    _es_scores_from_col_max_batch(colmax, scores)
    mean_ws = np.zeros(2, dtype=np.float32)
    running_ws = np.zeros(2, dtype=np.float32)
    max_abs_ws = np.zeros(2, dtype=np.float32)
    _es_scores_from_ranked_bestmatch(
        colmax.T,
        scores,
        mean_ws,
        running_ws,
        max_abs_ws,
    )


# Shared population context


class GSEAPrepContext:
    """Materialize E_pop = E_unit[pop] to a tempfile once, share across builds.

    Reuse across multiple ranked-list cache builds against the same
    (E_unit, pop); only ``ranked_emb`` is rewritten per ranked list.
    """

    def __init__(self, E_unit, pop):
        """Write E_pop = E_unit[pop] to a tempfile as a memory-mapped array.

        The file persists for the lifetime of this context and is cleaned up
        by cleanup() or by using the object as a context manager.  Worker
        processes load it with mmap_mode='r' to avoid duplicating memory.

        Parameters
        ----------
        E_unit : numpy.ndarray, float32 (N, d)
            L2-normalized embedding matrix.
        pop : numpy.ndarray, int32 (P,)
            Row indices selecting the background gene pool from E_unit.
        """
        self._tmp = tempfile.TemporaryDirectory(prefix="andes_gsea_")
        self.tmp_dir = self._tmp.name
        self.e_pop_path = os.path.join(self.tmp_dir, "E_pop.npy")

        E_pop = np.ascontiguousarray(
            E_unit[np.asarray(pop, dtype=np.int32)], dtype=np.float32
        )
        np.save(self.e_pop_path, E_pop)

        self.N_pop = E_pop.shape[0]
        self.d = E_pop.shape[1]
        self.emb_hash = _hash_array(np.asarray(E_unit, dtype=np.float32))
        self.pop_hash = _hash_array(np.asarray(pop, dtype=np.int32))

    def cleanup(self):
        """Delete the temp directory and its files.  Safe to call more than once."""
        try:
            self._tmp.cleanup()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()


# Worker

# Worker-process globals — set by _init_worker via ProcessPoolExecutor.
_W_E_POP = None  # float32 (P, d) — background embedding block; mmap'd read-only
_W_RANKED_EMB_T = (
    None  # float32 (d, L) — ranked-list embedding, transposed for matmul without .T
)
_W_ES_BATCH_BYTES = (
    None  # int — memory cap per worker for the (b*max_m, L) matmul workspace
)


def _init_worker(e_pop_path, ranked_path, blas_threads, es_batch_bytes):
    """ProcessPoolExecutor initializer — runs once per worker at fork.

    Loads E_pop and ranked_emb_T (stored as (d, L) for matmul without .T) from
    temp .npy files written by GSEAPrepContext and precompute_from_context.
    Uses mmap_mode='r' so the OS can share physical pages across workers when
    the arrays fit in the page cache.  Sets BLAS thread counts and optionally
    applies threadpoolctl limits to prevent thread oversubscription.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(blas_threads)
    from threadpoolctl import threadpool_limits

    threadpool_limits(blas_threads)

    global _W_E_POP, _W_RANKED_EMB_T, _W_ES_BATCH_BYTES
    _W_E_POP = np.load(e_pop_path, mmap_mode="r")
    _W_RANKED_EMB_T = np.load(ranked_path, mmap_mode="r")  # shape (d, L)
    _W_ES_BATCH_BYTES = int(es_batch_bytes)


def _compute_mc_chunk(args):
    """Welford stats over a contiguous range of iteration indices.

    All sizes are processed jointly via prefix coupling; each iteration uses
    one matmul at max_m and emits ES for every requested size.
    """
    iter_indices, sizes_arr, master_seed = args

    sizes = np.asarray(sizes_arr, dtype=np.int32)
    S = len(sizes)
    max_m = int(sizes[-1])

    N_pop = _W_E_POP.shape[0]
    d = _W_E_POP.shape[1]
    L = _W_RANKED_EMB_T.shape[1]  # _W_RANKED_EMB_T is (d, L)
    ite_chunk = len(iter_indices)

    max_A_rows = max(1, _W_ES_BATCH_BYTES // (L * 4))
    b = max(1, min(ite_chunk, max_A_rows // max_m))

    A_buf = np.empty((b * max_m, L), dtype=np.float32)
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

        # Per-iteration deterministic seed. SeedSequence([master, k]) makes
        # iteration k reproducible independent of worker count or batching.
        for bi in range(b_act):
            iter_idx = int(iter_indices[done + bi])
            rng = np.random.default_rng(
                np.random.SeedSequence([int(master_seed), iter_idx])
            )
            js_row = rng.integers(m_low, N_pop, dtype=np.int64)
            _fys_sample_single_into(perm, js_row, xi[bi])

        # NumPy fancy index gather: SIMD'd C path, allocation amortized.
        X = _W_E_POP[xi[:b_act].ravel()]  # (b_act * max_m, d)
        np.matmul(X, _W_RANKED_EMB_T, out=A_buf[: b_act * max_m])

        A3 = A_buf[: b_act * max_m].reshape(b_act, max_m, L)
        _prefix_es_welford(A3, sizes, means, M2s, counts, col_max_ws)

        done += b_act

    return means, M2s, counts


def _combine_welford(stats_list):
    """Merge per-worker Welford statistics into a single aggregate.

    stats_list : list of (means, M2s, counts) — one tuple per worker.
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


# Cache class


class RankedNullBuilder:
    """Prefix-coupled null cache. Same query interface as the previous version.

    Build path: one matmul at max_m per Monte Carlo iteration, ES extracted for
    every requested size via prefix col-max. Welford means/variances accumulate
    inline; workers parallelize over iteration indices, not sizes.
    """

    def __init__(self):
        self.cache: dict[int, tuple[float, float]] = {}  # {size m: (mu, sigma)}
        self.metadata: dict = {}  # build parameters; checked on load

    # metadata

    @staticmethod
    def build_metadata(E_unit, pop, ranked_emb, ite, seed):
        """Build the metadata dict that identifies a specific cache build.

        Contains BLAKE2b hashes of E_unit, pop, and ranked_emb (so any change
        in inputs produces a different key), plus ite and seed.  Used by
        metadata_matches to detect stale cache files.
        """
        return {
            "kind": "andes_gsea_es_null",
            "version": 4,
            "embedding_hash": _hash_array(np.asarray(E_unit, dtype=np.float32)),
            "population_hash": _hash_array(np.asarray(pop, dtype=np.int32)),
            "ranked_emb_hash": _hash_array(np.asarray(ranked_emb, dtype=np.float32)),
            "ite": int(ite),
            "seed": int(seed),
        }

    def metadata_matches(self, expected):
        """Check whether the loaded cache was built with the same inputs.

        Returns (True, "") if every key in expected matches self.metadata.
        Returns (False, reason) on the first mismatch or if metadata is absent.
        """
        if not self.metadata:
            return False, "cache has no metadata"
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                return False, f"metadata mismatch for {key}"
        return True, ""

    @staticmethod
    def resolve_seed(seed):
        """Return a concrete non-negative seed integer.

        If seed is None or negative, draws entropy from the OS via
        numpy.random.SeedSequence so the result is non-deterministic but
        still fully reproducible if passed back in on the next run.
        """
        if seed is None or int(seed) < 0:
            return int(np.random.SeedSequence().entropy)
        return int(seed)

    # query (hot path)

    def get_zscore(self, true_score: float, m: int) -> float:
        """Return (true_score - mu) / sigma for gene-set size m.

        Returns 0.0 when sigma is zero (degenerate null distribution).
        Raises KeyError if m is not in the cache.
        """
        mu, sigma = self.cache[int(m)]
        if sigma == 0.0:
            return 0.0
        return (true_score - mu) / sigma

    # build

    def precompute_parallel(
        self,
        E_unit,
        pop,
        gene_set_sizes,
        ranked_emb,
        ite: int = 1000,
        seed: int = 12345,
        verbose: bool = False,
        n_workers: int | None = None,
        blas_threads_per_worker: int = 1,
        es_batch_bytes: int = 128 * 1024 * 1024,
        show_progress: bool | None = None,
        overwrite_on_mismatch: bool = True,
        **legacy_kwargs,
    ):
        """One-shot parallel build. Spins up a fresh GSEAPrepContext.

        Use precompute_from_context to share E_pop across multiple ranked lists.
        Accepts (but ignores) deprecated kwargs: chunk_size, use_numba_below.
        """
        _allowed_legacy = {"chunk_size", "use_numba_below"}
        unknown = set(legacy_kwargs) - _allowed_legacy
        if unknown:
            raise TypeError(f"unexpected keyword argument(s): {sorted(unknown)}")

        with GSEAPrepContext(E_unit, pop) as ctx:
            self.precompute_from_context(
                ctx,
                gene_set_sizes,
                ranked_emb,
                ite=ite,
                seed=seed,
                verbose=verbose,
                n_workers=n_workers,
                blas_threads_per_worker=blas_threads_per_worker,
                es_batch_bytes=es_batch_bytes,
                show_progress=show_progress,
                overwrite_on_mismatch=overwrite_on_mismatch,
            )

    def precompute_from_context(
        self,
        ctx: GSEAPrepContext,
        gene_set_sizes,
        ranked_emb,
        ite: int = 1000,
        seed: int = 12345,
        verbose: bool = False,
        n_workers: int | None = None,
        blas_threads_per_worker: int = 1,
        es_batch_bytes: int = 128 * 1024 * 1024,
        show_progress: bool | None = None,
        overwrite_on_mismatch: bool = True,
    ):
        """Parallel build that reuses a GSEAPrepContext across ranked lists.

        If existing cache entries were built with different metadata
        (embedding/population/ranked_emb hash, ite, or seed), they are stale.
        With ``overwrite_on_mismatch=True`` (default) the cache is cleared and
        rebuilt. With ``overwrite_on_mismatch=False``, a ValueError is raised.
        """
        seed = self.resolve_seed(seed)
        sizes_sorted = sorted(set(int(m) for m in gene_set_sizes))
        if not sizes_sorted:
            return
        if min(sizes_sorted) < 1:
            raise ValueError(
                f"gene set sizes must be positive, got {min(sizes_sorted)}"
            )
        if int(ite) < 1:
            raise ValueError(f"ite must be >= 1, got {ite}")
        if max(sizes_sorted) > ctx.N_pop:
            raise ValueError(
                f"max gene set size {max(sizes_sorted)} exceeds "
                f"population size {ctx.N_pop}"
            )

        expected = {
            "kind": "andes_gsea_es_null",
            "version": 4,
            "embedding_hash": ctx.emb_hash,
            "population_hash": ctx.pop_hash,
            "ranked_emb_hash": _hash_array(np.asarray(ranked_emb, dtype=np.float32)),
            "ite": int(ite),
            "seed": int(seed),
        }

        # Reject stale cache entries before deciding what to compute.
        if self.cache:
            ok, reason = self.metadata_matches(expected)
            if not ok:
                if overwrite_on_mismatch:
                    if verbose:
                        print(f"Cache metadata mismatch ({reason}); clearing.")
                    self.cache.clear()
                else:
                    raise ValueError(f"cache metadata mismatch: {reason}")

        self.metadata = expected

        todo = [m for m in sizes_sorted if m not in self.cache]
        if not todo:
            if verbose:
                print(f"All {len(sizes_sorted)} sizes already cached.")
            return

        if n_workers is None:
            n_workers = min(8, os.cpu_count() or 1)
        n_workers = max(1, min(n_workers, ite))

        max_m = max(todo)

        # Contiguous iteration ranges per worker; deterministic per-iter seeding
        # makes the partitioning statistically irrelevant.
        idx_arr = np.arange(ite, dtype=np.int64)
        chunks = [c for c in np.array_split(idx_arr, n_workers) if len(c) > 0]
        sizes_arr = np.asarray(todo, dtype=np.int32)
        args_list = [(c, sizes_arr, int(seed)) for c in chunks]

        if verbose:
            a_mb = max_m * ranked_emb.shape[0] * 4 / 1e6
            print(
                f"Prefix-coupled ES null: {len(todo)} sizes  "
                f"workers={len(chunks)}  ite={ite}  max_m={max_m}\n"
                f"  per-iter (max_m, L) block: ~{a_mb:.1f} MB  "
                f"workspace cap: {es_batch_bytes / 1e6:.0f} MB"
            )

        ranked_fd, ranked_path = tempfile.mkstemp(
            suffix=".npy", prefix="ranked_emb_", dir=ctx.tmp_dir
        )
        os.close(ranked_fd)
        # Store as (d, L) C-contiguous so workers can matmul without a .T view.
        np.save(ranked_path, np.ascontiguousarray(ranked_emb.T, dtype=np.float32))

        try:
            with ProcessPoolExecutor(
                max_workers=len(chunks),
                initializer=_init_worker,
                initargs=(
                    ctx.e_pop_path,
                    ranked_path,
                    int(blas_threads_per_worker),
                    int(es_batch_bytes),
                ),
            ) as ex:
                it = ex.map(_compute_mc_chunk, args_list)
                _show = show_progress if show_progress is not None else verbose
                if _show:
                    it = tqdm(it, total=len(chunks), desc="ES null (parallel)")
                results = list(it)
        finally:
            try:
                os.remove(ranked_path)
            except OSError:
                pass

        agg_mean, agg_m2, agg_count = _combine_welford(results)
        for i, m in enumerate(todo):
            c = int(agg_count[i])
            mu = float(agg_mean[i])
            std = float(np.sqrt(agg_m2[i] / (c - 1))) if c > 1 else 0.0
            self.cache[int(m)] = (mu, std)

        if verbose:
            print(f"Cached {len(self.cache)} ES null distributions")

    def precompute(
        self,
        E_unit,
        pop,
        gene_set_sizes,
        ranked_emb,
        ite: int = 1000,
        seed: int = 12345,
        verbose: bool = False,
        es_batch_bytes: int = 128 * 1024 * 1024,
        overwrite_on_mismatch: bool = True,
    ):
        """Single-process sequential build through the same scoring kernels."""
        global _W_E_POP, _W_RANKED_EMB_T, _W_ES_BATCH_BYTES

        seed = self.resolve_seed(seed)
        sizes_sorted = sorted(set(int(m) for m in gene_set_sizes))
        if not sizes_sorted:
            return
        if min(sizes_sorted) < 1:
            raise ValueError(
                f"gene set sizes must be positive, got {min(sizes_sorted)}"
            )
        if int(ite) < 1:
            raise ValueError(f"ite must be >= 1, got {ite}")
        N_pop = len(pop)
        if max(sizes_sorted) > N_pop:
            raise ValueError(
                f"max gene set size {max(sizes_sorted)} exceeds population size {N_pop}"
            )

        expected = self.build_metadata(E_unit, pop, ranked_emb, ite, seed)
        if self.cache:
            ok, reason = self.metadata_matches(expected)
            if not ok:
                if overwrite_on_mismatch:
                    if verbose:
                        print(f"Cache metadata mismatch ({reason}); clearing.")
                    self.cache.clear()
                else:
                    raise ValueError(f"cache metadata mismatch: {reason}")
        self.metadata = expected

        todo = [m for m in sizes_sorted if m not in self.cache]
        if not todo:
            if verbose:
                print(f"All {len(sizes_sorted)} sizes already cached.")
            return

        _W_E_POP = np.ascontiguousarray(
            E_unit[np.asarray(pop, dtype=np.int32)], dtype=np.float32
        )
        _W_RANKED_EMB_T = np.ascontiguousarray(ranked_emb.T, dtype=np.float32)
        _W_ES_BATCH_BYTES = int(es_batch_bytes)

        iter_indices = np.arange(ite, dtype=np.int64)
        sizes_arr = np.asarray(todo, dtype=np.int32)

        if verbose:
            print(
                f"Sequential ES null: {len(todo)} sizes  ite={ite}  max_m={max(todo)}"
            )

        means, m2s, counts = _compute_mc_chunk((iter_indices, sizes_arr, seed))

        for i, m in enumerate(todo):
            c = int(counts[i])
            mu = float(means[i])
            std = float(np.sqrt(m2s[i] / (c - 1))) if c > 1 else 0.0
            self.cache[int(m)] = (mu, std)

        if verbose:
            print(f"Cached {len(self.cache)} ES null distributions")

    # persistence

    def save_artifact(self, path, *, overwrite=False):
        """Persist this cache as a validated, non-pickled null artifact."""
        from .nulls import RankedNullModel

        RankedNullModel.from_builder(self).save(path, overwrite=overwrite)

    @classmethod
    def load_artifact(cls, path):
        """Create a compatibility cache from a typed ranked null artifact."""
        from .nulls import RankedNullModel

        model = RankedNullModel.load(path)
        obj = cls()
        obj.cache = model.to_mapping()
        obj.metadata = {
            "kind": "andes_gsea_es_null",
            "version": 4,
            "embedding_hash": model.spec.embedding_hash,
            "population_hash": model.spec.population_hashes[0],
            "ranked_emb_hash": model.spec.ranked_hash,
            "ite": model.spec.iterations,
            "seed": model.spec.seed,
            "std_ddof": model.spec.ddof,
            "null_sampling": model.spec.sampling,
        }
        return obj

    @staticmethod
    def suggest_path(
        base_dir,
        E_unit,
        pop,
        ranked_emb,
        *,
        ite=1000,
        seed=12345,
        sampling="prefix_coupled",
        ddof=1,
    ):
        """Content-addressed path covering inputs and all null parameters."""
        from .nulls import NullSpec

        spec = NullSpec(
            kind="ranked",
            iterations=int(ite),
            seed=int(seed),
            sampling=str(sampling),
            ddof=int(ddof),
            embedding_hash=_hash_array(np.asarray(E_unit, dtype=np.float32)),
            population_hashes=(_hash_array(np.asarray(pop, dtype=np.int32)),),
            ranked_hash=_hash_array(np.asarray(ranked_emb, dtype=np.float32)),
        )
        return os.path.join(base_dir, f"es_{spec.fingerprint}.null")

    def __len__(self):
        return len(self.cache)

    def __contains__(self, m):
        return int(m) in self.cache

    def missing_sizes(self, gene_set_sizes):
        """Return sizes from gene_set_sizes that have no cache entry."""
        return [m for m in gene_set_sizes if int(m) not in self.cache]
