"""
Exact BMA scoring kernels, numerical null construction, and supporting utilities.

Best Match Average (BMA)
------------------------
BMA(A, B) = (sum_a max_b sim(a,b) + sum_b max_a sim(a,b)) / (|A| + |B|)

where sim is cosine similarity on L2-normalized embeddings.  Scores are
normalized against a null distribution (mu, sigma) per size pair (m, k),
giving a z-score.

Null builder: BmaNullBuilder
------------------------
For each unique (m, k) pair, run `ite` Monte Carlo iterations drawing random
gene sets of size m and k from the background pool, compute BMA, accumulate
mean and sample std (ddof=1).  The cache is keyed on (embedding hash, pop1
hash, pop2 hash, ite, seed, std ddof) so stale entries are detected on load.

Optimization notes
------------------
Batched FYS sampling
    rng.integers(low, high, size=(b, m)) generates all swap indices for b
    iterations in one NumPy call, fed into _batch_fys_sample which applies
    partial Fisher-Yates in a Numba loop at O(b*m) cost.  This replaces b
    separate rng.choice(N, m, replace=False) calls that each took O(N) time
    (N ≈ 18k in production, m ≈ 50-300 → ~60-360x speedup on sampling alone).

Cost-based chunking (_chunked_by_cost)
    Pairs are sorted descending by m*k cost and distributed to chunks with a
    greedy load-balancing step so no worker gets all the expensive pairs.

Batched GEMM in _bma_batch
    b iterations are packed into (b, m, d) and (b, k, d) tensors; a single
    np.matmul call produces (b, m, k).  This amortizes BLAS launch overhead
    across iterations.

Term-block cache (precompute_term_embedding_blocks)
    E_unit[idx] is gathered once per term at startup and stored as a
    contiguous float32 block.  Query scoring then calls matmul on
    pre-gathered blocks, avoiding repeated fancy-index gathers.

Batched row scorer (score_bma_zscore_matrix_batched)
    All gene-set-2 blocks are concatenated into one (total_genes, d) matrix.
    For each gene-set-1 term, one GEMM produces a (m, total_genes) result;
    np.maximum.reduceat extracts per-term row-max without looping.

Exact all-vs-all best-match scorer (score_bma_zscore_matrix_bestmatch)
    For target terms Y_j, precompute B2[g,j] = max_{y in Y_j} sim(g,y).
    Then D12 = M1 @ B2, where M1 is the sparse term-by-gene membership matrix.
    Repeat in the opposite direction and combine
    (D12 + D21.T) / (|X_i| + |Y_j|).  This is exact standard BMA, but reuses
    every gene-to-term best-match across all source terms.  It is most useful
    for large all-vs-all jobs where N_genes x N_terms memory, or chunked
    best-match blocks, are acceptable.

Prefix-coupled null builder (BmaNullBuilder.precompute_prefix)
    For each Monte Carlo iteration, sample full background permutations, compute
    A = E[perm1[:max_m]] @ E[perm2[:max_k]].T, and use cumulative maxima plus
    cumulative sums to update every requested prefix pair (m,k).  The cost
    changes from roughly ite * sum_{(m,k)} m*k*d to ite * max_m*max_k*d.
    Each prefix is a uniform sample without replacement, so every individual
    (m,k) null is marginally valid; estimates across sizes are correlated.

Numba fallback (compute_bma_numba)
    For very small sets where m*k < 400, BLAS launch overhead dominates.
    A pure-Numba triple loop avoids that overhead.  The threshold is tunable
    via --numba-threshold in andes.py.
"""

import os
from dataclasses import dataclass

import numpy as np
from numba import jit
from tqdm import tqdm
from itertools import chain
from scipy.stats import hypergeom
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from . import artifacts

# Worker-process globals — populated by _init_worker via ProcessPoolExecutor.
# Stored at module level so forked workers share the data without pickle round-trips.
_E_UNIT = None  # float32 (N, d) — full embedding matrix, loaded from mmap file
_POP1 = None  # int32   (P1,)  — background gene indices for gene-set-1
_POP2 = None  # int32   (P2,)  — background gene indices for gene-set-2
_ITE = None  # int     — Monte Carlo iterations per size pair
_SEED = None  # int     — master random seed


def _chunked(seq, n):
    """Yield successive non-overlapping chunks of size n from seq."""
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _chunked_by_cost(size_pairs, ite, n_workers, target_chunks_per_worker=4):
    """Partition size_pairs into cost-balanced chunks for parallel workers.

    Pairs are sorted descending by m*k (proxy for per-iteration GEMM cost),
    then distributed with a greedy bin-packing step: each pair goes to the
    chunk with the lowest accumulated cost so far.  This prevents all heavy
    pairs landing on one worker.

    target_chunks_per_worker controls granularity: more chunks → better
    balance but more IPC overhead.  The default of 4 is a good trade-off for
    most workloads.
    """
    pairs = sorted(size_pairs, key=lambda p: p[0] * p[1], reverse=True)
    if not pairs:
        return []

    n_chunks = min(len(pairs), max(1, int(n_workers) * target_chunks_per_worker))
    chunks = [[] for _ in range(n_chunks)]
    costs = [0] * n_chunks

    for pair in pairs:
        idx = min(range(n_chunks), key=costs.__getitem__)
        chunks[idx].append(pair)
        costs[idx] += int(pair[0]) * int(pair[1]) * int(ite)

    return [chunk for chunk in chunks if chunk]


def _hash_array(arr):
    """Return a 16-byte hex BLAKE2b digest of an array's shape, dtype, and data."""
    return artifacts.hash_array(arr)


# Memory cap for one batched GEMM workspace (X + Y + A tensors) in _bma_batch.
# Increasing this allows larger b (more iterations packed per matmul call) at
# the cost of peak RSS.  64 MB is conservative; 256 MB is fine on most nodes.
_BATCH_BYTES = 64 * 1024 * 1024


@jit(nopython=True, nogil=True, cache=True)
def _batch_fys_sample(perm, js, out):
    """Apply b independent partial Fisher-Yates samples on perm, restoring it after each.

    perm: int32 (N,) — identity [0..N-1] on entry and exit.
    js:   int64 (b, m) — swap targets; js[i, j] must lie in [j, N-1].
    out:  int32 (b, m) — receives the b sampled local-index vectors.

    Cost: O(b·m) numba ops — no O(N) work regardless of N.
    """
    b = js.shape[0]
    m = js.shape[1]
    for i in range(b):
        for j in range(m):
            t = js[i, j]
            tmp = perm[j]
            perm[j] = perm[t]
            perm[t] = tmp
            out[i, j] = perm[j]
        for j in range(m - 1, -1, -1):
            t = js[i, j]
            tmp = perm[j]
            perm[j] = perm[t]
            perm[t] = tmp


@jit(nopython=True, nogil=True, cache=True)
def _fys_sample_single(perm, js):
    """Single partial Fisher-Yates sample. O(m). Restores perm.

    perm: int32 (N,) — identity [0..N-1] on entry and exit.
    js:   int64 (m,) — swap targets; js[j] must lie in [j, N-1].
    returns int32 (m,) — sampled local indices drawn from [0, N).
    """
    m = js.shape[0]
    result = np.empty(m, dtype=np.int32)
    for j in range(m):
        t = js[j]
        tmp = perm[j]
        perm[j] = perm[t]
        perm[t] = tmp
        result[j] = perm[j]
    for j in range(m - 1, -1, -1):
        t = js[j]
        tmp = perm[j]
        perm[j] = perm[t]
        perm[t] = tmp
    return result


def _init_worker(
    e_unit_path,
    pop1_array,
    pop2_array,
    ite,
    seed,
    blas_threads_per_worker=1,
):
    """ProcessPoolExecutor initializer — runs once per worker process at fork.

    Loads the embedding from the temp .npy file into a private copy (not mmap)
    so that random-row gathers don't cause repeated page faults in each worker.
    pop1/pop2 are small int32 arrays passed directly as init args (pickling cost
    is negligible at typical background sizes).
    """
    blas_threads_per_worker = max(1, int(blas_threads_per_worker))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(blas_threads_per_worker)
    from threadpoolctl import threadpool_limits

    threadpool_limits(blas_threads_per_worker)
    global _E_UNIT, _POP1, _POP2, _ITE, _SEED
    # Full load (not mmap) so random-row gathers don't cause repeated page faults.
    _E_UNIT = np.load(e_unit_path).astype(np.float32, copy=False)
    _POP1 = pop1_array
    _POP2 = pop2_array
    _ITE = int(ite)
    _SEED = int(seed)


def _bma_batch(E, pop1, pop2, m, k, ite, rng, perm1, perm2):
    """Compute `ite` BMA null scores for (m, k) using batched GEMM.

    Sampling: one rng.integers call per batch (O(b·m+b·k) ints) fed into
    _batch_fys_sample (O(b·m) numba ops).  Previously b calls to
    rng.choice(N, m, replace=False) each did O(N) work — ~60-360x slower at
    production scale (N≈18k, m≈50-300).  GEMM path is unchanged.
    """
    N1 = perm1.shape[0]
    N2 = perm2.shape[0]
    d = E.shape[1]
    bytes_per_iter = (m + k) * d * 4 + m * k * 4
    max_b = max(1, _BATCH_BYTES // max(bytes_per_iter, 1))
    batch = min(ite, max_b)

    # Lower bounds for FYS: position j must swap with target in [j, N-1].
    m_low = np.arange(m, dtype=np.int64)
    k_low = np.arange(k, dtype=np.int64)

    xi = np.empty((batch, m), dtype=np.int32)
    yi = np.empty((batch, k), dtype=np.int32)
    scores = np.empty(ite, dtype=np.float32)
    done = 0
    while done < ite:
        b = min(batch, ite - done)

        # Vectorized swap-index generation: js[i, j] ∈ [j, N-1]. One rng call each.
        js_m = rng.integers(m_low, N1, size=(b, m), dtype=np.int64)
        js_k = rng.integers(k_low, N2, size=(b, k), dtype=np.int64)
        _batch_fys_sample(perm1, js_m, xi[:b])
        _batch_fys_sample(perm2, js_k, yi[:b])

        X = E[pop1[xi[:b]]]  # (b, m, d) fancy-index gather
        Y = E[pop2[yi[:b]]]  # (b, k, d)
        A = np.matmul(X, Y.swapaxes(-2, -1))  # (b, m, k) — batched GEMM
        scores[done : done + b] = (
            A.max(axis=2).sum(axis=1) + A.max(axis=1).sum(axis=1)
        ) / (m + k)
        done += b

    return float(scores.mean()), float(scores.std(ddof=1)) if ite > 1 else 0.0


def _compute_chunk(pairs_chunk):
    """Worker task: compute BMA null stats for a list of (m, k) pairs.

    Called by ProcessPoolExecutor.map; reads _E_UNIT/_POP1/_POP2/_ITE/_SEED
    from worker globals set by _init_worker.  Pairs are sorted cheapest-first
    within the chunk so that the FYS scratch arrays are sized to the largest
    pair only once.  Each pair gets a deterministic seed derived from
    (master_seed, m, k) so results are reproducible regardless of chunking.
    """
    if not pairs_chunk:
        return []
    # Allocate FYS scratch arrays once per chunk; _batch_fys_sample restores them.
    N1, N2 = _POP1.shape[0], _POP2.shape[0]
    perm1 = np.arange(N1, dtype=np.int32)
    perm2 = np.arange(N2, dtype=np.int32)
    out = []
    for m, k in sorted(pairs_chunk, key=lambda p: p[0] * p[1]):
        rng = np.random.default_rng(np.random.SeedSequence([_SEED, m, k]))
        mean, std = _bma_batch(_E_UNIT, _POP1, _POP2, m, k, _ITE, rng, perm1, perm2)
        out.append(((m, k), (mean, std)))
    return out


class BMAWorkspaceMax:
    """Pre-allocated scratch buffers for BMA scoring, sized to (max_m, max_k).

    Allocate once at the start of a scoring loop and reuse across all term
    pairs.  views(m, k) returns sliced views into the underlying buffers so no
    new arrays are allocated per pair.
    """

    def __init__(self, max_m: int, max_k: int, d: int):
        """Allocate buffers large enough to hold any (m, k) pair up to (max_m, max_k)."""
        self.X = np.empty((max_m, d), dtype=np.float32)
        self.Y = np.empty((max_k, d), dtype=np.float32)
        self.A = np.empty((max_m, max_k), dtype=np.float32)
        self.row_max = np.empty(max_m, dtype=np.float32)
        self.col_max = np.empty(max_k, dtype=np.float32)

    def views(self, m: int, k: int):
        """Return (X[:m], Y[:k], A[:m,:k], row_max[:m], col_max[:k]) as views."""
        return (
            self.X[:m],
            self.Y[:k],
            self.A[:m, :k],
            self.row_max[:m],
            self.col_max[:k],
        )


def compute_bma_fast_ws_view(E_unit, X_idx, Y_idx, views):
    """Compute BMA using pre-allocated buffer views from BMAWorkspaceMax.views().

    np.take gathers rows without allocating new arrays; matmul writes into the
    pre-sized A view; max reductions write into row_max and col_max views.
    """
    X, Y, A, row_max, col_max = views

    np.take(E_unit, X_idx, axis=0, out=X)
    np.take(E_unit, Y_idx, axis=0, out=Y)

    np.matmul(X, Y.T, out=A)

    A.max(axis=1, out=row_max)
    A.max(axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


def compute_bma_blocks_ws(X, Y, views):
    """Compute BMA from pre-gathered embedding blocks X and Y.

    X : float32 (m, d) — pre-gathered block for gene set A (from term-block cache)
    Y : float32 (k, d) — pre-gathered block for gene set B
    views : tuple from BMAWorkspaceMax.views(m, k) — output buffers

    Skips the np.take gather step; suitable when blocks are already materialized.
    """
    _, _, A, row_max, col_max = views
    m = X.shape[0]
    k = Y.shape[0]
    A_view = A[:m, :k]
    row_view = row_max[:m]
    col_view = col_max[:k]

    np.matmul(X, Y.T, out=A_view)
    A_view.max(axis=1, out=row_view)
    A_view.max(axis=0, out=col_view)

    return float((row_view.sum() + col_view.sum()) / (m + k))


def hypergeom_test(terms, g1_term2index, g2_term2index, g1_population, g2_population):
    """One-sided hypergeometric p-value for gene-set overlap.

    Tests whether the overlap between two gene sets is larger than expected
    by chance given the shared background (intersection of g1 and g2 populations).

    Parameters
    ----------
    terms : (str, str)
        (term1, term2) pair to test.
    g1_term2index, g2_term2index : dict
        Map term name → list of gene indices.
    g1_population, g2_population : iterable of int
        Background gene index sets for each database.

    Returns
    -------
    float
        Survival function P(X >= inter) from scipy.stats.hypergeom.
    """
    term1, term2 = terms
    indexes1 = set(g1_term2index[term1])
    indexes2 = set(g2_term2index[term2])

    inter = len(indexes1.intersection(indexes2))
    n = len(indexes1)
    m = len(indexes2)
    N = len(set(g1_population).intersection(g2_population))

    return hypergeom.sf(inter - 1, N, n, m)


def l2_normalize_rows(E, eps=1e-12):
    """L2-normalize embeddings so cosine similarity = dot product."""
    E = E.astype(np.float32, copy=False)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(norms, eps)


@dataclass(frozen=True)
class PackedTerms:
    """Canonical packed representation of a term collection.

    ``flat_indices[offsets[i]:offsets[i + 1]]`` contains the sorted unique
    embedding-row indices for ``terms[i]``.  Keeping memberships in four dense
    arrays avoids thousands of small embedding blocks and makes chunked
    best-match construction reusable by one-shot scoring and persistent indexes.
    """

    terms: tuple
    sizes: np.ndarray
    offsets: np.ndarray
    flat_indices: np.ndarray


def pack_term_indices(terms, term2indices):
    """Return deterministic, duplicate-free term memberships in packed form."""
    terms = tuple(terms)
    arrays = []
    for term in terms:
        values = np.asarray(term2indices[term])
        if values.ndim != 1:
            raise ValueError(f"term indices must be one-dimensional: {term}")
        if values.dtype.kind not in "iu":
            raise ValueError(f"term indices must be integers: {term}")
        int32_info = np.iinfo(np.int32)
        if values.size and (
            int(values.min()) < int32_info.min or int(values.max()) > int32_info.max
        ):
            raise ValueError(f"term index exceeds int32 range: {term}")
        values = np.unique(values.astype(np.int32, copy=False))
        if values.size == 0:
            raise ValueError(f"term has no embedding indices: {term}")
        arrays.append(values)

    sizes = np.asarray([values.size for values in arrays], dtype=np.int32)
    offsets = np.empty(len(terms) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    flat_indices = (
        np.concatenate(arrays).astype(np.int32, copy=False)
        if arrays
        else np.empty(0, dtype=np.int32)
    )
    return PackedTerms(
        terms=terms,
        sizes=sizes,
        offsets=offsets,
        flat_indices=np.ascontiguousarray(flat_indices),
    )


def packed_terms_equal(packed1, packed2):
    """Return True when two packed term axes are exactly identical."""
    return (
        packed1.terms == packed2.terms
        and np.array_equal(packed1.sizes, packed2.sizes)
        and np.array_equal(packed1.flat_indices, packed2.flat_indices)
    )


def precompute_term_embedding_blocks(E_unit, term2indices):
    """Gather and cache E_unit[idx] for every term as a C-contiguous float32 block.

    Calling this once at startup means query scoring never re-gathers rows from
    E_unit.  Memory cost is sum(|term|) * d * 4 bytes across all terms.
    """
    return {
        term: np.ascontiguousarray(E_unit[idx], dtype=np.float32)
        for term, idx in term2indices.items()
    }


def concatenate_term_embedding_blocks(terms, blocks):
    """Concatenate term embedding blocks into one (total_genes, d) matrix.

    Returns (concat, offsets, lengths) where:
      concat   : float32 (total_genes, d) — all term blocks stacked row-wise
      offsets  : int64 (len(terms)+1,)    — start row of each term in concat
      lengths  : int32 (len(terms),)      — number of genes per term

    Used by score_bma_zscore_matrix_batched so one GEMM covers all terms in
    gene-set-2 per row of gene-set-1.
    """
    lengths = np.asarray([blocks[t].shape[0] for t in terms], dtype=np.int32)
    offsets = np.empty(len(terms) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    if len(terms) == 0:
        return np.empty((0, 0), dtype=np.float32), offsets, lengths

    d = next(iter(blocks.values())).shape[1]
    concat = np.empty((int(offsets[-1]), d), dtype=np.float32)
    for term, start, end in zip(terms, offsets[:-1], offsets[1:]):
        concat[start:end] = blocks[term]
    return concat, offsets, lengths


def same_index_arrays_by_term(terms1, term2indices1, terms2, term2indices2):
    """Return True iff both axes contain identical terms and gene memberships.

    Used to detect symmetric scoring (same database on both axes), which halves
    computation by scoring only the upper triangle.  Membership comparison is
    order-independent so legacy unsorted arrays cannot hide symmetry.
    """
    if len(terms1) != len(terms2):
        return False
    for t1, t2 in zip(terms1, terms2):
        if t1 != t2:
            return False
        values1 = np.unique(np.asarray(term2indices1[t1], dtype=np.int32))
        values2 = np.unique(np.asarray(term2indices2[t2], dtype=np.int32))
        if not np.array_equal(values1, values2):
            return False
    return True


def _zscore_from_cache(cache, true_score, m, k):
    """Return (true_score - mu) / sigma from the null cache for size pair (m, k).

    Returns 0.0 when sigma is zero.  Raises KeyError if (m, k) is not cached.
    """
    mean_null, std_null = cache[(m, k)]
    return 0.0 if std_null == 0 else (true_score - mean_null) / std_null


def _null_lookup_tables(sizes1, sizes2, cache):
    """Build compact mean/std lookup grids for the requested size ranges."""
    sizes1 = np.asarray(sizes1, dtype=np.int32)
    sizes2 = np.asarray(sizes2, dtype=np.int32)
    max_m = int(sizes1.max()) if sizes1.size else 0
    max_k = int(sizes2.max()) if sizes2.size else 0
    requested = {(int(m), int(k)) for m in np.unique(sizes1) for k in np.unique(sizes2)}
    missing = [pair for pair in requested if pair not in cache]
    if missing:
        raise KeyError(missing[0])

    mu = np.zeros((max_m + 1, max_k + 1), dtype=np.float32)
    sigma = np.ones((max_m + 1, max_k + 1), dtype=np.float32)
    for (m, k), (mean, std) in cache.items():
        if int(m) <= max_m and int(k) <= max_k:
            mu[int(m), int(k)] = mean
            sigma[int(m), int(k)] = std
    return sizes1, sizes2, mu, sigma


def zscore_matrix_from_cache(true_scores, sizes1, sizes2, cache, out=None):
    """Normalize a true-score matrix without full mean/std broadcast temporaries.

    The cache lookup tables are at most ``(max_size1 + 1, max_size2 + 1)``.
    Rows are normalized in place into ``out``, reducing peak memory from several
    full score matrices to one output row plus the required output itself.
    """
    true_scores = np.asarray(true_scores)
    sizes1, sizes2, mu, sigma = _null_lookup_tables(sizes1, sizes2, cache)
    expected_shape = (sizes1.size, sizes2.size)
    if true_scores.shape != expected_shape:
        raise ValueError(
            f"true_scores shape {true_scores.shape} does not match {expected_shape}"
        )
    if out is None:
        out = np.empty(expected_shape, dtype=np.float32)
    elif out.shape != expected_shape:
        raise ValueError(f"out shape {out.shape} does not match {expected_shape}")

    for i, m in enumerate(sizes1):
        mu_row = mu[int(m), sizes2]
        sigma_row = sigma[int(m), sizes2]
        np.subtract(true_scores[i], mu_row, out=out[i], casting="unsafe")
        nonzero = sigma_row != 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(out[i], sigma_row, out=out[i], where=nonzero)
        out[i, ~nonzero] = 0.0
    return np.asarray(out, dtype=np.float32)


def combine_directed_scores(
    directed12,
    directed21,
    sizes1,
    sizes2,
    out=None,
):
    """Combine directed best-match sums into exact BMA scores row by row.

    ``directed12`` has shape ``(n_terms1, n_terms2)`` and ``directed21`` has
    shape ``(n_terms2, n_terms1)``.  The caller may supply ``out`` to reuse an
    existing float32 result allocation.
    """
    directed12 = np.asarray(directed12, dtype=np.float32)
    directed21 = np.asarray(directed21, dtype=np.float32)
    sizes1 = np.asarray(sizes1, dtype=np.int32)
    sizes2 = np.asarray(sizes2, dtype=np.int32)
    expected12 = (sizes1.size, sizes2.size)
    expected21 = (sizes2.size, sizes1.size)
    if directed12.shape != expected12 or directed21.shape != expected21:
        raise ValueError(
            "directed score shapes do not match term sizes: "
            f"{directed12.shape}, {directed21.shape}"
        )
    if out is None:
        out = np.empty(expected12, dtype=np.float32)
    elif out.shape != expected12:
        raise ValueError(f"out shape {out.shape} does not match {expected12}")

    sizes2_f = sizes2.astype(np.float32)
    for i, m in enumerate(sizes1):
        row = out[i]
        np.add(directed12[i], directed21[:, i], out=row)
        np.divide(row, float(m) + sizes2_f, out=row)
    return out


def combine_directed_scores_and_zscore(
    directed12,
    directed21,
    sizes1,
    sizes2,
    cache,
    out=None,
):
    """Compatibility helper that standardizes exact combined scores in place."""
    true_scores = combine_directed_scores(
        directed12,
        directed21,
        sizes1,
        sizes2,
        out=out,
    )
    return zscore_matrix_from_cache(
        true_scores,
        sizes1,
        sizes2,
        cache,
        out=true_scores,
    )


def score_bma_matrix_pairwise(
    E_unit,
    terms1,
    terms2,
    indices1,
    indices2,
    blocks1=None,
    blocks2=None,
    symmetric=False,
    n_workers=1,
    numba_threshold=400,
    show_progress=False,
):
    """Compute the exact full BMA matrix with the pairwise engine.

    Row-level parallelism uses ThreadPoolExecutor (NumPy/BLAS releases the
    GIL).  Null calibration is deliberately not part of this function.

    Parallelism: rows are split into n_workers contiguous slices so only
    n_workers Futures are created (not one per row).  Each worker allocates one
    BMAWorkspaceMax shared across its entire slice, avoiding repeated allocs.

    Scorer selection per pair:
      - m*k < numba_threshold → compute_bma_numba (avoids BLAS launch overhead)
      - blocks provided       → compute_bma_blocks_ws (pre-gathered blocks)
      - otherwise             → compute_bma_fast_ws_view (gather + matmul)

    With symmetric=True (same database on both axes), only the upper triangle
    is computed and mirrored, halving work.

    Parameters
    ----------
    E_unit : float32 (N, d)
    terms1, terms2 : list of str
    indices1, indices2 : dict {term: int32 array of gene indices}
    blocks1, blocks2 : dict {term: float32 (m, d)}, optional
    symmetric : bool
    n_workers : int — 0 or 1 for single-threaded
    numba_threshold : int — use Numba when m*k < this value
    show_progress : bool

    Returns
    -------
    numpy.ndarray, float32 (len(terms1), len(terms2))
    """
    true_scores = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
    max_m = max(len(indices1[t]) for t in terms1)
    max_k = max(len(indices2[t]) for t in terms2)
    use_blocks = blocks1 is not None and blocks2 is not None
    d = E_unit.shape[1]

    def score_slice(row_indices):
        # One workspace per slice; BMAWorkspaceMax sized to max over all terms1.
        ws = BMAWorkspaceMax(max_m, max_k, d)
        results = []
        for i in row_indices:
            t1 = terms1[i]
            X_idx = indices1[t1]
            m = len(X_idx)
            start_term = i if symmetric else 0
            row = np.zeros(len(terms2) - start_term, dtype=np.float32)
            mirrored = []

            for j, t2 in enumerate(terms2[start_term:], start=start_term):
                Y_idx = indices2[t2]
                k = len(Y_idx)

                if m * k < numba_threshold:
                    true_score = compute_bma_numba(E_unit, X_idx, Y_idx)
                elif use_blocks:
                    true_score = compute_bma_blocks_ws(
                        blocks1[t1], blocks2[t2], ws.views(m, k)
                    )
                else:
                    true_score = compute_bma_fast_ws_view(
                        E_unit, X_idx, Y_idx, ws.views(m, k)
                    )

                row[j - start_term] = true_score
                if symmetric and i != j:
                    mirrored.append((j, true_score))

            results.append((i, start_term, row, mirrored))
        return results

    all_rows = list(range(len(terms1)))

    if n_workers and n_workers > 1:
        slices = [s.tolist() for s in np.array_split(all_rows, n_workers) if len(s)]
        with ThreadPoolExecutor(max_workers=len(slices)) as ex:
            futures = [ex.submit(score_slice, s) for s in slices]
            for fut in tqdm(
                as_completed(futures),
                total=len(slices),
                desc="Computing",
                disable=not show_progress,
            ):
                for i, start_term, row, mirrored in fut.result():
                    true_scores[i, start_term : start_term + len(row)] = row
                    for j, value in mirrored:
                        true_scores[j, i] = value
    else:
        for i, start_term, row, mirrored in tqdm(
            score_slice(all_rows),
            total=len(all_rows),
            desc="Computing",
            disable=not show_progress,
        ):
            true_scores[i, start_term : start_term + len(row)] = row
            for j, value in mirrored:
                true_scores[j, i] = value

    return true_scores


def score_bma_zscore_matrix(
    E_unit,
    terms1,
    terms2,
    indices1,
    indices2,
    null_cache,
    blocks1=None,
    blocks2=None,
    symmetric=False,
    n_workers=1,
    numba_threshold=400,
    show_progress=False,
):
    """Compatibility wrapper for exact pairwise scoring plus calibration."""
    true_scores = score_bma_matrix_pairwise(
        E_unit,
        terms1,
        terms2,
        indices1,
        indices2,
        blocks1=blocks1,
        blocks2=blocks2,
        symmetric=symmetric,
        n_workers=n_workers,
        numba_threshold=numba_threshold,
        show_progress=show_progress,
    )
    sizes1 = np.asarray([len(indices1[term]) for term in terms1], dtype=np.int32)
    sizes2 = np.asarray([len(indices2[term]) for term in terms2], dtype=np.int32)
    return zscore_matrix_from_cache(
        true_scores,
        sizes1,
        sizes2,
        null_cache.cache,
        out=true_scores,
    )


def _term_ranges_for_workspace(lengths, max_m, max_workspace_mb):
    """Split term columns so one row workspace stays under the memory cap."""
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
        terms = 0
        for i, length in enumerate(lengths):
            next_cols = cols + int(length)
            next_terms = terms + 1
            next_bytes = max_m * next_cols * 4 + max_m * next_terms * 4 + next_cols * 4
            if terms and next_bytes > budget_bytes:
                ranges.append((start, i))
                start = i
                cols = int(length)
                terms = 1
            else:
                cols = next_cols
                terms = next_terms
        ranges.append((start, n_terms))

    max_bytes = 0
    for start, end in ranges:
        cols = int(lengths[start:end].sum())
        terms = end - start
        max_bytes = max(
            max_bytes,
            max_m * cols * 4 + max_m * terms * 4 + cols * 4,
        )
    return ranges, max_bytes / 1e6


def _bestmatch_chunk_bytes(
    lengths,
    start,
    end,
    source_rows,
    embedding_dim,
    aggregate_rows=0,
):
    """Estimate peak live arrays for one packed best-match term chunk."""
    occurrences = int(np.asarray(lengths[start:end], dtype=np.int64).sum())
    terms = int(end - start)
    itemsize = np.dtype(np.float32).itemsize

    # During construction, the gathered target embeddings, similarity matrix,
    # and reduced best-match block coexist. Before yielding, the first two are
    # released. A consumer may then aggregate B_chunk into source-term rows.
    compute_bytes = itemsize * (
        occurrences * int(embedding_dim)
        + int(source_rows) * occurrences
        + int(source_rows) * terms
    )
    consume_bytes = itemsize * (int(source_rows) * terms + int(aggregate_rows) * terms)
    return max(compute_bytes, consume_bytes)


def bestmatch_term_ranges(
    lengths,
    source_rows,
    embedding_dim,
    max_workspace_mb,
    aggregate_rows=0,
):
    """Plan packed target-term chunks against a temporary-array budget.

    The budget covers the gathered target embeddings, similarity matrix,
    reduced B chunk, and an optional dense aggregation result. It intentionally
    excludes persistent source embeddings, sparse memberships, directed score
    matrices, and the final output; callers report those separately.
    """
    lengths = np.asarray(lengths, dtype=np.int32)
    n_terms = int(lengths.size)
    if n_terms == 0:
        return [], 0.0

    if not max_workspace_mb or max_workspace_mb <= 0:
        ranges = [(0, n_terms)]
    else:
        budget_bytes = float(max_workspace_mb) * 1e6
        ranges = []
        start = 0
        for end in range(1, n_terms + 1):
            candidate = _bestmatch_chunk_bytes(
                lengths,
                start,
                end,
                source_rows,
                embedding_dim,
                aggregate_rows,
            )
            if end - start > 1 and candidate > budget_bytes:
                ranges.append((start, end - 1))
                start = end - 1
        ranges.append((start, n_terms))

    peak_bytes = max(
        _bestmatch_chunk_bytes(
            lengths,
            start,
            end,
            source_rows,
            embedding_dim,
            aggregate_rows,
        )
        for start, end in ranges
    )
    return ranges, peak_bytes / 1e6


def build_packed_membership_matrix(packed, n_genes, gene_universe=None):
    """Return a CSR membership matrix from packed term memberships.

    When ``gene_universe`` is supplied, columns are remapped to that sorted
    subset of global embedding rows.  One-shot scoring uses this to avoid
    constructing best-match rows that the membership multiplication can never
    consume.  Persistent indexes omit it because arbitrary future query genes
    require all embedding rows.
    """
    n_genes = int(n_genes)
    n_terms = len(packed.terms)
    if n_terms == 0:
        n_cols = n_genes if gene_universe is None else len(gene_universe)
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)

    if packed.flat_indices.size and (
        packed.flat_indices.min() < 0 or packed.flat_indices.max() >= n_genes
    ):
        raise IndexError("term membership contains an out-of-range gene index")

    row_idx = np.repeat(
        np.arange(n_terms, dtype=np.int32), packed.sizes.astype(np.int64)
    )
    if gene_universe is None:
        col_idx = packed.flat_indices
        n_cols = n_genes
    else:
        gene_universe = np.asarray(gene_universe, dtype=np.int32)
        if gene_universe.ndim != 1:
            raise ValueError("gene_universe must be one-dimensional")
        if gene_universe.size and (
            gene_universe.min() < 0 or gene_universe.max() >= n_genes
        ):
            raise IndexError("gene_universe contains an out-of-range gene index")
        if gene_universe.size > 1 and np.any(gene_universe[1:] <= gene_universe[:-1]):
            raise ValueError("gene_universe must be sorted and unique")
        global_to_local = np.full(n_genes, -1, dtype=np.int32)
        global_to_local[gene_universe] = np.arange(gene_universe.size, dtype=np.int32)
        col_idx = global_to_local[packed.flat_indices]
        if np.any(col_idx < 0):
            raise ValueError("gene_universe does not contain every term member")
        n_cols = gene_universe.size

    data = np.ones(col_idx.shape[0], dtype=np.float32)
    return sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n_terms, n_cols), dtype=np.float32
    )


def build_term_membership_matrix(terms, term2indices, n_genes):
    """Return a deterministic CSR term-by-gene membership matrix."""
    return build_packed_membership_matrix(
        pack_term_indices(terms, term2indices), n_genes
    )


def iter_gene_to_term_best_match_chunks(
    E_source,
    E_unit,
    packed,
    max_workspace_mb=1024,
    show_progress=False,
    aggregate_rows=0,
):
    """Yield exact ``B[:, start:end]`` chunks for a packed target database.

    Target embeddings are gathered once per chunk from ``flat_indices``.  The
    complete occurrence-concatenated target matrix and complete best-match
    matrix are never materialized.
    """
    E_source = np.ascontiguousarray(E_source, dtype=np.float32)
    E_unit = np.asarray(E_unit, dtype=np.float32)
    if E_source.ndim != 2 or E_unit.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if E_source.shape[1] != E_unit.shape[1]:
        raise ValueError("source and target embedding dimensions differ")
    if packed.flat_indices.size and (
        packed.flat_indices.min() < 0 or packed.flat_indices.max() >= E_unit.shape[0]
    ):
        raise IndexError("packed target contains an out-of-range gene index")

    term_ranges, _ = bestmatch_term_ranges(
        packed.sizes,
        E_source.shape[0],
        E_source.shape[1],
        max_workspace_mb,
        aggregate_rows=aggregate_rows,
    )
    iterator = tqdm(
        term_ranges,
        desc="Best-match blocks",
        disable=not show_progress,
    )
    for start, end in iterator:
        flat_start = int(packed.offsets[start])
        flat_end = int(packed.offsets[end])
        target_idx = packed.flat_indices[flat_start:flat_end]
        target_emb = np.ascontiguousarray(E_unit[target_idx], dtype=np.float32)
        sims = E_source @ target_emb.T
        local_offsets = packed.offsets[start:end] - flat_start
        best = np.maximum.reduceat(sims, local_offsets, axis=1)
        # The consumer needs only B_chunk. Releasing the larger construction
        # buffers before suspension prevents them overlapping sparse aggregation
        # or a memory-mapped write.
        del sims, target_emb
        yield start, end, np.asarray(best, dtype=np.float32)


def gene_to_term_best_match_matrix(
    E_unit,
    terms,
    blocks,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Compute B[g, t] = max similarity from gene g to any gene in term t.

    This is the reusable directed best-match matrix for exact all-vs-all ANDES
    scoring.  Terms are processed in contiguous chunks so the temporary
    similarity workspace is bounded by max_workspace_mb.
    """
    n_genes = E_unit.shape[0]
    best = np.empty((n_genes, len(terms)), dtype=np.float32)
    concat, offsets, lengths = concatenate_term_embedding_blocks(terms, blocks)
    term_ranges, workspace_mb = _term_ranges_for_workspace(
        lengths, n_genes, max_workspace_mb
    )

    iterator = tqdm(
        term_ranges,
        desc="Best-match blocks",
        disable=not show_progress,
    )
    for start, end in iterator:
        col_start = int(offsets[start])
        col_end = int(offsets[end])
        local_offsets = offsets[start:end] - col_start

        sims = E_unit @ concat[col_start:col_end].T
        best[:, start:end] = np.maximum.reduceat(sims, local_offsets, axis=1)

    return best, workspace_mb


def score_bma_matrix_bestmatch(
    E_unit,
    terms1,
    terms2,
    indices1,
    indices2,
    blocks1=None,
    blocks2=None,
    symmetric=False,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Return exact streamed all-vs-all BMA scores and workspace statistics.

    This preserves the ANDES score exactly for standard BMA:

      D12 = M1 @ B2, where B2[g, j] = max_{y in term2[j]} sim(g, y)
      D21 = M2 @ B1, where B1[g, i] = max_{x in term1[i]} sim(g, x)

    Only source genes referenced by M1/M2 receive best-match rows, target terms
    are packed into flat index arrays, and each B chunk is immediately consumed
    by sparse aggregation into D.  The complete B matrix is never retained.
    Final score combination is performed row-wise without a denominator
    broadcast matrix.  Null calibration is intentionally a separate operation.

    ``blocks1`` and ``blocks2`` remain accepted for API compatibility but are
    intentionally unused; the packed path supersedes the per-term block cache.
    """
    n_genes = E_unit.shape[0]
    use_symmetric_reuse = bool(symmetric) and same_index_arrays_by_term(
        terms1, indices1, terms2, indices2
    )
    packed1 = pack_term_indices(terms1, indices1)
    packed2 = packed1 if use_symmetric_reuse else pack_term_indices(terms2, indices2)
    if use_symmetric_reuse and not packed_terms_equal(packed1, packed2):
        use_symmetric_reuse = False
        packed2 = pack_term_indices(terms2, indices2)

    sizes1_i = packed1.sizes
    sizes2_i = packed2.sizes

    def directed_scores(source_packed, target_packed):
        source_genes = np.unique(source_packed.flat_indices)
        membership = build_packed_membership_matrix(
            source_packed,
            n_genes,
            gene_universe=source_genes,
        )
        source_embeddings = np.ascontiguousarray(E_unit[source_genes], dtype=np.float32)
        directed = np.empty(
            (len(source_packed.terms), len(target_packed.terms)),
            dtype=np.float32,
        )
        for start, end, best_chunk in iter_gene_to_term_best_match_chunks(
            source_embeddings,
            E_unit,
            target_packed,
            max_workspace_mb=max_workspace_mb,
            show_progress=show_progress,
            aggregate_rows=len(source_packed.terms),
        ):
            directed[:, start:end] = membership @ best_chunk
        _, chunk_workspace_mb = bestmatch_term_ranges(
            target_packed.sizes,
            source_genes.size,
            E_unit.shape[1],
            max_workspace_mb,
            aggregate_rows=len(source_packed.terms),
        )
        logical_bestmatch_mb = source_genes.size * len(target_packed.terms) * 4 / 1e6
        membership_mb = (
            membership.data.nbytes
            + membership.indices.nbytes
            + membership.indptr.nbytes
        ) / 1e6
        source_embedding_mb = source_embeddings.nbytes / 1e6
        directed_matrix_mb = directed.nbytes / 1e6
        fixed_direction_mb = (
            source_genes.nbytes / 1e6 + membership_mb + source_embedding_mb
        )
        return directed, {
            "source_genes": int(source_genes.size),
            "chunk_workspace_mb": float(chunk_workspace_mb),
            "logical_bestmatch_mb": float(logical_bestmatch_mb),
            "membership_mb": float(membership_mb),
            "source_embedding_mb": float(source_embedding_mb),
            "directed_matrix_mb": float(directed_matrix_mb),
            "fixed_direction_mb": float(fixed_direction_mb),
            "direction_peak_mb": float(
                fixed_direction_mb + directed_matrix_mb + chunk_workspace_mb
            ),
        }

    if use_symmetric_reuse:
        D, direction = directed_scores(packed1, packed1)
        true_scores = combine_directed_scores(
            D,
            D,
            sizes1_i,
            sizes1_i,
        )
        output_matrix_mb = true_scores.nbytes / 1e6
        estimated_peak_mb = max(
            direction["direction_peak_mb"],
            direction["directed_matrix_mb"] + output_matrix_mb,
        )
        return true_scores, {
            # workspace_mb is retained for callers, but now reports the complete
            # estimated peak of query-owned arrays rather than only GEMM scratch.
            "workspace_mb": float(estimated_peak_mb),
            "estimated_peak_mb": float(estimated_peak_mb),
            "chunk_workspace_mb": direction["chunk_workspace_mb"],
            "directed_matrix_mb": direction["directed_matrix_mb"],
            "output_matrix_mb": float(output_matrix_mb),
            "fixed_direction_mb": direction["fixed_direction_mb"],
            "bestmatch1_mb": direction["logical_bestmatch_mb"],
            "bestmatch2_mb": direction["logical_bestmatch_mb"],
            "bestmatch_materialized_mb": 0.0,
            "source_genes1": direction["source_genes"],
            "source_genes2": direction["source_genes"],
            "membership_occurrences1": int(packed1.flat_indices.size),
            "membership_occurrences2": int(packed1.flat_indices.size),
            "symmetric_reuse": True,
        }

    D12, direction12 = directed_scores(packed1, packed2)
    D21, direction21 = directed_scores(packed2, packed1)
    true_scores = combine_directed_scores(
        D12,
        D21,
        sizes1_i,
        sizes2_i,
    )
    matrix_mb = D12.nbytes / 1e6
    output_matrix_mb = true_scores.nbytes / 1e6
    estimated_peak_mb = max(
        direction12["direction_peak_mb"],
        # D12 remains resident while the reverse directed matrix is built.
        matrix_mb + direction21["direction_peak_mb"],
        # Both directed matrices remain resident while the output is filled.
        2.0 * matrix_mb + output_matrix_mb,
    )

    return true_scores, {
        "workspace_mb": float(estimated_peak_mb),
        "estimated_peak_mb": float(estimated_peak_mb),
        "chunk_workspace_mb": max(
            direction12["chunk_workspace_mb"],
            direction21["chunk_workspace_mb"],
        ),
        "directed_matrix_mb": float(matrix_mb),
        "output_matrix_mb": float(output_matrix_mb),
        "fixed_direction_mb": max(
            direction12["fixed_direction_mb"],
            direction21["fixed_direction_mb"],
        ),
        "bestmatch1_mb": direction21["logical_bestmatch_mb"],
        "bestmatch2_mb": direction12["logical_bestmatch_mb"],
        "bestmatch_materialized_mb": 0.0,
        "source_genes1": direction12["source_genes"],
        "source_genes2": direction21["source_genes"],
        "membership_occurrences1": int(packed1.flat_indices.size),
        "membership_occurrences2": int(packed2.flat_indices.size),
        "symmetric_reuse": False,
    }


def score_bma_zscore_matrix_bestmatch(
    E_unit,
    terms1,
    terms2,
    indices1,
    indices2,
    null_cache,
    blocks1=None,
    blocks2=None,
    symmetric=False,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Compatibility wrapper returning calibrated streamed best-match scores.

    Exact scoring is delegated to :func:`score_bma_matrix_bestmatch`.  The
    returned score allocation is then standardized in place, so separating the
    APIs does not add another full matrix to peak memory.
    """
    true_scores, stats = score_bma_matrix_bestmatch(
        E_unit,
        terms1,
        terms2,
        indices1,
        indices2,
        blocks1=blocks1,
        blocks2=blocks2,
        symmetric=symmetric,
        max_workspace_mb=max_workspace_mb,
        show_progress=show_progress,
    )
    sizes1 = np.asarray([len(indices1[term]) for term in terms1], dtype=np.int32)
    sizes2 = np.asarray([len(indices2[term]) for term in terms2], dtype=np.int32)
    zscores = zscore_matrix_from_cache(
        true_scores,
        sizes1,
        sizes2,
        null_cache.cache,
        out=true_scores,
    )
    return zscores, stats


def score_bma_matrix_batched(
    terms1,
    terms2,
    blocks1,
    blocks2,
    symmetric=False,
    n_workers=1,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Compute exact BMA scores using one GEMM per gene-set-1 row.

    Key optimization over score_bma_zscore_matrix: all gene-set-2 blocks are
    concatenated once into a single (total_genes2, d) matrix.  For each row
    (gene set A of size m), one matmul (m, d) @ (d, total_genes2) produces
    all similarities in one call.  np.maximum.reduceat then extracts the
    per-term row-max without looping over individual terms.

    Worker count is capped so that all active workers fit within
    max_workspace_mb total (each needs one (m, total_genes2) workspace).

    Requires blocks1 and blocks2 to be pre-computed (precompute_term_embedding_blocks).

    Parameters
    ----------
    terms1, terms2 : list of str
    blocks1, blocks2 : dict {term: float32 (m, d)}
    symmetric : bool
    n_workers : int
    max_workspace_mb : float — memory budget for all concurrent workspaces
    show_progress : bool

    Returns
    -------
    true_scores : float32 (len(terms1), len(terms2))
    effective_workers : int — actual worker count after memory cap
    workspace_mb : float — per-worker workspace size in MB
    """
    true_scores = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
    concat2, offsets2, lengths2 = concatenate_term_embedding_blocks(terms2, blocks2)

    max_m = max(blocks1[t].shape[0] for t in terms1)
    term_ranges, workspace_mb = _term_ranges_for_workspace(
        lengths2, max_m, max_workspace_mb
    )
    if max_workspace_mb and max_workspace_mb > 0:
        effective_workers = max(
            1, min(int(n_workers), int(max_workspace_mb // max(workspace_mb, 1e-9)))
        )
    else:
        effective_workers = max(1, int(n_workers))

    def score_row(i):
        X = blocks1[terms1[i]]
        m = X.shape[0]
        start_term = i if symmetric else 0
        score_chunks = []
        mirrored = []
        for range_start, range_end in term_ranges:
            chunk_start = max(range_start, start_term)
            if chunk_start >= range_end:
                continue

            col_start = int(offsets2[chunk_start])
            col_end = int(offsets2[range_end])
            Y = concat2[col_start:col_end]
            local_offsets = offsets2[chunk_start:range_end] - col_start
            lengths = lengths2[chunk_start:range_end]

            A = X @ Y.T
            row_term_max = np.maximum.reduceat(A, local_offsets, axis=1)
            row_sums = row_term_max.sum(axis=0)
            col_max = A.max(axis=0)
            col_sums = np.add.reduceat(col_max, local_offsets)
            scores = (row_sums + col_sums) / (m + lengths)

            exact = np.asarray(scores, dtype=np.float32)
            for p, score in enumerate(scores):
                j = chunk_start + p
                if symmetric and i != j:
                    mirrored.append((j, float(score)))
            score_chunks.append(exact)
        exact_scores = (
            np.concatenate(score_chunks)
            if score_chunks
            else np.empty(0, dtype=np.float32)
        )
        return i, start_term, exact_scores, mirrored

    def score_slice(row_indices):
        return [score_row(i) for i in row_indices]

    all_rows = list(range(len(terms1)))

    if effective_workers > 1:
        slices = [
            s.tolist() for s in np.array_split(all_rows, effective_workers) if len(s)
        ]
        with ThreadPoolExecutor(max_workers=len(slices)) as ex:
            futures = [ex.submit(score_slice, s) for s in slices]
            for fut in tqdm(
                as_completed(futures),
                total=len(slices),
                desc="Computing",
                disable=not show_progress,
            ):
                for i, start_term, exact_scores, mirrored in fut.result():
                    true_scores[i, start_term : start_term + len(exact_scores)] = (
                        exact_scores
                    )
                    for j, value in mirrored:
                        true_scores[j, i] = value
    else:
        for i, start_term, exact_scores, mirrored in tqdm(
            score_slice(all_rows),
            total=len(all_rows),
            desc="Computing",
            disable=not show_progress,
        ):
            true_scores[i, start_term : start_term + len(exact_scores)] = exact_scores
            for j, value in mirrored:
                true_scores[j, i] = value

    return true_scores, effective_workers, workspace_mb


def score_bma_zscore_matrix_batched(
    terms1,
    terms2,
    null_cache,
    blocks1,
    blocks2,
    symmetric=False,
    n_workers=1,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Compatibility wrapper for exact batched scoring plus calibration."""
    true_scores, effective_workers, workspace_mb = score_bma_matrix_batched(
        terms1,
        terms2,
        blocks1,
        blocks2,
        symmetric=symmetric,
        n_workers=n_workers,
        max_workspace_mb=max_workspace_mb,
        show_progress=show_progress,
    )
    sizes1 = np.asarray([blocks1[term].shape[0] for term in terms1], dtype=np.int32)
    sizes2 = np.asarray([blocks2[term].shape[0] for term in terms2], dtype=np.int32)
    zscores = zscore_matrix_from_cache(
        true_scores,
        sizes1,
        sizes2,
        null_cache.cache,
        out=true_scores,
    )
    return zscores, effective_workers, workspace_mb


@jit(nopython=True, nogil=True, cache=True)
def compute_bma_numba(E_unit, X_idx, Y_idx):
    """
    Numba version for very small sets where BLAS overhead dominates.
    Only faster for m*k < 400 (e.g., both sets < 20 genes).
    """
    m, k = len(X_idx), len(Y_idx)
    d = E_unit.shape[1]

    # Allocate similarity matrix once
    A = np.empty((m, k), dtype=np.float32)

    # Compute all similarities (do it once, not twice!)
    for i in range(m):
        for j in range(k):
            dot = 0.0
            for dim in range(d):
                dot += E_unit[X_idx[i], dim] * E_unit[Y_idx[j], dim]
            A[i, j] = dot

    # Compute row maxes manually (Numba doesn't support axis parameter)
    row_max_sum = 0.0
    for i in range(m):
        max_val = A[i, 0]
        for j in range(1, k):
            if A[i, j] > max_val:
                max_val = A[i, j]
        row_max_sum += max_val

    # Compute column maxes manually
    col_max_sum = 0.0
    for j in range(k):
        max_val = A[0, j]
        for i in range(1, m):
            if A[i, j] > max_val:
                max_val = A[i, j]
        col_max_sum += max_val

    return (row_max_sum + col_max_sum) / (m + k)


class BmaNullBuilder:
    """Null distribution cache for BMA scores, keyed by (m, k) size pair.

    Each entry is (mu, sigma) estimated from Monte Carlo BMA scores over
    random gene sets drawn from the background pool.  The cache is serialized
    to a pickle file and validated on reload via a metadata dict containing
    BLAKE2b hashes of the embedding and population arrays, plus ite and seed.
    """

    def __init__(self):
        self.cache = {}  # {(m, k): (float mu, float sigma)}
        self.metadata = {}  # build parameters; checked on load

    @staticmethod
    def build_metadata(
        E_unit,
        population_idx1,
        population_idx2,
        ite,
        seed,
        null_sampling="per_size_pair",
    ):
        """Build the metadata dict that identifies a specific null cache build.

        Hashes E_unit, pop1, and pop2 so any change in inputs generates a
        different key.  metadata_matches uses this to detect stale caches.
        """
        return {
            "kind": "andes_bma_null",
            "version": 2,
            "embedding_hash": _hash_array(np.asarray(E_unit, dtype=np.float32)),
            "population1_hash": _hash_array(
                np.asarray(population_idx1, dtype=np.int32)
            ),
            "population2_hash": _hash_array(
                np.asarray(population_idx2, dtype=np.int32)
            ),
            "ite": int(ite),
            "seed": int(seed),
            "std_ddof": 1,
            "null_sampling": str(null_sampling),
        }

    def metadata_matches(self, expected):
        """Return (True, "") if all keys in expected match self.metadata.

        Returns (False, reason) on first mismatch or if metadata is empty.
        """
        if not self.metadata:
            return False, "cache has no metadata"
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                return False, f"metadata mismatch for {key}"
        return True, ""

    def _prepare_build(self, expected_metadata):
        """Set build metadata, clearing entries if existing cache metadata is stale."""
        if self.cache:
            metadata_ok, _ = self.metadata_matches(expected_metadata)
            if not metadata_ok:
                self.cache.clear()
        self.metadata = expected_metadata

    @staticmethod
    def resolve_seed(seed):
        """Return a concrete non-negative int seed.  Negative/None → OS entropy."""
        if seed is None or int(seed) < 0:
            return int(np.random.SeedSequence().entropy)
        return int(seed)

    def precompute_parallel(
        self,
        E_unit,
        population_idx,
        size_pairs,
        ite=1000,
        verbose=True,
        seed=12345,
        n_workers=8,
        chunk_size=None,
        population_idx2=None,
        show_progress=False,
        blas_threads_per_worker=1,
    ):
        """Build the null cache in parallel using ProcessPoolExecutor.

        Serializes E_unit to a temp .npy file so workers can load it without
        pickling the full array through the IPC channel.  Each worker process
        calls _compute_chunk, which runs _bma_batch for every (m, k) in its
        chunk.  Results are merged into self.cache as they arrive.

        chunk_size=None triggers cost-balanced chunking (_chunked_by_cost);
        pass an explicit integer for fixed-size chunks (useful for debugging).
        population_idx2=None means both axes draw from the same background pool.
        """
        seed = self.resolve_seed(seed)
        mmap_path = f"E_unit_{os.getpid()}.npy"

        try:
            np.save(mmap_path, np.asarray(E_unit, dtype=np.float32))
            pop1 = np.asarray(population_idx, dtype=np.int32)
            pop2 = np.asarray(
                population_idx if population_idx2 is None else population_idx2,
                dtype=np.int32,
            )
            expected_metadata = self.build_metadata(E_unit, pop1, pop2, ite, seed)
            self._prepare_build(expected_metadata)

            if chunk_size is None or chunk_size <= 0:
                chunks = _chunked_by_cost(size_pairs, ite, n_workers)
            else:
                chunks = list(_chunked(size_pairs, chunk_size))
            n_chunks = len(chunks)

            if verbose:
                print(f"Precomputing BMA null for {len(size_pairs)} size pairs")
                print(f"Iterations: {ite}, seed: {seed}")
                chunk_desc = (
                    "auto-cost" if chunk_size is None or chunk_size <= 0 else chunk_size
                )
                print(
                    f"Workers: {n_workers}, worker BLAS threads: "
                    f"{max(1, int(blas_threads_per_worker))}, "
                    f"chunk_size: {chunk_desc}"
                )

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(
                    mmap_path,
                    pop1,
                    pop2,
                    ite,
                    seed,
                    blas_threads_per_worker,
                ),
            ) as ex:
                it = ex.map(_compute_chunk, chunks)
                it = tqdm(
                    it,
                    total=n_chunks,
                    desc="BMA null",
                    disable=not show_progress
                    if show_progress is not None
                    else not verbose,
                )

                for chunk_res in it:
                    for key, val in chunk_res:
                        self.cache[key] = val
        finally:
            try:
                os.remove(mmap_path)
            except OSError:
                pass

    def precompute(
        self,
        E_unit,
        population_idx,
        size_pairs,
        ite=1000,
        verbose=True,
        seed=12345,
        population_idx2=None,
    ):
        """Single-process null cache build using the batched BMA kernel.

        Iterates over size pairs in sorted order.  For each pair (m, k),
        runs ite BMA iterations, then stores (mu, sigma) in self.cache.  Already-
        cached pairs are skipped when metadata matches, so this can extend an
        existing cache.

        Uses the same per-size-pair SeedSequence and _bma_batch kernel as
        precompute_parallel, so results are independent of which other size
        pairs are present in the cache.
        """
        seed = self.resolve_seed(seed)
        pop1 = np.asarray(population_idx, dtype=np.int32)
        pop2 = np.asarray(
            population_idx if population_idx2 is None else population_idx2,
            dtype=np.int32,
        )
        expected_metadata = self.build_metadata(E_unit, pop1, pop2, ite, seed)
        self._prepare_build(expected_metadata)

        size_pairs_sorted = sorted(size_pairs)
        if not size_pairs_sorted:
            return

        if verbose:
            print(f"Precomputing BMA null for {len(size_pairs_sorted)} size pairs")
            print(f"Iterations: {ite}, seed: {seed}")

        N1, N2 = len(pop1), len(pop2)
        perm1 = np.arange(N1, dtype=np.int32)
        perm2 = np.arange(N2, dtype=np.int32)

        for m, k in tqdm(size_pairs_sorted, desc="BMA null", disable=not verbose):
            if (m, k) in self.cache:
                continue

            rng = np.random.default_rng(np.random.SeedSequence([seed, int(m), int(k)]))
            mean, std = _bma_batch(E_unit, pop1, pop2, m, k, ite, rng, perm1, perm2)
            self.cache[(m, k)] = (mean, std)

        if verbose:
            cache_mb = len(self.cache) * 2 * 8 / 1e6
            print(f"Cached {len(self.cache)} distributions (~{cache_mb:.1f} MB)")

    def precompute_prefix(
        self,
        E_unit,
        population_idx,
        size_pairs,
        ite=1000,
        verbose=True,
        seed=12345,
        population_idx2=None,
    ):
        """Build BMA nulls with prefix-coupled random samples.

        Each Monte Carlo iteration draws full background permutations and uses
        prefixes up to the largest requested m and k.  Every prefix pair is a
        uniform sample without replacement, so each requested (m, k) null is
        marginally valid while one similarity matrix updates all sizes at once.

        This is an additive prototype.  Its metadata uses
        null_sampling="prefix_coupled" so these entries are not mixed with the
        baseline per-size-pair cache builder.
        """
        seed = self.resolve_seed(seed)
        pop1 = np.asarray(population_idx, dtype=np.int32)
        pop2 = np.asarray(
            population_idx if population_idx2 is None else population_idx2,
            dtype=np.int32,
        )
        expected_metadata = self.build_metadata(
            E_unit,
            pop1,
            pop2,
            ite,
            seed,
            null_sampling="prefix_coupled",
        )
        self._prepare_build(expected_metadata)

        size_pairs_sorted = sorted((int(m), int(k)) for m, k in size_pairs)
        missing = [pair for pair in size_pairs_sorted if pair not in self.cache]
        if not missing:
            return

        max_m = max(m for m, _ in missing)
        max_k = max(k for _, k in missing)
        if max_m > len(pop1) or max_k > len(pop2):
            raise ValueError("requested null size exceeds background population")

        if verbose:
            print(f"Prefix-coupled BMA null for {len(missing)} size pairs")
            print(f"Max prefix: {max_m} x {max_k}; iterations: {ite}, seed: {seed}")

        rng = np.random.default_rng(seed)
        mean = np.zeros((max_m, max_k), dtype=np.float64)
        M2 = np.zeros((max_m, max_k), dtype=np.float64)
        denom = (
            np.arange(1, max_m + 1, dtype=np.float64)[:, None]
            + np.arange(1, max_k + 1, dtype=np.float64)[None, :]
        )

        iterator = tqdm(range(ite), desc="BMA prefix null", disable=not verbose)
        for i in iterator:
            X_idx = pop1[rng.permutation(len(pop1))[:max_m]]
            Y_idx = pop2[rng.permutation(len(pop2))[:max_k]]

            A = E_unit[X_idx] @ E_unit[Y_idx].T
            row_best = np.maximum.accumulate(A, axis=1)
            row_sum = np.cumsum(row_best, axis=0, dtype=np.float64)
            col_best = np.maximum.accumulate(A, axis=0)
            col_sum = np.cumsum(col_best, axis=1, dtype=np.float64)
            scores = (row_sum + col_sum) / denom

            delta = scores - mean
            mean += delta / (i + 1)
            M2 += delta * (scores - mean)

        std_grid = (
            np.sqrt(M2 / (ite - 1))
            if ite > 1
            else np.zeros((max_m, max_k), dtype=np.float64)
        )
        for m, k in missing:
            self.cache[(m, k)] = (
                float(mean[m - 1, k - 1]),
                float(std_grid[m - 1, k - 1]),
            )

        if verbose:
            cache_mb = len(self.cache) * 2 * 8 / 1e6
            print(f"Cached {len(self.cache)} distributions (~{cache_mb:.1f} MB)")

    def get_zscore(self, true_score, m, k):
        """Return (true_score - mu) / sigma for size pair (m, k).

        Returns 0.0 when sigma is zero.  Raises KeyError if (m, k) not cached.
        """
        mean_null, std_null = self.cache[(m, k)]
        return 0.0 if std_null == 0 else (true_score - mean_null) / std_null

    @staticmethod
    def suggest_path(
        base_dir,
        E_unit,
        population_idx1,
        population_idx2,
        *,
        ite=1000,
        seed=12345,
        null_sampling="per_size_pair",
        ddof=1,
    ):
        """Return a content-addressed cache path under base_dir.

        Every resolved scientific null parameter contributes to the filename.
        """
        from .nulls import NullSpec

        spec = NullSpec(
            kind="bma",
            iterations=int(ite),
            seed=int(seed),
            sampling=str(null_sampling),
            ddof=int(ddof),
            embedding_hash=_hash_array(np.asarray(E_unit, dtype=np.float32)),
            population_hashes=(
                _hash_array(np.asarray(population_idx1, dtype=np.int32)),
                _hash_array(np.asarray(population_idx2, dtype=np.int32)),
            ),
        )
        return os.path.join(base_dir, f"bma_{spec.fingerprint}.null")

    def save_artifact(self, path, *, overwrite=False):
        """Persist this cache as a validated, non-pickled null artifact."""
        from .nulls import BmaNullModel

        BmaNullModel.from_builder(self).save(path, overwrite=overwrite)

    def load_artifact(self, path):
        """Populate this compatibility cache from a typed null artifact."""
        from .nulls import BmaNullModel

        model = BmaNullModel.load(path)
        self.cache = model.to_mapping()
        self.metadata = {
            "kind": "andes_bma_null",
            "version": 2,
            "embedding_hash": model.spec.embedding_hash,
            "population1_hash": model.spec.population_hashes[0],
            "population2_hash": model.spec.population_hashes[1],
            "ite": model.spec.iterations,
            "seed": model.spec.seed,
            "std_ddof": model.spec.ddof,
            "null_sampling": model.spec.sampling,
        }


def warmup_numba(warm_bma=True, warm_fys=True):
    """Compile all Numba-jitted functions in this module before timed code runs.

    Calls each kernel with minimal synthetic inputs.  Run this once at startup
    (before the null cache build or query loop) so JIT latency doesn't appear
    in benchmark timings. Call ``ranked.warmup_numba_es()`` separately
    if GSEA kernels are also in use.
    """
    warmed = []
    if warm_bma:
        dummy_E = np.random.default_rng(0).normal(size=(100, 50)).astype(np.float32)
        dummy_E = l2_normalize_rows(dummy_E)
        dummy_idx = np.arange(20, dtype=np.int64)
        _ = compute_bma_numba(dummy_E, dummy_idx, dummy_idx)
        warmed.append("BMA")

    if warm_fys:
        perm = np.arange(30, dtype=np.int32)
        js_s = np.array([0, 2, 3, 5], dtype=np.int64)
        _ = _fys_sample_single(perm, js_s)
        js_b = np.array([[0, 2, 3, 5], [1, 1, 4, 5]], dtype=np.int64)
        out = np.empty((2, 4), dtype=np.int32)
        _batch_fys_sample(perm, js_b, out)
        warmed.append("Fisher-Yates")

    if warmed:
        print(f"Numba compilation complete ({', '.join(warmed)}).")


def get_background_indices(geneset, node_set, g_node2index):
    """Return sorted embedding indices for all genes that appear in geneset and node_set.

    The background pool for null sampling is the union of all gene-set members
    that are present in the embedding, not the full embedding.
    """
    all_genes = set(chain.from_iterable(geneset.values()))
    all_genes.intersection_update(node_set)
    return sorted(g_node2index[x] for x in all_genes)


def preconvert_indices_to_arrays(geneset_indices):
    """Canonicalize legacy membership mappings as sorted int32 arrays."""
    return {
        term: np.unique(np.asarray(indices, dtype=np.int32))
        for term, indices in geneset_indices.items()
    }
