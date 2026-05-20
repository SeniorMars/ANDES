import os
import hashlib

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from numba import jit
import pickle
from tqdm import tqdm
from itertools import chain
from scipy.stats import hypergeom

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice

_E_UNIT = None
_POP1 = None
_POP2 = None
_ITE = None
_SEED = None


def _chunked(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _chunked_by_cost(size_pairs, ite, n_workers, target_chunks_per_worker=4):
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
    arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


_BATCH_BYTES = 64 * 1024 * 1024  # 64 MB per batched matmul


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


def _init_worker(e_unit_path, pop1_array, pop2_array, ite, seed):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
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

        X = E[pop1[xi[:b]]]                          # (b, m, d) fancy-index gather
        Y = E[pop2[yi[:b]]]                          # (b, k, d)
        A = np.matmul(X, Y.swapaxes(-2, -1))         # (b, m, k) — batched GEMM
        scores[done:done + b] = (A.max(axis=2).sum(axis=1) + A.max(axis=1).sum(axis=1)) / (m + k)
        done += b

    return float(scores.mean()), float(scores.std(ddof=1)) if ite > 1 else 0.0


def _compute_chunk(pairs_chunk):
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
    def __init__(self, max_m: int, max_k: int, d: int):
        self.X = np.empty((max_m, d), dtype=np.float32)
        self.Y = np.empty((max_k, d), dtype=np.float32)
        self.A = np.empty((max_m, max_k), dtype=np.float32)
        self.row_max = np.empty(max_m, dtype=np.float32)
        self.col_max = np.empty(max_k, dtype=np.float32)

    def views(self, m: int, k: int):
        return (
            self.X[:m],
            self.Y[:k],
            self.A[:m, :k],
            self.row_max[:m],
            self.col_max[:k],
        )


def compute_bma_fast_ws_view(E_unit, X_idx, Y_idx, views):
    X, Y, A, row_max, col_max = views

    np.take(E_unit, X_idx, axis=0, out=X)
    np.take(E_unit, Y_idx, axis=0, out=Y)

    np.matmul(X, Y.T, out=A)

    A.max(axis=1, out=row_max)
    A.max(axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


def compute_bma_blocks_ws(X, Y, views):
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
    """
    Given a pair of terms, two annotation dicts, and background populations,
    calculate the hypergeometric test p-value.
    """
    term1, term2 = terms
    indexes1 = set(g1_term2index[term1])
    indexes2 = set(g2_term2index[term2])

    inter = len(indexes1.intersection(indexes2))
    n = len(indexes1)
    m = len(indexes2)
    N = len(set(g1_population).intersection(g2_population))

    return hypergeom.sf(inter - 1, N, n, m)


def t_score(x_w, x_b1, x_b2):
    """
    Unequal-variance t-test between within-set scores (x_w) and two backgrounds (x_b1, x_b2).
    Implementation based on GIANT.
    """
    x_w = np.asarray(x_w)
    x_b1 = np.asarray(x_b1)
    x_b2 = np.asarray(x_b2)

    s_w = x_w.std()
    n_w = x_w.size

    n_b = x_b1.size + x_b2.size
    mean_b = (x_b1.sum() + x_b2.sum()) / n_b

    v_b = ((x_b1 - mean_b) ** 2).sum() + ((x_b2 - mean_b) ** 2).sum()
    v_b /= n_b

    s_x = np.sqrt(s_w**2 / n_w + v_b / n_b)
    mean_w = x_w.mean()
    return (mean_w - mean_b) / s_x


def mean_embedding(
    terms, g1_embedding, g2_embedding, g1_term2index, g2_term2index, distinct=False
):
    """
    Given a pair of terms, two annotation dicts, and two embedding matrices,
    calculate the cosine similarity between term centroids.
    """
    term1, term2 = terms

    idx1 = list(g1_term2index[term1])
    idx2 = list(g2_term2index[term2])

    if distinct:
        idx2 = list(set(idx2).difference(idx1))
        if len(idx2) < 10:
            return (0, 0)

    v1 = g1_embedding[idx1].mean(axis=0)
    v2 = g2_embedding[idx2].mean(axis=0)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    score = float(np.dot(v1, v2) / denom)
    return score


def mean_matrix(terms, matrix, g1_term2index, g2_term2index, distinct=False):
    """
    Given a pair of terms, two annotation dicts, and a pairwise similarity matrix,
    calculate mean score between two terms.
    """
    term1, term2 = terms
    idx1 = list(g1_term2index[term1])
    idx2 = list(g2_term2index[term2])

    if distinct:
        idx2 = list(set(idx2).difference(idx1))
        if len(idx2) < 10:
            return (0, 0)

    mat = np.asarray(matrix)
    score = mat[np.ix_(idx1, idx2)].mean()
    return score


def l2_normalize_rows(E, eps=1e-12):
    """L2-normalize embeddings so cosine similarity = dot product."""
    E = E.astype(np.float32, copy=False)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(norms, eps)


def precompute_term_embedding_blocks(E_unit, term2indices):
    return {
        term: np.ascontiguousarray(E_unit[idx], dtype=np.float32)
        for term, idx in term2indices.items()
    }


def concatenate_term_embedding_blocks(terms, blocks):
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
    if len(terms1) != len(terms2):
        return False
    for t1, t2 in zip(terms1, terms2):
        if t1 != t2:
            return False
        if not np.array_equal(term2indices1[t1], term2indices2[t2]):
            return False
    return True


def _zscore_from_cache(cache, true_score, m, k):
    mean_null, std_null = cache[(m, k)]
    return 0.0 if std_null == 0 else (true_score - mean_null) / std_null


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
    zscores = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
    max_m = max(len(indices1[t]) for t in terms1)
    max_k = max(len(indices2[t]) for t in terms2)
    cache = null_cache.cache
    use_blocks = blocks1 is not None and blocks2 is not None

    def score_row(i):
        t1 = terms1[i]
        X_idx = indices1[t1]
        m = len(X_idx)
        start_term = i if symmetric else 0
        row = np.zeros(len(terms2) - start_term, dtype=np.float32)
        mirrored = []
        ws = BMAWorkspaceMax(max_m, max_k, E_unit.shape[1])
        term_iter = enumerate(terms2[start_term:], start=start_term)

        for j, t2 in term_iter:
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

            row[j - start_term] = _zscore_from_cache(cache, true_score, m, k)
            if symmetric and i != j:
                mirrored.append((j, _zscore_from_cache(cache, true_score, k, m)))

        return i, start_term, row, mirrored

    if n_workers and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            rows = ex.map(score_row, range(len(terms1)))
            rows = tqdm(
                rows, total=len(terms1), desc="Computing", disable=not show_progress
            )
            for i, start_term, row, mirrored in rows:
                zscores[i, start_term : start_term + len(row)] = row
                for j, value in mirrored:
                    zscores[j, i] = value
    else:
        rows = tqdm(range(len(terms1)), desc="Computing", disable=not show_progress)
        for i in rows:
            row_i, start_term, row, mirrored = score_row(i)
            zscores[row_i, start_term : start_term + len(row)] = row
            for j, value in mirrored:
                zscores[j, row_i] = value

    return zscores


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
    zscores = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
    concat2, offsets2, lengths2 = concatenate_term_embedding_blocks(terms2, blocks2)
    cache = null_cache.cache

    max_m = max(blocks1[t].shape[0] for t in terms1)
    max_cols = int(offsets2[-1])
    max_terms = len(terms2)
    workspace_mb = (
        max_m * max_cols * 4
        + max_m * max_terms * 4
        + max_cols * 4
    ) / 1e6
    if max_workspace_mb and max_workspace_mb > 0:
        effective_workers = max(1, min(int(n_workers), int(max_workspace_mb // max(workspace_mb, 1e-9))))
    else:
        effective_workers = max(1, int(n_workers))

    def score_row(i):
        X = blocks1[terms1[i]]
        m = X.shape[0]
        start_term = i if symmetric else 0
        col_start = int(offsets2[start_term])
        Y = concat2[col_start:]
        local_offsets = offsets2[start_term:-1] - col_start
        lengths = lengths2[start_term:]

        A = X @ Y.T
        row_term_max = np.maximum.reduceat(A, local_offsets, axis=1)
        row_sums = row_term_max.sum(axis=0)
        col_max = A.max(axis=0)
        col_sums = np.add.reduceat(col_max, local_offsets)
        scores = (row_sums + col_sums) / (m + lengths)

        zvals = np.empty(scores.shape[0], dtype=np.float32)
        mirrored = []
        for p, score in enumerate(scores):
            j = start_term + p
            k = int(lengths[p])
            zvals[p] = _zscore_from_cache(cache, float(score), m, k)
            if symmetric and i != j:
                mirrored.append((j, _zscore_from_cache(cache, float(score), k, m)))
        return i, start_term, zvals, mirrored

    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            rows = ex.map(score_row, range(len(terms1)))
            rows = tqdm(
                rows, total=len(terms1), desc="Computing", disable=not show_progress
            )
            for i, start_term, zvals, mirrored in rows:
                zscores[i, start_term : start_term + len(zvals)] = zvals
                for j, value in mirrored:
                    zscores[j, i] = value
    else:
        rows = tqdm(range(len(terms1)), desc="Computing", disable=not show_progress)
        for i in rows:
            row_i, start_term, zvals, mirrored = score_row(i)
            zscores[row_i, start_term : start_term + len(zvals)] = zvals
            for j, value in mirrored:
                zscores[j, row_i] = value

    return zscores, effective_workers, workspace_mb


def _bma_workspace(m: int, k: int, d: int):
    X = np.empty((m, d), dtype=np.float32)
    Y = np.empty((k, d), dtype=np.float32)
    A = np.empty((m, k), dtype=np.float32)
    row_max = np.empty(m, dtype=np.float32)
    col_max = np.empty(k, dtype=np.float32)
    return X, Y, A, row_max, col_max


def compute_bma_fast_ws(E_unit, X_idx, Y_idx, ws):
    X, Y, A, row_max, col_max = ws
    # Fill X,Y without allocating new arrays
    np.take(E_unit, X_idx, axis=0, out=X)
    np.take(E_unit, Y_idx, axis=0, out=Y)

    # GEMM into a preallocated matrix
    np.matmul(X, Y.T, out=A)

    # reductions into preallocated vectors
    A.max(axis=1, out=row_max)
    A.max(axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


@jit(nopython=True, nogil=True)
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


@jit(nopython=True)
def compute_es_numba(E_unit, gene_set_idx, ranked_list_idx):
    """Numba version for small gene sets."""
    m = len(gene_set_idx)
    L = len(ranked_list_idx)
    d = E_unit.shape[1]

    # Compute max similarity for each position
    col_max = np.empty(L, dtype=np.float32)
    for j in range(L):
        max_sim = -1.0
        for i in range(m):
            dot = 0.0
            for dim in range(d):
                dot += E_unit[gene_set_idx[i], dim] * E_unit[ranked_list_idx[j], dim]
            if dot > max_sim:
                max_sim = dot
        col_max[j] = max_sim

    # Mean-center
    mean_val = 0.0
    for j in range(L):
        mean_val += col_max[j]
    mean_val = mean_val / L

    for j in range(L):
        col_max[j] = col_max[j] - mean_val

    # Find max deviation using cumsum
    cumsum = 0.0
    max_pos = -1e9
    min_neg = 1e9

    for j in range(L):
        cumsum += col_max[j]
        if cumsum > max_pos:
            max_pos = cumsum
        if cumsum < min_neg:
            min_neg = cumsum

    if abs(max_pos) > abs(min_neg):
        return max_pos
    else:
        return min_neg


def compute_es_fast(E_unit, gene_set_idx, ranked_list_idx):
    """
    Compute GSEA enrichment score using BLAS.
    """
    X = E_unit[gene_set_idx]  # (m, d)
    Y = E_unit[ranked_list_idx]  # (L, d)

    # BLAS GEMM
    A = X @ Y.T  # (m, L)

    # Best match for each position in ranked list
    col_max = A.max(axis=0)  # (L,)

    # Mean-center and find max deviation
    col_max = col_max - col_max.mean()
    cumsum = np.cumsum(col_max)

    max_pos = cumsum.max()
    min_neg = cumsum.min()

    return float(max_pos if abs(max_pos) > abs(min_neg) else min_neg)


# Optional: avoid even the tuple packing by passing 5 arrays directly
def compute_bma_fast_ws_5(E_unit, X_idx, Y_idx, X, Y, A, row_max, col_max):
    np.take(E_unit, X_idx, axis=0, out=X)
    np.take(E_unit, Y_idx, axis=0, out=Y)

    np.matmul(X, Y.T, out=A)

    A.max(axis=1, out=row_max)
    A.max(axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


class NullCacheBMA:
    def __init__(self):
        self.cache = {}
        self.metadata = {}

    @staticmethod
    def build_metadata(E_unit, population_idx1, population_idx2, ite, seed):
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
        }

    def metadata_matches(self, expected):
        if not self.metadata:
            return False, "cache has no metadata"
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                return False, f"metadata mismatch for {key}"
        return True, ""

    @staticmethod
    def resolve_seed(seed):
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
    ):
        seed = self.resolve_seed(seed)
        mmap_path = f"E_unit_{os.getpid()}.npy"

        try:
            np.save(mmap_path, np.asarray(E_unit, dtype=np.float32))
            pop1 = np.asarray(population_idx, dtype=np.int32)
            pop2 = np.asarray(
                population_idx if population_idx2 is None else population_idx2,
                dtype=np.int32,
            )
            self.metadata = self.build_metadata(E_unit, pop1, pop2, ite, seed)

            if chunk_size is None or chunk_size <= 0:
                chunks = _chunked_by_cost(size_pairs, ite, n_workers)
            else:
                chunks = list(_chunked(size_pairs, chunk_size))
            n_chunks = len(chunks)

            if verbose:
                print(f"Precomputing BMA null for {len(size_pairs)} size pairs")
                print(f"Iterations: {ite}, seed: {seed}")
                chunk_desc = "auto-cost" if chunk_size is None or chunk_size <= 0 else chunk_size
                print(f"Workers: {n_workers}, chunk_size: {chunk_desc}")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(mmap_path, pop1, pop2, ite, seed),
            ) as ex:
                it = ex.map(_compute_chunk, chunks)
                it = tqdm(it, total=n_chunks, desc="BMA null",
                          disable=not show_progress if show_progress is not None else not verbose)

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
        seed = self.resolve_seed(seed)
        rng = np.random.default_rng(seed)
        pop1 = np.asarray(population_idx, dtype=np.int32)
        pop2 = np.asarray(
            population_idx if population_idx2 is None else population_idx2,
            dtype=np.int32,
        )
        self.metadata = self.build_metadata(E_unit, pop1, pop2, ite, seed)
        d = E_unit.shape[1]

        size_pairs_sorted = sorted(size_pairs)
        if not size_pairs_sorted:
            return

        if verbose:
            print(f"Precomputing BMA null for {len(size_pairs_sorted)} size pairs")
            print(f"Iterations: {ite}, seed: {seed}")

        max_m = max(m for m, k in size_pairs_sorted)
        max_k = max(k for m, k in size_pairs_sorted)
        ws_max = BMAWorkspaceMax(max_m, max_k, d)

        bma_nb = compute_bma_numba
        get_views = ws_max.views

        N1, N2 = len(pop1), len(pop2)
        perm1 = np.arange(N1, dtype=np.int32)
        perm2 = np.arange(N2, dtype=np.int32)

        for m, k in tqdm(size_pairs_sorted, desc="BMA null", disable=not verbose):
            if (m, k) in self.cache:
                continue

            mean = 0.0
            M2 = 0.0
            use_numba = m * k < 400

            X, Y, A, row_max, col_max = get_views(m, k)
            m_low = np.arange(m, dtype=np.int64)
            k_low = np.arange(k, dtype=np.int64)

            for i in range(ite):
                X_idx = pop1[_fys_sample_single(perm1, rng.integers(m_low, N1, dtype=np.int64))]
                Y_idx = pop2[_fys_sample_single(perm2, rng.integers(k_low, N2, dtype=np.int64))]

                if use_numba:
                    s = bma_nb(E_unit, X_idx, Y_idx)
                else:
                    s = compute_bma_fast_ws_5(
                        E_unit, X_idx, Y_idx, X, Y, A, row_max, col_max
                    )

                delta = s - mean
                mean += delta / (i + 1)
                M2 += delta * (s - mean)

            std = float(np.sqrt(M2 / (ite - 1))) if ite > 1 else 0.0
            self.cache[(m, k)] = (float(mean), std)

        if verbose:
            cache_mb = len(self.cache) * 2 * 8 / 1e6
            print(f"Cached {len(self.cache)} distributions (~{cache_mb:.1f} MB)")

    def get_zscore(self, true_score, m, k):
        mean_null, std_null = self.cache[(m, k)]
        return 0.0 if std_null == 0 else (true_score - mean_null) / std_null

    @staticmethod
    def suggest_path(base_dir, E_unit, population_idx1, population_idx2):
        """Return a content-addressed cache path under base_dir.

        Different embeddings or populations get distinct filenames, so
        parallel runs and multi-dataset workflows never share a stale cache.
        """
        emb_h  = _hash_array(np.asarray(E_unit, dtype=np.float32))[:8]
        pop1_h = _hash_array(np.asarray(population_idx1, dtype=np.int32))[:8]
        pop2_h = _hash_array(np.asarray(population_idx2, dtype=np.int32))[:8]
        return os.path.join(base_dir, f"bma_{emb_h}_{pop1_h}_{pop2_h}.pkl")

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({"metadata": self.metadata, "cache": self.cache}, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "cache" in payload:
            self.cache = payload["cache"]
            self.metadata = payload.get("metadata", {})
        else:
            self.cache = payload
            self.metadata = {}


class NullCacheES:
    """Null distribution cache for GSEA enrichment scores."""

    def __init__(self):
        self.cache = {}

    def precompute(
        self,
        E_unit,
        population_idx,
        gene_set_sizes,
        ranked_list_idx,
        ite=1000,
        verbose=True,
        seed=12345,
    ):
        """
        Precompute ES null distributions.

        IMPORTANT: This uses the OBSERVED ranked list, with random gene sets.
        Null hypothesis: "gene set membership is random, given this ranking."

        Parameters:
        -----------
        ranked_list_idx : np.ndarray
            The ACTUAL observed ranked list indices (not random!)
        """
        np.random.seed(seed)
        pop_array = np.array(population_idx, dtype=np.int32)
        L = len(ranked_list_idx)

        if verbose:
            print(f"Precomputing ES null for {len(gene_set_sizes)} gene set sizes")
            print(f"Ranked list length: {L}")
            print(f"Iterations: {ite}, seed: {seed}")
            print("Null: random gene sets vs OBSERVED ranking")

        for m in tqdm(sorted(gene_set_sizes), desc="ES null", disable=not verbose):
            key = (m, L)
            if key in self.cache:
                continue

            scores = np.empty(ite, dtype=np.float32)

            for i in range(ite):
                # Random gene set of size m
                gene_set_idx = np.random.choice(pop_array, m, replace=False)

                # Use OBSERVED ranked list (not random!)
                scores[i] = compute_es_fast(E_unit, gene_set_idx, ranked_list_idx)

            self.cache[key] = (float(scores.mean()), float(scores.std()))

        if verbose:
            print(f"Cached {len(self.cache)} ES distributions")

    def get_zscore(self, true_score, m, ranked_list_length):
        """Get z-score for ES."""
        key = (m, ranked_list_length)
        if key not in self.cache:
            raise KeyError(f"Size ({m}, {ranked_list_length}) not in cache")

        mean_null, std_null = self.cache[key]
        if std_null == 0:
            return 0.0

        return (true_score - mean_null) / std_null

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.cache, f)
        print(f"Saved ES cache to {filename}")

    def load(self, filename):
        with open(filename, "rb") as f:
            self.cache = pickle.load(f)
        print(f"Loaded {len(self.cache)} ES distributions from {filename}")


def warmup_numba():
    """Compile numba functions before main computation."""
    dummy_E = np.random.randn(100, 50).astype(np.float32)
    dummy_E = l2_normalize_rows(dummy_E)
    dummy_idx = np.arange(20, dtype=np.int64)

    _ = compute_bma_numba(dummy_E, dummy_idx, dummy_idx)
    _ = compute_es_numba(dummy_E, dummy_idx, dummy_idx)

    # Warm up FYS samplers
    perm = np.arange(30, dtype=np.int32)
    js_s = np.array([0, 2, 3, 5], dtype=np.int64)
    _ = _fys_sample_single(perm, js_s)
    js_b = np.array([[0, 2, 3, 5], [1, 1, 4, 5]], dtype=np.int64)
    out = np.empty((2, 4), dtype=np.int32)
    _batch_fys_sample(perm, js_b, out)

    print("Numba compilation complete.")


def get_background_indices(geneset, node_set, g_node2index):
    """Get sorted background gene indices."""
    all_genes = set(chain.from_iterable(geneset.values()))
    all_genes.intersection_update(node_set)
    return sorted(g_node2index[x] for x in all_genes)


def preconvert_indices_to_arrays(geneset_indices):
    """Convert gene set indices to numpy arrays once."""
    return {
        term: np.array(list(indices), dtype=np.int32)
        for term, indices in geneset_indices.items()
    }

# import os
# import hashlib
#
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#
# import numpy as np
# from numba import jit
# import pickle
# from tqdm import tqdm
# from itertools import chain
# from scipy.stats import hypergeom
#
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from itertools import islice
#
# _W_S = None   # precomputed pairwise similarity matrix: E[pop1] @ E[pop2].T
# _ITE = None
# _SEED = None
#
#
# def _chunked(seq, n):
#     seq = list(seq)
#     for i in range(0, len(seq), n):
#         yield seq[i : i + n]
#
#
# def _chunked_by_cost(size_pairs, ite, n_workers, target_chunks_per_worker=4):
#     pairs = sorted(size_pairs, key=lambda p: p[0] * p[1], reverse=True)
#     if not pairs:
#         return []
#
#     n_chunks = min(len(pairs), max(1, int(n_workers) * target_chunks_per_worker))
#     chunks = [[] for _ in range(n_chunks)]
#     costs = [0] * n_chunks
#
#     for pair in pairs:
#         idx = min(range(n_chunks), key=costs.__getitem__)
#         chunks[idx].append(pair)
#         costs[idx] += int(pair[0]) * int(pair[1]) * int(ite)
#
#     return [chunk for chunk in chunks if chunk]
#
#
# def _hash_array(arr):
#     arr = np.ascontiguousarray(arr)
#     h = hashlib.blake2b(digest_size=16)
#     h.update(str(arr.shape).encode("utf-8"))
#     h.update(str(arr.dtype).encode("utf-8"))
#     h.update(arr.view(np.uint8))
#     return h.hexdigest()
#
#
# _BATCH_BYTES = 64 * 1024 * 1024  # 64 MB per batched matmul
#
#
# def _init_worker(s_path, ite, seed):
#     for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
#                 "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
#                 "NUMEXPR_NUM_THREADS"):
#         os.environ[var] = "1"
#     try:
#         from threadpoolctl import threadpool_limits
#         threadpool_limits(1)
#     except Exception:
#         pass
#     global _W_S, _ITE, _SEED
#     # mmap-mode: all workers share one physical copy via kernel page cache.
#     _W_S = np.load(s_path, mmap_mode='r')
#     _ITE = int(ite)
#     _SEED = int(seed)
#
#
# def _bma_batch_precomp(S, m, k, ite, rng):
#     """Compute `ite` BMA null scores using precomputed S = E[pop1] @ E[pop2].T.
#
#     Each iteration gathers a random (m, k) submatrix from S and applies BMA
#     reductions.  No GEMM in the inner loop — eliminates the embedding dimension
#     d from per-iteration cost.  Memory is capped at _BATCH_BYTES so large (m,k)
#     pairs are processed in sub-batches.
#     """
#     N1, N2 = S.shape
#     batch = min(ite, max(1, _BATCH_BYTES // max(m * k * 4, 1)))
#
#     scores = np.empty(ite, dtype=np.float32)
#     done = 0
#     while done < ite:
#         b = min(batch, ite - done)
#
#         xi = np.stack([rng.choice(N1, size=m, replace=False, shuffle=False)
#                        for _ in range(b)])  # (b, m)
#         yi = np.stack([rng.choice(N2, size=k, replace=False, shuffle=False)
#                        for _ in range(b)])  # (b, k)
#
#         # Broadcasting fancy index → (b, m, k), pure gather, no FLOP
#         sub = S[xi[:, :, None], yi[:, None, :]]
#         row_sum = sub.max(axis=2).sum(axis=1)   # (b,)
#         col_sum = sub.max(axis=1).sum(axis=1)   # (b,)
#         scores[done:done + b] = (row_sum + col_sum) / (m + k)
#         done += b
#
#     return float(scores.mean()), float(scores.std(ddof=1)) if ite > 1 else 0.0
#
#
# def _compute_chunk(pairs_chunk):
#     if not pairs_chunk:
#         return []
#     out = []
#     # smallest first so peak memory stays lower early in the chunk
#     for m, k in sorted(pairs_chunk, key=lambda p: p[0] * p[1]):
#         rng = np.random.default_rng(np.random.SeedSequence([_SEED, m, k]))
#         mean, std = _bma_batch_precomp(_W_S, m, k, _ITE, rng)
#         out.append(((m, k), (mean, std)))
#     return out
#
#
# class BMAWorkspaceMax:
#     def __init__(self, max_m: int, max_k: int, d: int):
#         self.X = np.empty((max_m, d), dtype=np.float32)
#         self.Y = np.empty((max_k, d), dtype=np.float32)
#         self.A = np.empty((max_m, max_k), dtype=np.float32)
#         self.row_max = np.empty(max_m, dtype=np.float32)
#         self.col_max = np.empty(max_k, dtype=np.float32)
#
#     def views(self, m: int, k: int):
#         return (
#             self.X[:m],
#             self.Y[:k],
#             self.A[:m, :k],
#             self.row_max[:m],
#             self.col_max[:k],
#         )
#
#
# def compute_bma_fast_ws_view(E_unit, X_idx, Y_idx, views):
#     X, Y, A, row_max, col_max = views
#
#     np.take(E_unit, X_idx, axis=0, out=X)
#     np.take(E_unit, Y_idx, axis=0, out=Y)
#
#     np.matmul(X, Y.T, out=A)
#
#     A.max(axis=1, out=row_max)
#     A.max(axis=0, out=col_max)
#
#     return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))
#
#
# def compute_bma_blocks_ws(X, Y, views):
#     _, _, A, row_max, col_max = views
#     m = X.shape[0]
#     k = Y.shape[0]
#     A_view = A[:m, :k]
#     row_view = row_max[:m]
#     col_view = col_max[:k]
#
#     np.matmul(X, Y.T, out=A_view)
#     A_view.max(axis=1, out=row_view)
#     A_view.max(axis=0, out=col_view)
#
#     return float((row_view.sum() + col_view.sum()) / (m + k))
#
#
# def hypergeom_test(terms, g1_term2index, g2_term2index, g1_population, g2_population):
#     """
#     Given a pair of terms, two annotation dicts, and background populations,
#     calculate the hypergeometric test p-value.
#     """
#     term1, term2 = terms
#     indexes1 = set(g1_term2index[term1])
#     indexes2 = set(g2_term2index[term2])
#
#     inter = len(indexes1.intersection(indexes2))
#     n = len(indexes1)
#     m = len(indexes2)
#     N = len(set(g1_population).intersection(g2_population))
#
#     return hypergeom.sf(inter - 1, N, n, m)
#
#
# def t_score(x_w, x_b1, x_b2):
#     """
#     Unequal-variance t-test between within-set scores (x_w) and two backgrounds (x_b1, x_b2).
#     Implementation based on GIANT.
#     """
#     x_w = np.asarray(x_w)
#     x_b1 = np.asarray(x_b1)
#     x_b2 = np.asarray(x_b2)
#
#     s_w = x_w.std()
#     n_w = x_w.size
#
#     n_b = x_b1.size + x_b2.size
#     mean_b = (x_b1.sum() + x_b2.sum()) / n_b
#
#     v_b = ((x_b1 - mean_b) ** 2).sum() + ((x_b2 - mean_b) ** 2).sum()
#     v_b /= n_b
#
#     s_x = np.sqrt(s_w**2 / n_w + v_b / n_b)
#     mean_w = x_w.mean()
#     return (mean_w - mean_b) / s_x
#
#
# def mean_embedding(
#     terms, g1_embedding, g2_embedding, g1_term2index, g2_term2index, distinct=False
# ):
#     """
#     Given a pair of terms, two annotation dicts, and two embedding matrices,
#     calculate the cosine similarity between term centroids.
#     """
#     term1, term2 = terms
#
#     idx1 = list(g1_term2index[term1])
#     idx2 = list(g2_term2index[term2])
#
#     if distinct:
#         idx2 = list(set(idx2).difference(idx1))
#         if len(idx2) < 10:
#             return (0, 0)
#
#     v1 = g1_embedding[idx1].mean(axis=0)
#     v2 = g2_embedding[idx2].mean(axis=0)
#
#     denom = np.linalg.norm(v1) * np.linalg.norm(v2)
#     if denom == 0:
#         return 0.0
#
#     score = float(np.dot(v1, v2) / denom)
#     return score
#
#
# def mean_matrix(terms, matrix, g1_term2index, g2_term2index, distinct=False):
#     """
#     Given a pair of terms, two annotation dicts, and a pairwise similarity matrix,
#     calculate mean score between two terms.
#     """
#     term1, term2 = terms
#     idx1 = list(g1_term2index[term1])
#     idx2 = list(g2_term2index[term2])
#
#     if distinct:
#         idx2 = list(set(idx2).difference(idx1))
#         if len(idx2) < 10:
#             return (0, 0)
#
#     mat = np.asarray(matrix)
#     score = mat[np.ix_(idx1, idx2)].mean()
#     return score
#
#
# def l2_normalize_rows(E, eps=1e-12):
#     """L2-normalize embeddings so cosine similarity = dot product."""
#     E = E.astype(np.float32, copy=False)
#     norms = np.linalg.norm(E, axis=1, keepdims=True)
#     return E / np.maximum(norms, eps)
#
#
# def precompute_term_embedding_blocks(E_unit, term2indices):
#     return {
#         term: np.ascontiguousarray(E_unit[idx], dtype=np.float32)
#         for term, idx in term2indices.items()
#     }
#
#
# def concatenate_term_embedding_blocks(terms, blocks):
#     lengths = np.asarray([blocks[t].shape[0] for t in terms], dtype=np.int32)
#     offsets = np.empty(len(terms) + 1, dtype=np.int64)
#     offsets[0] = 0
#     np.cumsum(lengths, out=offsets[1:])
#     if len(terms) == 0:
#         return np.empty((0, 0), dtype=np.float32), offsets, lengths
#
#     d = next(iter(blocks.values())).shape[1]
#     concat = np.empty((int(offsets[-1]), d), dtype=np.float32)
#     for term, start, end in zip(terms, offsets[:-1], offsets[1:]):
#         concat[start:end] = blocks[term]
#     return concat, offsets, lengths
#
#
# def same_index_arrays_by_term(terms1, term2indices1, terms2, term2indices2):
#     if len(terms1) != len(terms2):
#         return False
#     for t1, t2 in zip(terms1, terms2):
#         if t1 != t2:
#             return False
#         if not np.array_equal(term2indices1[t1], term2indices2[t2]):
#             return False
#     return True
#
#
# def _zscore_from_cache(cache, true_score, m, k):
#     mean_null, std_null = cache[(m, k)]
#     return 0.0 if std_null == 0 else (true_score - mean_null) / std_null
#
#
# def score_bma_zscore_matrix(
#     E_unit,
#     terms1,
#     terms2,
#     indices1,
#     indices2,
#     null_cache,
#     blocks1=None,
#     blocks2=None,
#     symmetric=False,
#     n_workers=1,
#     numba_threshold=400,
#     show_progress=False,
# ):
#     zscores = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
#     max_m = max(len(indices1[t]) for t in terms1)
#     max_k = max(len(indices2[t]) for t in terms2)
#     cache = null_cache.cache
#     use_blocks = blocks1 is not None and blocks2 is not None
#
#     def score_row(i):
#         t1 = terms1[i]
#         X_idx = indices1[t1]
#         m = len(X_idx)
#         start_term = i if symmetric else 0
#         row = np.zeros(len(terms2) - start_term, dtype=np.float32)
#         mirrored = []
#         ws = BMAWorkspaceMax(max_m, max_k, E_unit.shape[1])
#         term_iter = enumerate(terms2[start_term:], start=start_term)
#
#         for j, t2 in term_iter:
#             Y_idx = indices2[t2]
#             k = len(Y_idx)
#
#             if m * k < numba_threshold:
#                 true_score = compute_bma_numba(E_unit, X_idx, Y_idx)
#             elif use_blocks:
#                 true_score = compute_bma_blocks_ws(
#                     blocks1[t1], blocks2[t2], ws.views(m, k)
#                 )
#             else:
#                 true_score = compute_bma_fast_ws_view(
#                     E_unit, X_idx, Y_idx, ws.views(m, k)
#                 )
#
#             row[j - start_term] = _zscore_from_cache(cache, true_score, m, k)
#             if symmetric and i != j:
#                 mirrored.append((j, _zscore_from_cache(cache, true_score, k, m)))
#
#         return i, start_term, row, mirrored
#
#     if n_workers and n_workers > 1:
#         with ThreadPoolExecutor(max_workers=n_workers) as ex:
#             rows = ex.map(score_row, range(len(terms1)))
#             rows = tqdm(
#                 rows, total=len(terms1), desc="Computing", disable=not show_progress
#             )
#             for i, start_term, row, mirrored in rows:
#                 zscores[i, start_term : start_term + len(row)] = row
#                 for j, value in mirrored:
#                     zscores[j, i] = value
#     else:
#         rows = tqdm(range(len(terms1)), desc="Computing", disable=not show_progress)
#         for i in rows:
#             row_i, start_term, row, mirrored = score_row(i)
#             zscores[row_i, start_term : start_term + len(row)] = row
#             for j, value in mirrored:
#                 zscores[j, row_i] = value
#
#     return zscores
#
#
# def score_bma_zscore_matrix_batched(
#     terms1,
#     terms2,
#     null_cache,
#     blocks1,
#     blocks2,
#     symmetric=False,
#     n_workers=1,
#     max_workspace_mb=1024,
#     show_progress=False,
# ):
#     zscores = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
#     concat2, offsets2, lengths2 = concatenate_term_embedding_blocks(terms2, blocks2)
#     cache = null_cache.cache
#
#     max_m = max(blocks1[t].shape[0] for t in terms1)
#     max_cols = int(offsets2[-1])
#     max_terms = len(terms2)
#     workspace_mb = (
#         max_m * max_cols * 4
#         + max_m * max_terms * 4
#         + max_cols * 4
#     ) / 1e6
#     if max_workspace_mb and max_workspace_mb > 0:
#         effective_workers = max(1, min(int(n_workers), int(max_workspace_mb // max(workspace_mb, 1e-9))))
#     else:
#         effective_workers = max(1, int(n_workers))
#
#     def score_row(i):
#         X = blocks1[terms1[i]]
#         m = X.shape[0]
#         start_term = i if symmetric else 0
#         col_start = int(offsets2[start_term])
#         Y = concat2[col_start:]
#         local_offsets = offsets2[start_term:-1] - col_start
#         lengths = lengths2[start_term:]
#
#         A = X @ Y.T
#         row_term_max = np.maximum.reduceat(A, local_offsets, axis=1)
#         row_sums = row_term_max.sum(axis=0)
#         col_max = A.max(axis=0)
#         col_sums = np.add.reduceat(col_max, local_offsets)
#         scores = (row_sums + col_sums) / (m + lengths)
#
#         zvals = np.empty(scores.shape[0], dtype=np.float32)
#         mirrored = []
#         for p, score in enumerate(scores):
#             j = start_term + p
#             k = int(lengths[p])
#             zvals[p] = _zscore_from_cache(cache, float(score), m, k)
#             if symmetric and i != j:
#                 mirrored.append((j, _zscore_from_cache(cache, float(score), k, m)))
#         return i, start_term, zvals, mirrored
#
#     if effective_workers > 1:
#         with ThreadPoolExecutor(max_workers=effective_workers) as ex:
#             rows = ex.map(score_row, range(len(terms1)))
#             rows = tqdm(
#                 rows, total=len(terms1), desc="Computing", disable=not show_progress
#             )
#             for i, start_term, zvals, mirrored in rows:
#                 zscores[i, start_term : start_term + len(zvals)] = zvals
#                 for j, value in mirrored:
#                     zscores[j, i] = value
#     else:
#         rows = tqdm(range(len(terms1)), desc="Computing", disable=not show_progress)
#         for i in rows:
#             row_i, start_term, zvals, mirrored = score_row(i)
#             zscores[row_i, start_term : start_term + len(zvals)] = zvals
#             for j, value in mirrored:
#                 zscores[j, row_i] = value
#
#     return zscores, effective_workers, workspace_mb
#
#
# def _bma_workspace(m: int, k: int, d: int):
#     X = np.empty((m, d), dtype=np.float32)
#     Y = np.empty((k, d), dtype=np.float32)
#     A = np.empty((m, k), dtype=np.float32)
#     row_max = np.empty(m, dtype=np.float32)
#     col_max = np.empty(k, dtype=np.float32)
#     return X, Y, A, row_max, col_max
#
#
# def compute_bma_fast_ws(E_unit, X_idx, Y_idx, ws):
#     X, Y, A, row_max, col_max = ws
#     # Fill X,Y without allocating new arrays
#     np.take(E_unit, X_idx, axis=0, out=X)
#     np.take(E_unit, Y_idx, axis=0, out=Y)
#
#     # GEMM into a preallocated matrix
#     np.matmul(X, Y.T, out=A)
#
#     # reductions into preallocated vectors
#     A.max(axis=1, out=row_max)
#     A.max(axis=0, out=col_max)
#
#     return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))
#
#
# @jit(nopython=True, nogil=True)
# def compute_bma_numba(E_unit, X_idx, Y_idx):
#     """
#     Numba version for very small sets where BLAS overhead dominates.
#     Only faster for m*k < 400 (e.g., both sets < 20 genes).
#     """
#     m, k = len(X_idx), len(Y_idx)
#     d = E_unit.shape[1]
#
#     # Allocate similarity matrix once
#     A = np.empty((m, k), dtype=np.float32)
#
#     # Compute all similarities (do it once, not twice!)
#     for i in range(m):
#         for j in range(k):
#             dot = 0.0
#             for dim in range(d):
#                 dot += E_unit[X_idx[i], dim] * E_unit[Y_idx[j], dim]
#             A[i, j] = dot
#
#     # Compute row maxes manually (Numba doesn't support axis parameter)
#     row_max_sum = 0.0
#     for i in range(m):
#         max_val = A[i, 0]
#         for j in range(1, k):
#             if A[i, j] > max_val:
#                 max_val = A[i, j]
#         row_max_sum += max_val
#
#     # Compute column maxes manually
#     col_max_sum = 0.0
#     for j in range(k):
#         max_val = A[0, j]
#         for i in range(1, m):
#             if A[i, j] > max_val:
#                 max_val = A[i, j]
#         col_max_sum += max_val
#
#     return (row_max_sum + col_max_sum) / (m + k)
#
#
# @jit(nopython=True)
# def compute_es_numba(E_unit, gene_set_idx, ranked_list_idx):
#     """Numba version for small gene sets."""
#     m = len(gene_set_idx)
#     L = len(ranked_list_idx)
#     d = E_unit.shape[1]
#
#     # Compute max similarity for each position
#     col_max = np.empty(L, dtype=np.float32)
#     for j in range(L):
#         max_sim = -1.0
#         for i in range(m):
#             dot = 0.0
#             for dim in range(d):
#                 dot += E_unit[gene_set_idx[i], dim] * E_unit[ranked_list_idx[j], dim]
#             if dot > max_sim:
#                 max_sim = dot
#         col_max[j] = max_sim
#
#     # Mean-center
#     mean_val = 0.0
#     for j in range(L):
#         mean_val += col_max[j]
#     mean_val = mean_val / L
#
#     for j in range(L):
#         col_max[j] = col_max[j] - mean_val
#
#     # Find max deviation using cumsum
#     cumsum = 0.0
#     max_pos = -1e9
#     min_neg = 1e9
#
#     for j in range(L):
#         cumsum += col_max[j]
#         if cumsum > max_pos:
#             max_pos = cumsum
#         if cumsum < min_neg:
#             min_neg = cumsum
#
#     if abs(max_pos) > abs(min_neg):
#         return max_pos
#     else:
#         return min_neg
#
#
# def compute_es_fast(E_unit, gene_set_idx, ranked_list_idx):
#     """
#     Compute GSEA enrichment score using BLAS.
#     """
#     X = E_unit[gene_set_idx]  # (m, d)
#     Y = E_unit[ranked_list_idx]  # (L, d)
#
#     # BLAS GEMM
#     A = X @ Y.T  # (m, L)
#
#     # Best match for each position in ranked list
#     col_max = A.max(axis=0)  # (L,)
#
#     # Mean-center and find max deviation
#     col_max = col_max - col_max.mean()
#     cumsum = np.cumsum(col_max)
#
#     max_pos = cumsum.max()
#     min_neg = cumsum.min()
#
#     return float(max_pos if abs(max_pos) > abs(min_neg) else min_neg)
#
#
# # Optional: avoid even the tuple packing by passing 5 arrays directly
# def compute_bma_fast_ws_5(E_unit, X_idx, Y_idx, X, Y, A, row_max, col_max):
#     np.take(E_unit, X_idx, axis=0, out=X)
#     np.take(E_unit, Y_idx, axis=0, out=Y)
#
#     np.matmul(X, Y.T, out=A)
#
#     A.max(axis=1, out=row_max)
#     A.max(axis=0, out=col_max)
#
#     return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))
#
#
# class NullCacheBMA:
#     def __init__(self):
#         self.cache = {}
#         self.metadata = {}
#
#     @staticmethod
#     def build_metadata(E_unit, population_idx1, population_idx2, ite, seed):
#         return {
#             "kind": "andes_bma_null",
#             "version": 2,
#             "embedding_hash": _hash_array(np.asarray(E_unit, dtype=np.float32)),
#             "population1_hash": _hash_array(
#                 np.asarray(population_idx1, dtype=np.int32)
#             ),
#             "population2_hash": _hash_array(
#                 np.asarray(population_idx2, dtype=np.int32)
#             ),
#             "ite": int(ite),
#             "seed": int(seed),
#         }
#
#     def metadata_matches(self, expected):
#         if not self.metadata:
#             return False, "cache has no metadata"
#         for key, value in expected.items():
#             if self.metadata.get(key) != value:
#                 return False, f"metadata mismatch for {key}"
#         return True, ""
#
#     @staticmethod
#     def resolve_seed(seed):
#         if seed is None or int(seed) < 0:
#             return int(np.random.SeedSequence().entropy)
#         return int(seed)
#
#     def precompute_parallel(
#         self,
#         E_unit,
#         population_idx,
#         size_pairs,
#         ite=1000,
#         verbose=True,
#         seed=12345,
#         n_workers=8,
#         chunk_size=None,
#         population_idx2=None,
#         show_progress=None,
#     ):
#         seed = self.resolve_seed(seed)
#         s_path = f"S_null_{os.getpid()}.npy"
#
#         try:
#             E_f = np.asarray(E_unit, dtype=np.float32)
#             pop1 = np.asarray(population_idx, dtype=np.int32)
#             pop2 = np.asarray(
#                 population_idx if population_idx2 is None else population_idx2,
#                 dtype=np.int32,
#             )
#             self.metadata = self.build_metadata(E_f, pop1, pop2, ite, seed)
#
#             # Precompute full pairwise similarity matrix once; workers share via mmap.
#             S = (E_f[pop1] @ E_f[pop2].T).astype(np.float32)
#             np.save(s_path, S)
#             del S
#
#             if chunk_size is None or chunk_size <= 0:
#                 chunks = _chunked_by_cost(size_pairs, ite, n_workers)
#             else:
#                 chunks = list(_chunked(size_pairs, chunk_size))
#             n_chunks = len(chunks)
#
#             if verbose:
#                 print(f"Precomputing BMA null for {len(size_pairs)} size pairs")
#                 print(f"Iterations: {ite}, seed: {seed}")
#                 chunk_desc = "auto-cost" if chunk_size is None or chunk_size <= 0 else chunk_size
#                 print(f"Workers: {n_workers}, chunk_size: {chunk_desc}")
#
#             with ProcessPoolExecutor(
#                 max_workers=n_workers,
#                 initializer=_init_worker,
#                 initargs=(s_path, ite, seed),
#             ) as ex:
#                 it = ex.map(_compute_chunk, chunks)
#                 _show = show_progress if show_progress is not None else verbose
#                 it = tqdm(it, total=n_chunks, desc="BMA null", disable=not _show)
#
#                 for chunk_res in it:
#                     for key, val in chunk_res:
#                         self.cache[key] = val
#         finally:
#             try:
#                 os.remove(s_path)
#             except OSError:
#                 pass
#
#     def precompute(
#         self,
#         E_unit,
#         population_idx,
#         size_pairs,
#         ite=1000,
#         verbose=True,
#         seed=12345,
#         population_idx2=None,
#     ):
#         seed = self.resolve_seed(seed)
#         rng = np.random.default_rng(seed)
#         E_f = np.asarray(E_unit, dtype=np.float32)
#         pop1 = np.asarray(population_idx, dtype=np.int32)
#         pop2 = np.asarray(
#             population_idx if population_idx2 is None else population_idx2,
#             dtype=np.int32,
#         )
#         self.metadata = self.build_metadata(E_f, pop1, pop2, ite, seed)
#
#         size_pairs_sorted = sorted(size_pairs)
#         if not size_pairs_sorted:
#             return
#
#         if verbose:
#             print(f"Precomputing BMA null for {len(size_pairs_sorted)} size pairs")
#             print(f"Iterations: {ite}, seed: {seed}")
#
#         # Precompute full pairwise similarity matrix once; each iteration is a gather.
#         S = (E_f[pop1] @ E_f[pop2].T).astype(np.float32)
#         N1, N2 = S.shape
#         choice = rng.choice
#
#         for m, k in tqdm(size_pairs_sorted, desc="BMA null", disable=not verbose):
#             if (m, k) in self.cache:
#                 continue
#
#             mean = 0.0
#             M2 = 0.0
#
#             for i in range(ite):
#                 xi = choice(N1, size=m, replace=False, shuffle=False)
#                 yi = choice(N2, size=k, replace=False, shuffle=False)
#                 sub = S[np.ix_(xi, yi)]
#                 s = float((sub.max(axis=1).sum() + sub.max(axis=0).sum()) / (m + k))
#
#                 delta = s - mean
#                 mean += delta / (i + 1)
#                 M2 += delta * (s - mean)
#
#             std = float(np.sqrt(M2 / (ite - 1))) if ite > 1 else 0.0
#             self.cache[(m, k)] = (float(mean), std)
#
#         if verbose:
#             cache_mb = len(self.cache) * 2 * 8 / 1e6
#             print(f"Cached {len(self.cache)} distributions (~{cache_mb:.1f} MB)")
#
#     def get_zscore(self, true_score, m, k):
#         mean_null, std_null = self.cache[(m, k)]
#         return 0.0 if std_null == 0 else (true_score - mean_null) / std_null
#
#     @staticmethod
#     def suggest_path(base_dir, E_unit, population_idx1, population_idx2):
#         """Return a content-addressed cache path under base_dir.
#
#         Different embeddings or populations get distinct filenames, so
#         parallel runs and multi-dataset workflows never share a stale cache.
#         """
#         emb_h  = _hash_array(np.asarray(E_unit, dtype=np.float32))[:8]
#         pop1_h = _hash_array(np.asarray(population_idx1, dtype=np.int32))[:8]
#         pop2_h = _hash_array(np.asarray(population_idx2, dtype=np.int32))[:8]
#         return os.path.join(base_dir, f"bma_{emb_h}_{pop1_h}_{pop2_h}.pkl")
#
#     def save(self, filename):
#         with open(filename, "wb") as f:
#             pickle.dump({"metadata": self.metadata, "cache": self.cache}, f)
#
#     def load(self, filename):
#         with open(filename, "rb") as f:
#             payload = pickle.load(f)
#         if isinstance(payload, dict) and "cache" in payload:
#             self.cache = payload["cache"]
#             self.metadata = payload.get("metadata", {})
#         else:
#             self.cache = payload
#             self.metadata = {}
#
#
# class NullCacheES:
#     """Null distribution cache for GSEA enrichment scores."""
#
#     def __init__(self):
#         self.cache = {}
#
#     def precompute(
#         self,
#         E_unit,
#         population_idx,
#         gene_set_sizes,
#         ranked_list_idx,
#         ite=1000,
#         verbose=True,
#         seed=12345,
#     ):
#         """
#         Precompute ES null distributions.
#
#         IMPORTANT: This uses the OBSERVED ranked list, with random gene sets.
#         Null hypothesis: "gene set membership is random, given this ranking."
#
#         Parameters:
#         -----------
#         ranked_list_idx : np.ndarray
#             The ACTUAL observed ranked list indices (not random!)
#         """
#         np.random.seed(seed)
#         pop_array = np.array(population_idx, dtype=np.int32)
#         L = len(ranked_list_idx)
#
#         if verbose:
#             print(f"Precomputing ES null for {len(gene_set_sizes)} gene set sizes")
#             print(f"Ranked list length: {L}")
#             print(f"Iterations: {ite}, seed: {seed}")
#             print("Null: random gene sets vs OBSERVED ranking")
#
#         for m in tqdm(sorted(gene_set_sizes), desc="ES null", disable=not verbose):
#             key = (m, L)
#             if key in self.cache:
#                 continue
#
#             scores = np.empty(ite, dtype=np.float32)
#
#             for i in range(ite):
#                 # Random gene set of size m
#                 gene_set_idx = np.random.choice(pop_array, m, replace=False)
#
#                 # Use OBSERVED ranked list (not random!)
#                 scores[i] = compute_es_fast(E_unit, gene_set_idx, ranked_list_idx)
#
#             self.cache[key] = (float(scores.mean()), float(scores.std()))
#
#         if verbose:
#             print(f"Cached {len(self.cache)} ES distributions")
#
#     def get_zscore(self, true_score, m, ranked_list_length):
#         """Get z-score for ES."""
#         key = (m, ranked_list_length)
#         if key not in self.cache:
#             raise KeyError(f"Size ({m}, {ranked_list_length}) not in cache")
#
#         mean_null, std_null = self.cache[key]
#         if std_null == 0:
#             return 0.0
#
#         return (true_score - mean_null) / std_null
#
#     def save(self, filename):
#         with open(filename, "wb") as f:
#             pickle.dump(self.cache, f)
#         print(f"Saved ES cache to {filename}")
#
#     def load(self, filename):
#         with open(filename, "rb") as f:
#             self.cache = pickle.load(f)
#         print(f"Loaded {len(self.cache)} ES distributions from {filename}")
#
#
# def warmup_numba():
#     """Compile numba functions before main computation."""
#     dummy_E = np.random.randn(100, 50).astype(np.float32)
#     dummy_E = l2_normalize_rows(dummy_E)
#     dummy_idx = np.arange(20, dtype=np.int64)
#
#     # Warmup
#     _ = compute_bma_numba(dummy_E, dummy_idx, dummy_idx)
#     _ = compute_es_numba(dummy_E, dummy_idx, dummy_idx)
#
#     print("Numba compilation complete.")
#
#
# def get_background_indices(geneset, node_set, g_node2index):
#     """Get sorted background gene indices."""
#     all_genes = set(chain.from_iterable(geneset.values()))
#     all_genes.intersection_update(node_set)
#     return sorted(g_node2index[x] for x in all_genes)
#
#
# def preconvert_indices_to_arrays(geneset_indices):
#     """Convert gene set indices to numpy arrays once."""
#     return {
#         term: np.array(list(indices), dtype=np.int32)
#         for term, indices in geneset_indices.items()
#     }
