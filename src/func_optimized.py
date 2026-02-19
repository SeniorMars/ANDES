import os

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
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

_E_UNIT = None
_POP = None
_ITE = None
_SEED = None


def _chunked(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _init_worker(e_unit_path, pop_array, ite, seed):
    global _E_UNIT, _POP, _ITE, _SEED
    _E_UNIT = np.load(e_unit_path, mmap_mode="r")
    _POP = pop_array
    _ITE = int(ite)
    _SEED = int(seed)


def _compute_chunk(pairs_chunk):
    d = _E_UNIT.shape[1]
    max_m = max(m for m, k in pairs_chunk)
    max_k = max(k for m, k in pairs_chunk)

    ws = BMAWorkspaceMax(max_m, max_k, d)
    out = []
    if not pairs_chunk:
        return out

    # smallest first reduces peak working-set early, but optional
    for m, k in sorted(pairs_chunk, key=lambda p: p[0] * p[1]):
        X, Y, A, row_max, col_max = ws.views(m, k)
        use_numba = m * k < 400

        # deterministic per-(m,k)
        rng = np.random.default_rng(np.random.SeedSequence([_SEED, m, k]))

        mean = 0.0
        M2 = 0.0

        for i in range(_ITE):
            X_idx = rng.choice(_POP, size=m, replace=False, shuffle=False)
            Y_idx = rng.choice(_POP, size=k, replace=False, shuffle=False)

            if use_numba:
                s = compute_bma_numba(_E_UNIT, X_idx, Y_idx)
            else:
                s = compute_bma_fast_ws_5(
                    _E_UNIT, X_idx, Y_idx, X, Y, A, row_max, col_max
                )

            delta = s - mean
            mean += delta / (i + 1)
            M2 += delta * (s - mean)

        n = _ITE
        std = float(np.sqrt(M2 / (n - 1))) if n > 1 else 0.0
        out.append(((m, k), (float(mean), float(std))))
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

    np.max(A, axis=1, out=row_max)
    np.max(A, axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


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
    np.max(A, axis=1, out=row_max)
    np.max(A, axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


@jit(nopython=True)
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

    np.max(A, axis=1, out=row_max)
    np.max(A, axis=0, out=col_max)

    return float((row_max.sum() + col_max.sum()) / (len(X_idx) + len(Y_idx)))


class NullCacheBMA:
    def __init__(self):
        self.cache = {}

    def precompute_parallel(
        self,
        E_unit,
        population_idx,
        size_pairs,
        ite=1000,
        verbose=True,
        seed=12345,
        n_workers=8,
        chunk_size=64,
    ):
        mmap_path = f"E_unit_{os.getpid()}.npy"

        try:
            np.save(mmap_path, np.asarray(E_unit, dtype=np.float32))

            n_chunks = (len(size_pairs) + chunk_size - 1) // chunk_size
            chunks = _chunked(size_pairs, chunk_size)

            if verbose:
                print(f"Precomputing BMA null for {len(size_pairs)} size pairs")
                print(f"Iterations: {ite}, seed: {seed}")
                print(f"Workers: {n_workers}, chunk_size: {chunk_size}")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(mmap_path, population_idx, ite, seed),
            ) as ex:
                it = ex.map(_compute_chunk, chunks)
                it = tqdm(it, total=n_chunks, desc="BMA null", disable=not verbose)

                for chunk_res in it:
                    for key, val in chunk_res:
                        self.cache[key] = val
        finally:
            try:
                os.remove(mmap_path)
            except OSError:
                pass

    def precompute(
        self, E_unit, population_idx, size_pairs, ite=1000, verbose=True, seed=12345
    ):
        rng = np.random.default_rng(seed)
        pop = np.asarray(population_idx, dtype=np.int32)
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
        choice = rng.choice

        get_views = ws_max.views

        for m, k in tqdm(size_pairs_sorted, desc="BMA null", disable=not verbose):
            if (m, k) in self.cache:
                continue

            mean = 0.0
            M2 = 0.0
            use_numba = m * k < 400

            X, Y, A, row_max, col_max = get_views(m, k)

            for i in range(ite):
                X_idx = choice(pop, size=m, replace=False, shuffle=False)
                Y_idx = choice(pop, size=k, replace=False, shuffle=False)

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

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.cache, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.cache = pickle.load(f)


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

    # Warmup
    _ = compute_bma_numba(dummy_E, dummy_idx, dummy_idx)
    _ = compute_es_numba(dummy_E, dummy_idx, dummy_idx)

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
