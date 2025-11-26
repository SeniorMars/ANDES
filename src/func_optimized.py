import numpy as np
import random
from scipy.stats import hypergeom
from numba import njit, prange

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


def best_match_average(matrix):
    """
    Given pairwise similarity between genes in two sets,
    calculate row-wise and column-wise maxima and return
    the weighted average of those maxima.
    """
    mat = np.asarray(matrix)
    # axis=0: column-wise, axis=1: row-wise
    max_cols = mat.max(axis=0)
    max_rows = mat.max(axis=1)
    rows, cols = mat.shape
    return (max_cols.sum() + max_rows.sum()) / (rows + cols)


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

    s_x = np.sqrt(s_w ** 2 / n_w + v_b / n_b)
    mean_w = x_w.mean()
    return (mean_w - mean_b) / s_x


def mean_embedding(terms, g1_embedding, g2_embedding, g1_term2index, g2_term2index, distinct=False):
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


def andes(terms, matrix, g1_term2index, g2_term2index,
          g1_population, g2_population, ite=1000, distinct=False):
    """
    Vectorized ANDES implementation.

    Parameters
    ----------
    terms : tuple
        (term1, term2) to be matched.
    matrix : np.ndarray
        Full pairwise similarity matrix.
    g1_term2index : dict
        term -> annotated indices for group 1.
    g2_term2index : dict
        term -> annotated indices for group 2.
    g1_population : list[int]
        Population indices to sample for group 1.
    g2_population : list[int]
        Population indices to sample for group 2.
    ite : int
        Number of random samples for background.
    distinct : bool
        If True, remove overlapping annotated indices from term2 and assign them to term1.
    """

    term1, term2 = terms

    idx1 = np.fromiter(g1_term2index[term1], dtype=int)
    idx2 = np.fromiter(g2_term2index[term2], dtype=int)

    if distinct:
        # keep only indices that are not in idx1
        idx2_set = set(idx2)
        idx2 = np.array(sorted(idx2_set.difference(idx1)), dtype=int)
        if idx2.size < 10:
            return (0.0, 0.0)

    mat = np.asarray(matrix)

    # True score
    true_sub = mat[np.ix_(idx1, idx2)]
    true_score = best_match_average(true_sub)

    # Random background samples (still sample w/o replacement per sample,
    # but vectorize the scoring over all samples).
    len1 = idx1.size
    len2 = idx2.size

    # Python loop just for sampling; numeric work is vectorized.
    rand_idx1_list = [random.sample(g1_population, len1) for _ in range(ite)]
    rand_idx2_list = [random.sample(g2_population, len2) for _ in range(ite)]

    rand_idx1 = np.asarray(rand_idx1_list, dtype=int)  # (ite, len1)
    rand_idx2 = np.asarray(rand_idx2_list, dtype=int)  # (ite, len2)

    # Build all background submatrices at once:
    # shape: (ite, len1, len2)
    back_mats = mat[rand_idx1[:, :, None], rand_idx2[:, None, :]]

    # Vectorized best_match_average over the 0th axis
    # max over rows (axis=1) and columns (axis=2)
    max_cols = back_mats.max(axis=1)   # (ite, len2)
    max_rows = back_mats.max(axis=2)   # (ite, len1)

    back_scores = (max_cols.sum(axis=1) + max_rows.sum(axis=1)) / (len1 + len2)  # (ite,)

    mean_back = back_scores.mean()
    std_back = back_scores.std()

    # guard against zero-variance background
    if std_back == 0:
        z_score = 0.0
    else:
        z_score = (true_score - mean_back) / std_back

    return float(true_score), float(z_score)


# global-ish cache: (len1, len2) -> (rand_idx1, rand_idx2)
_ANDES_INDEX_CACHE = {}

def andes_cached(terms, matrix, g1_term2index, g2_term2index,
                 g1_population, g2_population, ite=1000, distinct=False,
                 seed=12345):
    """
    ANDES with cached random indices per (len1, len2) to avoid
    repeated random.sample calls in the hot path.
    """
    term1, term2 = terms

    idx1 = np.fromiter(g1_term2index[term1], dtype=int)
    idx2 = np.fromiter(g2_term2index[term2], dtype=int)

    if distinct:
        idx2_set = set(idx2)
        idx2 = np.array(sorted(idx2_set.difference(idx1)), dtype=int)
        if idx2.size < 10:
            return (0.0, 0.0)

    mat = np.asarray(matrix)
    true_sub = mat[np.ix_(idx1, idx2)]
    true_score = best_match_average(true_sub)

    len1 = idx1.size
    len2 = idx2.size
    key = (len1, len2, ite)

    # build cache entry if needed
    if key not in _ANDES_INDEX_CACHE:
        rng = random.Random(seed + len1 * 10007 + len2)  # deterministic but different per size

        # slowest part
        rand_idx1_list = [rng.sample(g1_population, len1) for _ in range(ite)]
        rand_idx2_list = [rng.sample(g2_population, len2) for _ in range(ite)]

        _ANDES_INDEX_CACHE[key] = (
            np.asarray(rand_idx1_list, dtype=int),
            np.asarray(rand_idx2_list, dtype=int),
        )

    rand_idx1, rand_idx2 = _ANDES_INDEX_CACHE[key]  # shapes: (ite, len1), (ite, len2)

    # vectorized scoring over cached indices
    back_mats = mat[rand_idx1[:, :, None], rand_idx2[:, None, :]]  # (ite, len1, len2)

    max_cols = back_mats.max(axis=1)   # (ite, len2)
    max_rows = back_mats.max(axis=2)   # (ite, len1)
    back_scores = (max_cols.sum(axis=1) + max_rows.sum(axis=1)) / (len1 + len2)

    mean_back = back_scores.mean()
    std_back = back_scores.std()
    if std_back == 0:
        z_score = 0.0
    else:
        z_score = (true_score - mean_back) / std_back

    return float(true_score), float(z_score)


_ANDES_INDEX_CACHE_NP = {}
def andes_cached_np(
    terms,
    matrix,
    g1_term2index,
    g2_term2index,
    g1_population,
    g2_population,
    ite=1000,
    distinct=False,
    rng_seed=12345,
):
    """
    ANDES with cached random indices per (len1, len2, ite), using
    numpy.random.Generator.choice instead of random.sample
    to build the cache (potentially faster first call).

    Sampling is still WITHOUT replacement within each random gene set.
    """

    term1, term2 = terms

    # term indices
    idx1 = np.fromiter(g1_term2index[term1], dtype=int)
    idx2 = np.fromiter(g2_term2index[term2], dtype=int)

    if distinct:
        idx2_set = set(idx2)
        idx2 = np.array(sorted(idx2_set.difference(idx1)), dtype=int)
        if idx2.size < 10:
            return (0.0, 0.0)

    mat = np.asarray(matrix)

    # true score
    true_sub = mat[np.ix_(idx1, idx2)]
    true_score = best_match_average(true_sub)

    len1 = idx1.size
    len2 = idx2.size
    key = (len1, len2, ite, rng_seed)

    # build cache entry if needed
    if key not in _ANDES_INDEX_CACHE_NP:
        rng = np.random.default_rng(rng_seed + len1 * 10007 + len2)

        # We still want WITHOUT replacement within each random gene set.
        # NumPy doesn't give us "2D without replacement per row" in one call,
        # so we loop over ite but use np RNG (fast) instead of Python's random.sample.
        rand_idx1 = np.stack(
            [rng.choice(g1_population, size=len1, replace=False) for _ in range(ite)],
            axis=0,
        ).astype(int)

        rand_idx2 = np.stack(
            [rng.choice(g2_population, size=len2, replace=False) for _ in range(ite)],
            axis=0,
        ).astype(int)

        _ANDES_INDEX_CACHE_NP[key] = (rand_idx1, rand_idx2)

    rand_idx1, rand_idx2 = _ANDES_INDEX_CACHE_NP[key]   # (ite, len1), (ite, len2)

    # vectorized scoring over cached indices
    back_mats = mat[rand_idx1[:, :, None], rand_idx2[:, None, :]]  # (ite, len1, len2)

    max_cols = back_mats.max(axis=1)   # (ite, len2)
    max_rows = back_mats.max(axis=2)   # (ite, len1)
    back_scores = (max_cols.sum(axis=1) + max_rows.sum(axis=1)) / (len1 + len2)

    mean_back = back_scores.mean()
    std_back = back_scores.std()
    if std_back == 0:
        z_score = 0.0
    else:
        z_score = (true_score - mean_back) / std_back

    return float(true_score), float(z_score)

# ============================================================
# Numba-compiled helper functions
# ============================================================

@njit(cache=True)
def _compute_bma_from_indices(matrix, idx1, idx2):
    """
    Compute best_match_average directly from indices in a single pass.
    - Avoids forming a submatrix
    - Visits each (i, j) entry exactly once
    """
    n1 = idx1.size
    n2 = idx2.size

    # you can assume n1, n2 > 0 in your use case; if not, guard here
    row_max = np.empty(n1)
    col_max = np.empty(n2)

    # initialize with very small numbers (or matrix[idx1[0], idx2[0]] if you prefer)
    for i in range(n1):
        row_max[i] = -1e308  # ~ -inf for float64
    for j in range(n2):
        col_max[j] = -1e308

    # single pass: update both row and column maxima
    for i in range(n1):
        r = idx1[i]
        for j in range(n2):
            c = idx2[j]
            val = matrix[r, c]
            if val > row_max[i]:
                row_max[i] = val
            if val > col_max[j]:
                col_max[j] = val

    row_sum = 0.0
    for i in range(n1):
        row_sum += row_max[i]

    col_sum = 0.0
    for j in range(n2):
        col_sum += col_max[j]

    return (row_sum + col_sum) / (n1 + n2)

@njit(cache=True, parallel=True)
def _andes_background_parallel(matrix, n1, n2, pop1, pop2, ite, seed):
    """
    Parallel computation of background scores.
    Uses sampling WITH replacement for speed (statistically equivalent for large populations).
    """
    back_scores = np.empty(ite)
    
    for i in prange(ite):
        # Different seed per iteration for proper parallelism
        np.random.seed(seed + i)
        ri1 = np.random.choice(pop1, n1, replace=False)
        ri2 = np.random.choice(pop2, n2, replace=False)
        back_scores[i] = _compute_bma_from_indices(matrix, ri1, ri2)
    
    return back_scores


@njit(cache=True, parallel=True)
def _andes_core_cached(matrix, idx1, idx2, rand_idx1, rand_idx2):
    """
    Numba core that assumes random indices are precomputed and cached.

    Parameters
    ----------
    matrix : 2D float64 array (full similarity matrix, C-contiguous)
    idx1   : 1D int64 array for term1 indices
    idx2   : 1D int64 array for term2 indices
    rand_idx1 : 2D int64 array, shape (ite, len(idx1))
    rand_idx2 : 2D int64 array, shape (ite, len(idx2))

    Returns
    -------
    true_score : float64
    z_score    : float64
    """
    # compute true score once
    true_score = _compute_bma_from_indices(matrix, idx1, idx2)

    ite = rand_idx1.shape[0]
    back_scores = np.empty(ite)

    for i in prange(ite):
        back_scores[i] = _compute_bma_from_indices(
            matrix,
            rand_idx1[i],
            rand_idx2[i],
        )

    mean_back = back_scores.mean()
    std_back = back_scores.std()
    if std_back == 0.0:
        z_score = 0.0
    else:
        z_score = (true_score - mean_back) / std_back

    return true_score, z_score

# ============================================================
# Numba-accelerated versions
# ============================================================

def andes_numba_parallel(terms, matrix, g1_term2index, g2_term2index, 

                          g1_population, g2_population, ite=1000, distinct=False,
                          seed=None):
    """
    Numba-parallelized andes function.
    
    Note: Uses sampling WITH replacement for background, which is
    statistically equivalent for large populations but much faster.
    """
    term1, term2 = terms
    
    indexes1 = list(g1_term2index[term1])
    indexes2 = list(g2_term2index[term2])
    
    if distinct:
        indexes2 = list(set(indexes2).difference(indexes1))
        if len(indexes2) < 10:
            return (0, 0)
    
    # Ensure contiguous int64 arrays for numba
    idx1 = np.ascontiguousarray(indexes1, dtype=np.int64)
    idx2 = np.ascontiguousarray(indexes2, dtype=np.int64)
    pop1 = np.ascontiguousarray(g1_population, dtype=np.int64)
    pop2 = np.ascontiguousarray(g2_population, dtype=np.int64)
    
    # Ensure matrix is contiguous
    if not matrix.flags['C_CONTIGUOUS']:
        matrix = np.ascontiguousarray(matrix)
    
    n1, n2 = len(idx1), len(idx2)
    
    # Compute true score
    true_score = _compute_bma_from_indices(matrix, idx1, idx2)
    
    # Compute background scores in parallel
    if seed is None:
        seed = np.random.randint(0, 2**31)
    
    back_scores = _andes_background_parallel(matrix, n1, n2, pop1, pop2, ite, seed)
    
    z_score = (true_score - back_scores.mean()) / back_scores.std()
    return (true_score, z_score)

# global cache: (len1, len2, ite, seed) -> (rand_idx1, rand_idx2)
_NUMBA_INDEX_CACHE = {}

def andes_numba_cached(
    terms,
    matrix,
    g1_term2index,
    g2_term2index,
    g1_population,
    g2_population,
    ite=1000,
    distinct=False,
    seed=12345,
):
    """
    ANDES with:
      - Numba-accelerated scoring
      - Python-level cached random indices per (len1, len2, ite, seed).

    This keeps the same null model as the Python cached version
    (sampling WITHOUT replacement) but uses Numba for the heavy loops.
    """
    term1, term2 = terms

    # --- term indices ---
    indexes1 = list(g1_term2index[term1])
    indexes2 = list(g2_term2index[term2])

    if distinct:
        indexes2 = list(set(indexes2).difference(indexes1))
        if len(indexes2) < 10:
            return (0.0, 0.0)

    # contiguous int64 arrays for numba
    idx1 = np.ascontiguousarray(indexes1, dtype=np.int64)
    idx2 = np.ascontiguousarray(indexes2, dtype=np.int64)

    pop1 = np.ascontiguousarray(g1_population, dtype=np.int64)
    pop2 = np.ascontiguousarray(g2_population, dtype=np.int64)

    # ensure matrix is contiguous
    if not matrix.flags["C_CONTIGUOUS"]:
        mat = np.ascontiguousarray(matrix)
    else:
        mat = matrix

    len1 = idx1.size
    len2 = idx2.size

    key = (len1, len2, ite, seed)

    # --- build / reuse cache entry ---
    if key not in _NUMBA_INDEX_CACHE:
        rng = np.random.default_rng(seed)

        rand_idx1 = np.stack(
            [rng.choice(pop1, size=len1, replace=False) for _ in range(ite)],
            axis=0,
        ).astype(np.int64)

        rand_idx2 = np.stack(
            [rng.choice(pop2, size=len2, replace=False) for _ in range(ite)],
            axis=0,
        ).astype(np.int64)

        _NUMBA_INDEX_CACHE[key] = (
            np.ascontiguousarray(rand_idx1),
            np.ascontiguousarray(rand_idx2),
        )

    rand_idx1, rand_idx2 = _NUMBA_INDEX_CACHE[key]

    true_score, z_score = _andes_core_cached(mat, idx1, idx2, rand_idx1, rand_idx2)
    return float(true_score), float(z_score)

def t_score_with_background_correction(terms, matrix, g1_term2index, g2_term2index,
                                       g1_population, g2_population,
                                       ite=1000, distinct=False):
    """
    Given two term annotations, calculate t-score with background correction.

    This is still loop-based (for clarity), but uses NumPy efficiently.
    """
    term1, term2 = terms

    mat = np.asarray(matrix)

    g1_true_index = np.array(list(g1_term2index[term1]), dtype=int)
    g2_true_index = np.array(list(g2_term2index[term2]), dtype=int)

    if distinct:
        g2_true_index = np.array(list(set(g2_true_index).difference(g1_true_index)), dtype=int)
        if g2_true_index.size < 10:
            return (0.0, 0.0)

    true_matrix = mat[np.ix_(g1_true_index, g2_true_index)]
    back_matrix1 = mat[np.ix_(g1_true_index, g2_population)]
    back_matrix2 = mat[np.ix_(g1_population, g2_true_index)]

    true_score = t_score(true_matrix, back_matrix1, back_matrix2)

    rand_scores = np.empty(ite, dtype=float)

    len1 = g1_true_index.size
    len2 = g2_true_index.size

    for k in range(ite):
        rand1 = random.sample(g1_population, len1)
        rand2 = random.sample(g2_population, len2)

        rand_true = mat[np.ix_(rand1, rand2)]
        rand_b1 = mat[np.ix_(rand1, g2_population)]
        rand_b2 = mat[np.ix_(g1_population, rand2)]

        rand_scores[k] = t_score(rand_true, rand_b1, rand_b2)

    mean_rand = rand_scores.mean()
    std_rand = rand_scores.std()
    if std_rand == 0:
        z_scores = 0.0
    else:
        z_scores = (true_score - mean_rand) / std_rand

    return float(true_score), float(z_scores)


def best_match_ranked_list(matrix):
    """
    Given the pairwise similarity between a ranked list and a known gene set,
    calculate the maximal absolute deviation of the running sum.
    """
    mat = np.asarray(matrix)
    maxs = mat.max(axis=0)
    centered = maxs - maxs.mean()
    cumsum = np.cumsum(centered)
    # maximum absolute deviation
    return float(cumsum[np.abs(cumsum).argmax()])


def gsea_andes(term, ranked_list, matrix, term2indices, annotated_indices, ite=1000):
    """
    Given gene set and ranked list information,
    calculate the corrected enrichment z-score using ANDES-like null.
    """
    mat = np.asarray(matrix)
    term_indices = np.array(list(term2indices[term]), dtype=int)

    # true score
    true_score = best_match_ranked_list(mat[np.ix_(term_indices, ranked_list)])

    # background
    len_term = term_indices.size
    rand_scores = np.empty(ite, dtype=float)

    for i in range(ite):
        rand_idx = random.sample(annotated_indices, len_term)
        back_matrix = mat[np.ix_(rand_idx, ranked_list)]
        rand_scores[i] = best_match_ranked_list(back_matrix)

    mean_back = rand_scores.mean()
    std_back = rand_scores.std()
    if std_back == 0:
        z_score = 0.0
    else:
        z_score = (true_score - mean_back) / std_back

    return float(true_score), float(z_score)

@njit(cache=True)
def _gsea_score_from_indices(matrix, term_idx, ranked):
    """
    Numba version of best_match_ranked_list on a submatrix defined by:
      - rows   = term_idx
      - columns = ranked

    It:
      1) takes the max across rows for each ranked column
      2) centers by the mean
      3) computes the running sum
      4) returns the maximum absolute deviation of the running sum
    """
    n_term = term_idx.size
    n_rank = ranked.size

    # 1) column-wise maxima over the term_idx rows
    maxs = np.empty(n_rank)
    for j in range(n_rank):
        c = ranked[j]
        # initialize with the first row
        v = matrix[term_idx[0], c]
        for i in range(1, n_term):
            val = matrix[term_idx[i], c]
            if val > v:
                v = val
        maxs[j] = v

    # 2) mean of maxs
    s = 0.0
    for j in range(n_rank):
        s += maxs[j]
    mean_val = s / n_rank

    # 3) running sum of centered maxs, 4) max abs deviation
    running = 0.0
    max_abs = 0.0
    for j in range(n_rank):
        running += maxs[j] - mean_val
        if running >= 0:
            if running > max_abs:
                max_abs = running
        else:
            if -running > max_abs:
                max_abs = -running

    return max_abs


@njit(cache=True, parallel=True)
def _gsea_core_cached(matrix, term_idx, ranked, rand_idx):
    """
    Numba core for GSEA-ANDES with cached random indices.

    Parameters
    ----------
    matrix : 2D float64 array, full similarity matrix (C-contiguous)
    term_idx : 1D int64 array, indices of the true gene set
    ranked : 1D int64 array, ranked gene indices
    rand_idx : 2D int64 array, shape (ite, len(term_idx))
              each row is a background gene set

    Returns
    -------
    true_score : float64
    z_score    : float64
    """
    true_score = _gsea_score_from_indices(matrix, term_idx, ranked)

    ite = rand_idx.shape[0]
    back_scores = np.empty(ite)

    for i in prange(ite):
        back_scores[i] = _gsea_score_from_indices(matrix, rand_idx[i], ranked)

    mean_back = back_scores.mean()
    std_back = back_scores.std()

    if std_back == 0.0:
        z_score = 0.0
    else:
        z_score = (true_score - mean_back) / std_back

    return true_score, z_score

_GSEA_NUMBA_INDEX_CACHE = {}

def gsea_andes_numba_cached(
    term,
    ranked_list,
    matrix,
    term2indices,
    annotated_indices,
    ite=1000,
    seed=12345,
):
    """
    Cached + Numba-accelerated GSEA-ANDES.

    Parameters
    ----------
    term : hashable
        Gene set / GO term identifier (key into term2indices).
    ranked_list : list/array of int
        Ranked gene indices (embedding indices).
    matrix : array-like
        Full similarity matrix S.
    term2indices : dict
        term -> iterable of gene indices.
    annotated_indices : list/array of int
        Background pool to sample from.
    ite : int
        Number of Monte Carlo samples.
    seed : int
        Random seed for generating (and caching) background sets.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if not mat.flags["C_CONTIGUOUS"]:
        mat = np.ascontiguousarray(mat)

    ranked = np.asarray(ranked_list, dtype=np.int64)
    term_indices = np.asarray(list(term2indices[term]), dtype=np.int64)
    if term_indices.size == 0:
        return 0.0, 0.0

    annotated = np.asarray(annotated_indices, dtype=np.int64)
    len_term = term_indices.size

    # ----- build / reuse cached random indices -----
    key = (len_term, ite, seed)
    if key not in _GSEA_NUMBA_INDEX_CACHE:
        rng = np.random.default_rng(seed)
        rand_idx = np.stack(
            [rng.choice(annotated, size=len_term, replace=False) for _ in range(ite)],
            axis=0,
        ).astype(np.int64)
        rand_idx = np.ascontiguousarray(rand_idx)
        _GSEA_NUMBA_INDEX_CACHE[key] = rand_idx

    rand_idx = _GSEA_NUMBA_INDEX_CACHE[key]  # shape (ite, len_term)

    # ----- call numba core -----
    true_score, z_score = _gsea_core_cached(mat, term_indices, ranked, rand_idx)
    return float(true_score), float(z_score)

