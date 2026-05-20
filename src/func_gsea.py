"""
func_gsea.py — optimized GSEA-ANDES null cache and scoring primitives

Improvements over the original (set_analysis_func.gsea_andes):

  1. BLAS thread limits set INSIDE worker initializer (not just parent env),
     belt-and-suspenders with threadpool_limits. Fixes macOS 'spawn' workers
     and any platform where env vars are read once at numpy import.

  2. Auto-tuned chunk size (~4 chunks per worker). For 249 sizes and 8
     workers, chunk_size goes from the original 16 (only 16 chunks total,
     2 per worker, poor load balance) to ~7 (~36 chunks, ~4-5 per worker).

  3. Per-size deterministic seeding via SeedSequence([seed, m]), independent
     of how chunks are partitioned. Fixes the chunk_seed = sum(sizes_chunk)
     collision in the original.

  4. No full n×n cosine_similarity matrix — scoring is done via direct
     BLAS matmul on embedding blocks.

  5. Batched Monte Carlo iterations per GEMM. Instead of one (m, d)@(d, L)
     call per iteration, batch b iterations into a single (b*m, d)@(d, L)
     GEMM, then reshape and reduce. Batch size b is chosen so that b*m*L*4
     bytes fits in _ES_BATCH_BYTES (default 64 MB). For m=50, L=10000 this
     gives b≈32, turning 1000 small GEMMs into 32 large ones per size —
     BLAS utilisation improves substantially for small gene sets. The
     cumsum+argmax ES reduction is done by a numba kernel with no Python
     overhead per iteration.

Usage
-----
  from func_gsea import NullCacheESBetter

  cache = NullCacheESBetter()
  cache.precompute_parallel(
      E_unit, pop, sizes, ranked_emb,
      ite=1000, seed=12345, verbose=True,
      n_workers=8,                    # default: min(8, cpu_count())
      chunk_size=None,                # default: auto-tuned
      blas_threads_per_worker=1,      # default: 1
  )
"""

import os
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import jit
from tqdm import tqdm


_ES_BATCH_BYTES = 64 * 1024 * 1024  # workspace cap for batched ES GEMM (64 MB)


def _hash_array(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


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


@jit(nopython=True, nogil=True, cache=True)
def _batch_fys_sample(perm, js, out):
    """Apply b independent partial Fisher-Yates samples on perm, restoring it.

    perm: int32 (N,) — identity [0..N-1] on entry and exit.
    js:   int64 (b, m) — swap targets; js[i, j] must lie in [j, N-1].
    out:  int32 (b, m) — receives the b sampled local-index vectors.
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


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring functions (used by both the null builder and callers)
# ─────────────────────────────────────────────────────────────────────────────


def compute_ranked_emb(E_unit, ranked_list_idx):
    """Extract the ranked-list embedding block. Call once per ranked list."""
    return np.ascontiguousarray(
        E_unit[np.asarray(ranked_list_idx, dtype=np.int32)], dtype=np.float32
    )


def compute_es_score(E_unit, gene_set_idx, ranked_emb):
    """Signed enrichment score from L2-normalized embeddings."""
    X = E_unit[np.asarray(gene_set_idx, dtype=np.int32)]
    col_max = (X @ ranked_emb.T).max(axis=0)
    cs = np.cumsum(col_max - col_max.mean())
    return float(cs[np.abs(cs).argmax()])


def compute_es_trace(E_unit, gene_set_idx, ranked_emb):
    """Full enrichment trace (for plotting). ES matches compute_es_score."""
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
    """Trigger JIT compilation for all ES numba kernels before timed code."""
    E = np.eye(10, dtype=np.float32)
    idx = np.arange(4, dtype=np.int32)
    rem = np.ascontiguousarray(E[:6], dtype=np.float32)
    _es_score_numba_small(E, idx, rem)

    perm = np.arange(10, dtype=np.int32)
    js_single = np.array([0, 2, 3, 4], dtype=np.int64)
    _fys_sample_single(perm, js_single)

    perm2 = np.arange(10, dtype=np.int32)
    js_batch = np.array([[0, 2, 3, 4], [1, 3, 4, 5]], dtype=np.int64)
    out_batch = np.empty((2, 4), dtype=np.int32)
    _batch_fys_sample(perm2, js_batch, out_batch)

    col_max = np.zeros((2, 6), dtype=np.float32)
    scores_buf = np.zeros(4, dtype=np.float32)
    _es_scores_from_col_max(col_max, scores_buf, 0)


def _chunk_sizes_by_cost(sizes, n_workers, target_chunks_per_worker=4):
    sizes = sorted((int(m) for m in sizes), reverse=True)
    if not sizes:
        return []

    n_chunks = min(len(sizes), max(1, int(n_workers) * target_chunks_per_worker))
    chunks = [[] for _ in range(n_chunks)]
    costs = [0] * n_chunks

    for m in sizes:
        idx = min(range(n_chunks), key=costs.__getitem__)
        chunks[idx].append(m)
        costs[idx] += m

    return [sorted(chunk) for chunk in chunks if chunk]


# ─────────────────────────────────────────────────────────────────────────────
# Score function (per-iteration, BLAS-backed)
# ─────────────────────────────────────────────────────────────────────────────


@jit(nopython=True, cache=True)
def _es_score_numba_small(E_unit, gene_set_idx, ranked_emb):
    """Numba scalar kernel for very small m * L. Rarely useful for full
    ranked lists, kept for completeness."""
    m = gene_set_idx.shape[0]
    L = ranked_emb.shape[0]
    d = E_unit.shape[1]

    col_max = np.full(L, -1e9, dtype=np.float32)
    for i in range(m):
        gi = gene_set_idx[i]
        for j in range(L):
            dot = 0.0
            for k in range(d):
                dot += E_unit[gi, k] * ranked_emb[j, k]
            if dot > col_max[j]:
                col_max[j] = dot

    mean_val = 0.0
    for j in range(L):
        mean_val += col_max[j]
    mean_val /= L

    running = 0.0
    max_abs = 0.0
    best_signed = 0.0
    for j in range(L):
        running += col_max[j] - mean_val
        a = running if running >= 0.0 else -running
        if a > max_abs:
            max_abs = a
            best_signed = running
    return best_signed


def _es_score_blas(E_unit, gene_set_idx, ranked_emb, A_buf, col_max_buf):
    """One ES score via BLAS. Uses caller-provided workspace buffers to
    avoid per-iteration allocation.

    Memory: only (m, L) for A_buf and (L,) for col_max_buf, never (B, m, L).
    """
    m = len(gene_set_idx)
    # Gather rows: NumPy fancy index allocates an (m, d) array; this is the
    # one allocation we cannot avoid without numba, but it is small.
    X = E_unit[gene_set_idx]  # (m, d)
    np.matmul(X, ranked_emb.T, out=A_buf[:m])  # (m, L) -> A_buf
    A_buf[:m].max(axis=0, out=col_max_buf)  # (L,)
    mean = col_max_buf.mean()
    cs = np.cumsum(col_max_buf - mean)  # (L,) - one alloc here
    return float(cs[np.abs(cs).argmax()])


@jit(nopython=True, nogil=True, cache=True)
def _es_scores_from_col_max(col_max_batch, scores, offset):
    """Compute ES scores for each row of col_max_batch without Python overhead.

    col_max_batch: float32 (b, L) — b precomputed col_max vectors
    scores:        float32 (ite,) — output array; writes to scores[offset:offset+b]
    offset:        int — starting position in scores
    """
    b = col_max_batch.shape[0]
    L = col_max_batch.shape[1]
    for i in range(b):
        mean_val = 0.0
        for j in range(L):
            mean_val += col_max_batch[i, j]
        mean_val /= L
        running = 0.0
        max_abs = 0.0
        best_signed = 0.0
        for j in range(L):
            running += col_max_batch[i, j] - mean_val
            a = running if running >= 0.0 else -running
            if a > max_abs:
                max_abs = a
                best_signed = running
        scores[offset + i] = best_signed


# ─────────────────────────────────────────────────────────────────────────────
# Worker globals and initializer
# ─────────────────────────────────────────────────────────────────────────────

_W_E_UNIT = None
_W_RANKED_EMB = None
_W_POP = None
_W_ITE = None
_W_SEED = None
_W_USE_NUMBA_BELOW = None


def _init_worker(
    e_path, ranked_path, pop_arr, ite, seed, blas_threads, use_numba_below,
):
    """Set BLAS thread caps inside the worker, load the embeddings via
    mmap so all workers share the same backing pages."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(blas_threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(blas_threads)
    except Exception:
        pass

    global _W_E_UNIT, _W_RANKED_EMB, _W_POP, _W_ITE, _W_SEED
    global _W_USE_NUMBA_BELOW
    _W_E_UNIT = np.load(e_path, mmap_mode="r")
    _W_RANKED_EMB = np.load(ranked_path, mmap_mode="r")
    _W_POP = pop_arr
    _W_ITE = int(ite)
    _W_SEED = int(seed)
    _W_USE_NUMBA_BELOW = int(use_numba_below)


def _compute_sizes_chunk(sizes_chunk):
    """Compute null ES distribution for each m in sizes_chunk.

    Sampling: O(m) partial Fisher-Yates (FYS) per iteration.
    Scoring: b iterations batched into one (b*m, d)@(d, L) GEMM, where b is
    chosen so that b*m*L*4 bytes ≤ _ES_BATCH_BYTES (default 64 MB). The
    cumsum+argmax ES reduction runs in a numba kernel with no Python overhead
    per iteration.
    """
    if not sizes_chunk:
        return []

    pop = _W_POP
    N_pop = pop.shape[0]
    L = _W_RANKED_EMB.shape[0]
    ite = _W_ITE

    # A_buf rows: how many (m, L)-sized rows fit in the workspace budget.
    max_A_rows = max(1, _ES_BATCH_BYTES // (L * 4))
    A_buf = np.empty((max_A_rows, L), dtype=np.float32)

    sizes_sorted = sorted(sizes_chunk)
    min_m = sizes_sorted[0]
    max_m = sizes_sorted[-1]

    # Pre-allocate worst-case buffers (reused across all sizes).
    max_b = min(ite, max(1, max_A_rows // min_m))
    col_max_batch = np.empty((max_b, L), dtype=np.float32)
    xi_buf = np.empty((max_b, max_m), dtype=np.int32)
    scores = np.empty(ite, dtype=np.float32)
    perm = np.arange(N_pop, dtype=np.int32)

    out = []
    for m in sizes_sorted:
        rng = np.random.default_rng(np.random.SeedSequence([_W_SEED, int(m)]))
        m_low = np.arange(m, dtype=np.int64)
        b = min(ite, max(1, max_A_rows // m))

        if m * L < _W_USE_NUMBA_BELOW:
            for i in range(ite):
                local = _fys_sample_single(perm, rng.integers(m_low, N_pop, dtype=np.int64))
                scores[i] = _es_score_numba_small(_W_E_UNIT, pop[local], _W_RANKED_EMB)
        else:
            xi = xi_buf[:b, :m]       # view into pre-allocated buffer, no alloc
            cm = col_max_batch[:b]    # view into pre-allocated buffer, no alloc
            done = 0
            while done < ite:
                b_act = min(b, ite - done)

                # Batch b_act FYS samples at once (O(b_act * m) numba ops).
                js = rng.integers(m_low, N_pop, size=(b_act, m), dtype=np.int64)
                _batch_fys_sample(perm, js, xi[:b_act])

                # Flat gather: (b_act * m,) → one mmap access instead of b_act separate ones.
                gene_idx = pop[xi[:b_act].ravel()]
                X = _W_E_UNIT[gene_idx]   # (b_act * m, d)

                # Single large GEMM: (b_act*m, d) @ (d, L) → (b_act*m, L).
                np.matmul(X, _W_RANKED_EMB.T, out=A_buf[:b_act * m])

                # Reduce max over the m-axis: (b_act, m, L) → (b_act, L).
                A_buf[:b_act * m].reshape(b_act, m, L).max(axis=1, out=cm[:b_act])

                # Compute b_act ES scores in numba — no Python loop per iteration.
                _es_scores_from_col_max(cm[:b_act], scores, done)

                done += b_act

        mu = float(scores.mean())
        std = float(scores.std(ddof=1)) if ite > 1 else 0.0
        out.append((int(m), (mu, std)))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cache class
# ─────────────────────────────────────────────────────────────────────────────


class NullCacheESBetter:
    """
    Same query-time interface as NullCacheES. Different build path:
      - BLAS pinned per worker
      - chunk_size auto-tuned for load balance
      - per-size deterministic seeding
      - per-worker workspace (m_max, L), reused across all sizes

    Peak memory (per worker): ~ max_m * L * 4 + L * 4 bytes
                              ~21 MB at max_m=300, L=18000.
    Times n_workers, plus one shared mmap of E_unit and ranked_emb.
    """

    def __init__(self):
        self.cache: dict[int, tuple[float, float]] = {}
        self.metadata: dict = {}

    @staticmethod
    def build_metadata(E_unit, pop, ranked_emb, ite, seed):
        return {
            "kind": "andes_gsea_es_null",
            "version": 2,
            "embedding_hash": _hash_array(np.asarray(E_unit, dtype=np.float32)),
            "population_hash": _hash_array(np.asarray(pop, dtype=np.int32)),
            "ranked_emb_hash": _hash_array(np.asarray(ranked_emb, dtype=np.float32)),
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

    # ---- query (hot path) ---------------------------------------------------

    def get_zscore(self, true_score: float, m: int) -> float:
        mu, sigma = self.cache[int(m)]
        if sigma == 0.0:
            return 0.0
        return (true_score - mu) / sigma

    # ---- build --------------------------------------------------------------

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
        chunk_size: int | None = None,
        blas_threads_per_worker: int = 1,
        use_numba_below: int = 4_000,
        show_progress: bool | None = None,
    ):
        """Build the null cache in parallel. Memory-bounded version: per
        worker, one (max_m, L) workspace plus shared mmap of E_unit and
        ranked_emb. Total resident memory roughly:
            n_workers * max_m * L * 4 bytes  (workspaces)
          + |E_unit| + |ranked_emb|          (shared, mmap)
        For n_workers=8, max_m=300, L=18000: ~170 MB workspaces + ~30 MB mmap.
        """
        seed = self.resolve_seed(seed)
        sizes_sorted = sorted(set(int(m) for m in gene_set_sizes))
        if not sizes_sorted:
            return
        self.metadata = self.build_metadata(E_unit, pop, ranked_emb, ite, seed)

        if n_workers is None:
            n_workers = min(8, os.cpu_count() or 1)

        todo = [m for m in sizes_sorted if m not in self.cache]
        if not todo:
            if verbose:
                print(f"All {len(sizes_sorted)} sizes already cached.")
            return

        max_m = max(todo)
        if chunk_size is None or chunk_size <= 0:
            chunks = _chunk_sizes_by_cost(todo, n_workers)
            chunk_desc = "auto-cost"
        else:
            chunks = [todo[i : i + chunk_size] for i in range(0, len(todo), chunk_size)]
            chunk_desc = str(chunk_size)
        if verbose:
            ws_mb = max_m * ranked_emb.shape[0] * 4 / 1e6
            print(
                f"Parallel ES null: {len(todo)} sizes  "
                f"workers={n_workers}  chunk_size={chunk_desc}  "
                f"chunks={len(chunks)}  max_m={max_m}\n"
                f"  per-worker workspace: ~{ws_mb:.0f} MB  "
                f"(total: ~{ws_mb * n_workers:.0f} MB)"
            )

        # Spill embeddings to disk so workers share via mmap rather than
        # pickling and copying. Use unique names so concurrent runs do not
        # clobber each other.
        e_path = f"E_unit_es_{os.getpid()}.npy"
        ranked_path = f"ranked_emb_es_{os.getpid()}.npy"
        try:
            np.save(e_path, np.ascontiguousarray(E_unit, dtype=np.float32))
            np.save(ranked_path, np.ascontiguousarray(ranked_emb, dtype=np.float32))

            pop_arr = np.asarray(pop, dtype=np.int32)

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(
                    e_path,
                    ranked_path,
                    pop_arr,
                    int(ite),
                    int(seed),
                    int(blas_threads_per_worker),
                    int(use_numba_below),
                ),
            ) as ex:
                it = ex.map(_compute_sizes_chunk, chunks)
                _show = show_progress if show_progress is not None else verbose
                if _show:
                    it = tqdm(it, total=len(chunks), desc="ES null (parallel)")
                for chunk_res in it:
                    for m, val in chunk_res:
                        self.cache[m] = val
        finally:
            for p in (e_path, ranked_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

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
        use_numba_below: int = 4_000,
    ):
        """Sequential build through the same scoring kernels."""
        seed = self.resolve_seed(seed)
        sizes_sorted = sorted(set(int(m) for m in gene_set_sizes))
        todo = [m for m in sizes_sorted if m not in self.cache]
        if not todo:
            return
        self.metadata = self.build_metadata(E_unit, pop, ranked_emb, ite, seed)

        global _W_E_UNIT, _W_RANKED_EMB, _W_POP, _W_ITE, _W_SEED
        global _W_USE_NUMBA_BELOW
        _W_E_UNIT = np.ascontiguousarray(E_unit, dtype=np.float32)
        _W_RANKED_EMB = np.ascontiguousarray(ranked_emb, dtype=np.float32)
        _W_POP = np.asarray(pop, dtype=np.int32)
        _W_ITE = int(ite)
        _W_SEED = int(seed)
        _W_USE_NUMBA_BELOW = int(use_numba_below)

        iterator = tqdm([todo], desc="ES null (seq)", disable=not verbose)
        for chunk in iterator:
            for m, val in _compute_sizes_chunk(chunk):
                self.cache[m] = val

        if verbose:
            print(f"Cached {len(self.cache)} ES null distributions")

    # ---- persistence --------------------------------------------------------

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {"metadata": self.metadata, "cache": self.cache},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved {len(self.cache)} ES cache entries to {path}")

    @classmethod
    def load(cls, path: str) -> "NullCacheESBetter":
        obj = cls()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "cache" in payload:
            obj.cache = payload["cache"]
            obj.metadata = payload.get("metadata", {})
        else:
            obj.cache = payload
            obj.metadata = {}
        print(f"Loaded {len(obj.cache)} ES cache entries from {path}")
        return obj

    @staticmethod
    def suggest_path(base_dir, E_unit, pop, ranked_emb):
        """Return a content-addressed cache path under base_dir.

        Different embeddings, populations, or ranked lists get distinct
        filenames so re-runs with new data never reuse a stale cache.
        """
        emb_h = _hash_array(np.asarray(E_unit, dtype=np.float32))[:8]
        pop_h = _hash_array(np.asarray(pop, dtype=np.int32))[:8]
        rank_h = _hash_array(np.asarray(ranked_emb, dtype=np.float32))[:8]
        return os.path.join(base_dir, f"es_{emb_h}_{pop_h}_{rank_h}.pkl")

    def __len__(self):
        return len(self.cache)

    def __contains__(self, m):
        return int(m) in self.cache

    def missing_sizes(self, gene_set_sizes):
        return [m for m in gene_set_sizes if int(m) not in self.cache]


# """
# func_gsea.py — optimized GSEA-ANDES null cache and scoring primitives
#
# v3 optimizations over v2:
#
#   1. Precompute S_full = E[pop] @ ranked_emb.T once (|pop|, L); each null
#      iteration is a row-gather from S_full — no GEMM in the inner loop.
#
#   2. Incremental col_max across sizes: one random permutation of pop per outer
#      iteration, walk from index 0 to m_max, snapshot ES at each target size.
#      Total work: O(ite × m_max × L) instead of O(ite × sum(sizes) × L).
#      At sizes=10..300 with ite=1000: ~80× fewer inner-loop ops than v2.
#
#   3. Welford online stats: workers hold (n, mean, M2) per size — O(|sizes|)
#      memory regardless of ite. Parallel results merged via the parallel-Welford
#      formula.
#
#   4. Parallelism by iteration chunk (not size chunk): every chunk costs m_max
#      gathers, giving perfect load balance.
#
#   5. _es_from_col_max: single-sweep numba kernel replaces cumsum + argmax.
# """
#
# import os
# import pickle
# import hashlib
# from concurrent.futures import ProcessPoolExecutor
#
# import numpy as np
# from numba import jit
# from tqdm import tqdm
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
# # ─────────────────────────────────────────────────────────────────────────────
# # Public scoring functions (unchanged API)
# # ─────────────────────────────────────────────────────────────────────────────
#
# def compute_ranked_emb(E_unit, ranked_list_idx):
#     """Extract the ranked-list embedding block. Call once per ranked list."""
#     return np.ascontiguousarray(
#         E_unit[np.asarray(ranked_list_idx, dtype=np.int32)], dtype=np.float32
#     )
#
#
# def compute_es_score(E_unit, gene_set_idx, ranked_emb):
#     """Signed enrichment score from L2-normalized embeddings."""
#     X = E_unit[np.asarray(gene_set_idx, dtype=np.int32)]
#     col_max = (X @ ranked_emb.T).max(axis=0)
#     cs = np.cumsum(col_max - col_max.mean())
#     return float(cs[np.abs(cs).argmax()])
#
#
# def compute_es_trace(E_unit, gene_set_idx, ranked_emb):
#     """Full enrichment trace (for plotting). ES matches compute_es_score."""
#     gene_set_idx = np.asarray(gene_set_idx, dtype=np.int32)
#     X = E_unit[gene_set_idx]
#     A = X @ ranked_emb.T
#     best_gene_set_position = A.argmax(axis=0).astype(np.int32)
#     cols = np.arange(A.shape[1])
#     best_match_score = A[best_gene_set_position, cols].astype(np.float32)
#     centered_score   = (best_match_score - best_match_score.mean()).astype(np.float32)
#     running_es       = np.cumsum(centered_score, dtype=np.float32)
#     es_index = int(np.abs(running_es).argmax())
#     return {
#         "best_match_score":       best_match_score,
#         "best_gene_set_position": best_gene_set_position,
#         "centered_score":         centered_score,
#         "running_es":             running_es,
#         "es_index":               es_index,
#         "es":                     float(running_es[es_index]),
#     }
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Numba kernels
# # ─────────────────────────────────────────────────────────────────────────────
#
# @jit(nopython=True, cache=True)
# def _es_from_col_max(v):
#     """ES score from a precomputed col_max vector. Single sweep, no allocation."""
#     L = v.shape[0]
#     s = 0.0
#     for j in range(L):
#         s += v[j]
#     mu = s / L
#     run = 0.0
#     best_abs = 0.0
#     best_signed = 0.0
#     for j in range(L):
#         run += v[j] - mu
#         a = run if run >= 0.0 else -run
#         if a > best_abs:
#             best_abs = a
#             best_signed = run
#     return best_signed
#
#
# @jit(nopython=True, cache=True)
# def _es_score_numba_small(E_unit, gene_set_idx, ranked_emb):
#     """Kept for external callers / warmup; not used in null building."""
#     m = gene_set_idx.shape[0]
#     L = ranked_emb.shape[0]
#     d = E_unit.shape[1]
#     col_max = np.full(L, -1e9, dtype=np.float32)
#     for i in range(m):
#         gi = gene_set_idx[i]
#         for j in range(L):
#             dot = 0.0
#             for k in range(d):
#                 dot += E_unit[gi, k] * ranked_emb[j, k]
#             if dot > col_max[j]:
#                 col_max[j] = dot
#     mean_val = 0.0
#     for j in range(L):
#         mean_val += col_max[j]
#     mean_val /= L
#     running = 0.0
#     max_abs = 0.0
#     best_signed = 0.0
#     for j in range(L):
#         running += col_max[j] - mean_val
#         a = running if running >= 0.0 else -running
#         if a > max_abs:
#             max_abs = a
#             best_signed = running
#     return best_signed
#
#
# def warmup_numba_es():
#     """Trigger JIT compilation before timed code."""
#     E   = np.eye(10, dtype=np.float32)
#     idx = np.arange(4, dtype=np.int32)
#     rem = np.ascontiguousarray(E[:6], dtype=np.float32)
#     _es_score_numba_small(E, idx, rem)
#     _es_from_col_max(np.zeros(6, dtype=np.float32))
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Iteration chunking and Welford merge
# # ─────────────────────────────────────────────────────────────────────────────
#
# def _chunk_iters(ite, n_workers, target_chunks_per_worker=4):
#     """Split [0, ite) into n_workers * target_chunks_per_worker ranges."""
#     n_chunks = max(1, n_workers * target_chunks_per_worker)
#     base = ite // n_chunks
#     extra = ite % n_chunks
#     ranges = []
#     start = 0
#     for i in range(n_chunks):
#         end = start + base + (1 if i < extra else 0)
#         if end > start:
#             ranges.append((start, end))
#         start = end
#     return ranges
#
#
# def _merge_welford(a, b):
#     """Combine two Welford (n, mean, M2) accumulators."""
#     na, mean_a, M2_a = a
#     nb, mean_b, M2_b = b
#     n = na + nb
#     if n == 0:
#         return (0, 0.0, 0.0)
#     delta = mean_b - mean_a
#     mean = mean_a + delta * nb / n
#     M2 = M2_a + M2_b + delta ** 2 * na * nb / n
#     return (n, mean, M2)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Worker globals and initializer
# # ─────────────────────────────────────────────────────────────────────────────
#
# _W_S_FULL = None        # (N_pop, L) — E[pop] @ ranked_emb.T, mmap shared
# _W_SEED = None
# _W_SIZES_SORTED = None  # sorted list of int sizes to snapshot
#
#
# def _init_worker(s_path, seed, sizes_sorted):
#     for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
#                 "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
#                 "NUMEXPR_NUM_THREADS"):
#         os.environ[var] = "1"
#     try:
#         from threadpoolctl import threadpool_limits
#         threadpool_limits(1)
#     except Exception:
#         pass
#     global _W_S_FULL, _W_SEED, _W_SIZES_SORTED
#     # mmap_mode='r': all workers share one physical copy via kernel page cache.
#     _W_S_FULL = np.load(s_path, mmap_mode="r")
#     _W_SEED = int(seed)
#     _W_SIZES_SORTED = list(sizes_sorted)
#
#
# def _compute_iter_chunk(iter_range):
#     """Run outer iterations [it_start, it_end) across all sizes.
#
#     One random permutation of pop per outer iteration.  col_max is accumulated
#     incrementally left-to-right; we snapshot ES whenever we reach a target size.
#     Returns {m: (n, mean, M2)} partial Welford statistics.
#     """
#     S = _W_S_FULL           # (N_pop, L), read-only mmap
#     sizes = _W_SIZES_SORTED  # sorted ints, ascending
#     seed = _W_SEED
#     N_pop, L = S.shape
#     m_max = sizes[-1]
#     it_start, it_end = iter_range
#
#     stats = {m: [0, 0.0, 0.0] for m in sizes}  # [n, mean, M2]
#     col_max = np.empty(L, dtype=np.float32)
#
#     for it in range(it_start, it_end):
#         rng = np.random.default_rng(np.random.SeedSequence([seed, it]))
#         xi = rng.choice(N_pop, size=m_max, replace=False, shuffle=False)
#
#         col_max.fill(-np.inf)
#         sizes_iter = iter(sizes)
#         next_m = next(sizes_iter)
#
#         for i in range(m_max):
#             np.maximum(col_max, S[xi[i]], out=col_max)
#             if i + 1 == next_m:
#                 es = float(_es_from_col_max(col_max))
#                 st = stats[next_m]
#                 st[0] += 1
#                 delta = es - st[1]
#                 st[1] += delta / st[0]
#                 st[2] += delta * (es - st[1])
#                 try:
#                     next_m = next(sizes_iter)
#                 except StopIteration:
#                     break
#
#     return {m: tuple(v) for m, v in stats.items()}
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Cache class
# # ─────────────────────────────────────────────────────────────────────────────
#
# class NullCacheESBetter:
#     """
#     Null distribution cache for GSEA enrichment scores.
#
#     Build path (v3): precomputes S_full = E[pop] @ ranked_emb.T once, then
#     runs an incremental prefix walk so all sizes share each outer iteration.
#     Total null work: O(ite × m_max × L) instead of O(ite × sum(sizes) × L).
#     """
#
#     def __init__(self):
#         self.cache: dict[int, tuple[float, float]] = {}
#         self.metadata: dict = {}
#
#     @staticmethod
#     def build_metadata(E_unit, pop, ranked_emb, ite, seed):
#         return {
#             "kind": "andes_gsea_es_null",
#             "version": 3,
#             "null_scheme": "incremental_perm",
#             "embedding_hash": _hash_array(np.asarray(E_unit, dtype=np.float32)),
#             "population_hash": _hash_array(np.asarray(pop, dtype=np.int32)),
#             "ranked_emb_hash": _hash_array(np.asarray(ranked_emb, dtype=np.float32)),
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
#     # ---- query ---------------------------------------------------------------
#
#     def get_zscore(self, true_score: float, m: int) -> float:
#         mu, sigma = self.cache[int(m)]
#         if sigma == 0.0:
#             return 0.0
#         return (true_score - mu) / sigma
#
#     # ---- build (parallel) ---------------------------------------------------
#
#     def precompute_parallel(
#         self,
#         E_unit,
#         pop,
#         gene_set_sizes,
#         ranked_emb,
#         ite: int = 1000,
#         seed: int = 12345,
#         verbose: bool = False,
#         n_workers: int | None = None,
#         chunk_size: int | None = None,   # ignored; kept for API compatibility
#         blas_threads_per_worker: int = 1, # ignored; kept for API compatibility
#         use_numba_below: int = 4_000,     # ignored; kept for API compatibility
#         show_progress: bool | None = None,
#     ):
#         seed = self.resolve_seed(seed)
#         sizes_sorted = sorted(set(int(m) for m in gene_set_sizes))
#         if not sizes_sorted:
#             return
#         todo = [m for m in sizes_sorted if m not in self.cache]
#         if not todo:
#             if verbose:
#                 print(f"All {len(sizes_sorted)} sizes already cached.")
#             return
#         self.metadata = self.build_metadata(E_unit, pop, ranked_emb, ite, seed)
#
#         if n_workers is None:
#             n_workers = min(8, os.cpu_count() or 1)
#
#         E_f = np.ascontiguousarray(E_unit, dtype=np.float32)
#         pop_arr = np.asarray(pop, dtype=np.int32)
#         ranked_f = np.ascontiguousarray(ranked_emb, dtype=np.float32)
#
#         s_path = f"S_es_{os.getpid()}.npy"
#         try:
#             # One GEMM upfront; workers share S_full via mmap.
#             S_full = (E_f[pop_arr] @ ranked_f.T).astype(np.float32)
#             np.save(s_path, S_full)
#             del S_full
#
#             iter_ranges = _chunk_iters(ite, n_workers)
#             n_chunks = len(iter_ranges)
#
#             if verbose:
#                 print(
#                     f"Parallel ES null: {len(todo)} sizes, ite={ite}, "
#                     f"workers={n_workers}, chunks={n_chunks}, m_max={todo[-1]}"
#                 )
#
#             merged = {m: (0, 0.0, 0.0) for m in todo}
#             with ProcessPoolExecutor(
#                 max_workers=n_workers,
#                 initializer=_init_worker,
#                 initargs=(s_path, int(seed), todo),
#             ) as ex:
#                 it = ex.map(_compute_iter_chunk, iter_ranges)
#                 _show = show_progress if show_progress is not None else verbose
#                 if _show:
#                     it = tqdm(it, total=n_chunks, desc="ES null (parallel)")
#                 for chunk_res in it:
#                     for m, partial in chunk_res.items():
#                         merged[m] = _merge_welford(merged[m], partial)
#
#             for m, (n, mean, M2) in merged.items():
#                 std = float(np.sqrt(M2 / (n - 1))) if n > 1 else 0.0
#                 self.cache[m] = (float(mean), std)
#
#         finally:
#             try:
#                 os.remove(s_path)
#             except OSError:
#                 pass
#
#         if verbose:
#             print(f"Cached {len(self.cache)} ES null distributions")
#
#     # ---- build (sequential) -------------------------------------------------
#
#     def precompute(
#         self,
#         E_unit,
#         pop,
#         gene_set_sizes,
#         ranked_emb,
#         ite: int = 1000,
#         seed: int = 12345,
#         verbose: bool = False,
#         use_numba_below: int = 4_000,  # ignored; kept for API compatibility
#     ):
#         """Sequential build using the incremental prefix approach."""
#         seed = self.resolve_seed(seed)
#         sizes_sorted = sorted(set(int(m) for m in gene_set_sizes))
#         todo = [m for m in sizes_sorted if m not in self.cache]
#         if not todo:
#             return
#         self.metadata = self.build_metadata(E_unit, pop, ranked_emb, ite, seed)
#
#         E_f = np.ascontiguousarray(E_unit, dtype=np.float32)
#         pop_arr = np.asarray(pop, dtype=np.int32)
#         ranked_f = np.ascontiguousarray(ranked_emb, dtype=np.float32)
#
#         # Precompute S_full = E[pop] @ ranked_emb.T once.
#         S_full = (E_f[pop_arr] @ ranked_f.T).astype(np.float32)
#         N_pop, L = S_full.shape
#         m_max = todo[-1]
#
#         stats = {m: [0, 0.0, 0.0] for m in todo}  # [n, mean, M2]
#         col_max = np.empty(L, dtype=np.float32)
#
#         for it in tqdm(range(ite), desc="ES null", disable=not verbose):
#             rng = np.random.default_rng(np.random.SeedSequence([seed, it]))
#             xi = rng.choice(N_pop, size=m_max, replace=False, shuffle=False)
#
#             col_max.fill(-np.inf)
#             sizes_iter = iter(todo)
#             next_m = next(sizes_iter)
#
#             for i in range(m_max):
#                 np.maximum(col_max, S_full[xi[i]], out=col_max)
#                 if i + 1 == next_m:
#                     es = float(_es_from_col_max(col_max))
#                     st = stats[next_m]
#                     st[0] += 1
#                     delta = es - st[1]
#                     st[1] += delta / st[0]
#                     st[2] += delta * (es - st[1])
#                     try:
#                         next_m = next(sizes_iter)
#                     except StopIteration:
#                         break
#
#         for m, (n, mean, M2) in stats.items():
#             std = float(np.sqrt(M2 / (n - 1))) if n > 1 else 0.0
#             self.cache[m] = (float(mean), std)
#
#         if verbose:
#             print(f"Cached {len(self.cache)} ES null distributions")
#
#     # ---- persistence --------------------------------------------------------
#
#     def save(self, path: str):
#         with open(path, "wb") as f:
#             pickle.dump(
#                 {"metadata": self.metadata, "cache": self.cache},
#                 f,
#                 protocol=pickle.HIGHEST_PROTOCOL,
#             )
#         print(f"Saved {len(self.cache)} ES cache entries to {path}")
#
#     @classmethod
#     def load(cls, path: str) -> "NullCacheESBetter":
#         obj = cls()
#         with open(path, "rb") as f:
#             payload = pickle.load(f)
#         if isinstance(payload, dict) and "cache" in payload:
#             obj.cache = payload["cache"]
#             obj.metadata = payload.get("metadata", {})
#         else:
#             obj.cache = payload
#             obj.metadata = {}
#         print(f"Loaded {len(obj.cache)} ES cache entries from {path}")
#         return obj
#
#     @staticmethod
#     def suggest_path(base_dir, E_unit, pop, ranked_emb):
#         """Content-addressed cache path — different inputs get distinct filenames."""
#         emb_h  = _hash_array(np.asarray(E_unit, dtype=np.float32))[:8]
#         pop_h  = _hash_array(np.asarray(pop, dtype=np.int32))[:8]
#         rank_h = _hash_array(np.asarray(ranked_emb, dtype=np.float32))[:8]
#         return os.path.join(base_dir, f"es_{emb_h}_{pop_h}_{rank_h}.pkl")
#
#     def __len__(self):
#         return len(self.cache)
#
#     def __contains__(self, m):
#         return int(m) in self.cache
#
#     def missing_sizes(self, gene_set_sizes):
#         return [m for m in gene_set_sizes if int(m) not in self.cache]
