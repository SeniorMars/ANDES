"""Exact best-match-average scoring and prefix-coupled BMA nulls.

For sets A and B, ANDES averages both directed best-match sums:

    (sum[a in A] max[b in B] sim(a, b)
     + sum[b in B] max[a in A] sim(a, b)) / (|A| + |B|)

The matrix scorer builds gene-to-term maxima in bounded chunks,
immediately aggregates each chunk into directed term scores, and never
materializes a full one-shot best-match matrix. Symmetric comparisons reuse
one direction exactly.

Similarity, directed aggregation, and exact-score output use float32. Null
means and variances use float64 Welford accumulation with ``ddof=1``.
Prefix-coupled sampling shares random prefixes across requested size pairs:
each marginal is a valid Monte Carlo estimate. Estimates across sizes are
correlated by construction. Numerical scoring is calibration-free; null models
apply z-scoring at the application boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Protocol, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse  # pyright: ignore[reportMissingTypeStubs]

from .data import PackedTermAxis

FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]
SparseIndexArray: TypeAlias = NDArray[np.int32] | NDArray[np.int64]
TermRange: TypeAlias = tuple[int, int]
NullSizePair: TypeAlias = tuple[int, int]
NullStatistic: TypeAlias = tuple[float, float]
NullStatistics: TypeAlias = dict[NullSizePair, NullStatistic]
ScoreMetadata: TypeAlias = dict[str, bool | float | int]


class _CsrFloatMatrix(Protocol):
    """The small CSR surface used by the numerical core."""

    @property
    def data(self) -> FloatArray: ...

    @property
    def indices(self) -> SparseIndexArray: ...

    @property
    def indptr(self) -> SparseIndexArray: ...

    def sort_indices(self) -> None: ...

    def __matmul__(self, other: FloatArray, /) -> FloatArray: ...


class _DirectionStats(TypedDict):
    source_genes: int
    chunk_workspace_mb: float
    logical_bestmatch_mb: float
    directed_matrix_mb: float
    fixed_direction_mb: float
    direction_peak_mb: float


def _int32_at(values: IntArray, position: int) -> int:
    """Read one int32 scalar without leaking NumPy's scalar Any type."""
    return int(cast(np.int32, values[position]))


def _int64_at(values: NDArray[np.int64], position: int) -> int:
    """Read one int64 scalar without leaking NumPy's scalar Any type."""
    return int(cast(np.int64, values[position]))


def _float64_at(values: NDArray[np.float64], row: int, column: int) -> float:
    """Read one float64 scalar without leaking NumPy's scalar Any type."""
    return float(cast(np.float64, values[row, column]))


def packed_terms_equal(left: PackedTermAxis, right: PackedTermAxis) -> bool:
    """Return whether two packed term axes are identical."""
    return (
        left.terms == right.terms
        and np.array_equal(left.sizes, right.sizes)
        and np.array_equal(left.members, right.members)
    )


def combine_directed_scores(
    directed12: ArrayLike,
    directed21: ArrayLike,
    sizes1: ArrayLike,
    sizes2: ArrayLike,
    out: FloatArray | None = None,
) -> FloatArray:
    """Combine two directed best-match sums into exact BMA scores."""
    directed12_values: FloatArray = np.asarray(directed12, dtype=np.float32)
    directed21_values: FloatArray = np.asarray(directed21, dtype=np.float32)
    sizes1_values: IntArray = np.asarray(sizes1, dtype=np.int32)
    sizes2_values: IntArray = np.asarray(sizes2, dtype=np.int32)
    expected12 = (sizes1_values.size, sizes2_values.size)
    expected21 = (sizes2_values.size, sizes1_values.size)
    if directed12_values.shape != expected12 or directed21_values.shape != expected21:
        observed = f"{directed12_values.shape}, {directed21_values.shape}"
        raise ValueError(f"directed score shapes do not match term sizes: {observed}")
    if out is None:
        out = np.empty(expected12, dtype=np.float32)
    elif out.shape != expected12:
        raise ValueError(f"out shape {out.shape} does not match {expected12}")

    sizes2_float: FloatArray = sizes2_values.astype(np.float32)
    for i in range(sizes1_values.size):
        output_row = out[i : i + 1, :]
        _ = np.add(
            directed12_values[i : i + 1, :],
            directed21_values[:, i : i + 1].T,
            out=output_row,
        )
        denominator = _int32_at(sizes1_values, i) + sizes2_float
        _ = np.divide(output_row, denominator[None, :], out=output_row)
    return out


def _bestmatch_chunk_bytes(
    lengths: IntArray,
    start: int,
    end: int,
    source_rows: int,
    embedding_dim: int,
    aggregate_rows: int = 0,
) -> int:
    occurrences = int(cast(np.int64, lengths[start:end].sum(dtype=np.int64)))
    terms = int(end - start)
    itemsize = np.dtype(np.float32).itemsize
    return itemsize * (
        occurrences * int(embedding_dim)
        + int(source_rows) * occurrences
        + int(source_rows) * terms
        + int(aggregate_rows) * terms
    )


def _ranges_with_caps(
    lengths: IntArray,
    occurrence_cap: int,
    term_cap: int,
) -> list[TermRange]:
    """Pack contiguous terms within occurrence and term-count caps."""
    ranges: list[TermRange] = []
    start = 0
    occurrences = 0
    for end in range(1, lengths.size + 1):
        length = _int32_at(lengths, end - 1)
        next_occurrences = occurrences + length
        next_terms = end - start
        if end - start > 1 and (
            next_occurrences > occurrence_cap or next_terms > term_cap
        ):
            ranges.append((start, end - 1))
            start = end - 1
            occurrences = length
        else:
            occurrences = next_occurrences
    ranges.append((start, len(lengths)))
    return ranges


def _reusable_workspace_bytes(
    lengths: IntArray,
    ranges: Sequence[TermRange],
    source_rows: int,
    embedding_dim: int,
    aggregate_rows: int,
) -> int:
    """Return the fixed buffers retained while one chunk is consumed."""
    max_occurrences = max(
        int(cast(np.int64, lengths[start:end].sum(dtype=np.int64)))
        for start, end in ranges
    )
    max_terms = max(end - start for start, end in ranges)
    itemsize = np.dtype(np.float32).itemsize
    return itemsize * (
        max_occurrences * (int(embedding_dim) + int(source_rows))
        + max_terms * (int(source_rows) + int(aggregate_rows))
    )


def bestmatch_term_ranges(
    lengths: ArrayLike,
    source_rows: int,
    embedding_dim: int,
    max_workspace_mb: float | None,
    aggregate_rows: int = 0,
) -> tuple[list[TermRange], float]:
    """Plan target-term chunks against the temporary-array budget."""
    length_values: IntArray = np.asarray(lengths, dtype=np.int32)
    n_terms = int(length_values.size)
    if n_terms == 0:
        return [], 0.0

    budget_bytes = (
        None
        if max_workspace_mb is None or max_workspace_mb <= 0
        else float(max_workspace_mb) * 1e6
    )
    if budget_bytes is None:
        ranges = [(0, n_terms)]
    else:
        ranges: list[TermRange] = []
        start = 0
        for end in range(1, n_terms + 1):
            candidate = _bestmatch_chunk_bytes(
                length_values,
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

    peak_bytes = _reusable_workspace_bytes(
        length_values,
        ranges,
        source_rows,
        embedding_dim,
        aggregate_rows,
    )
    if budget_bytes is not None and peak_bytes > budget_bytes:
        occurrence_bytes = np.dtype(np.float32).itemsize * (
            int(embedding_dim) + int(source_rows)
        )
        term_bytes = np.dtype(np.float32).itemsize * (
            int(source_rows) + int(aggregate_rows)
        )
        max_length = int(cast(np.int32, length_values.max()))
        minimum_bytes = max_length * occurrence_bytes + term_bytes

        if minimum_bytes >= budget_bytes:
            occurrence_cap = max_length
            term_cap = 1
        else:
            current_occurrences = max(
                int(cast(np.int64, length_values[start:end].sum(dtype=np.int64)))
                for start, end in ranges
            )
            current_terms = max(end - start for start, end in ranges)
            remaining_bytes = budget_bytes - minimum_bytes
            extra_bytes = (current_occurrences - max_length) * occurrence_bytes + (
                current_terms - 1
            ) * term_bytes
            scale = min(1.0, remaining_bytes / max(1, extra_bytes))
            occurrence_cap = max_length + int(
                (current_occurrences - max_length) * scale
            )
            term_cap = 1 + int((current_terms - 1) * scale)

        ranges = _ranges_with_caps(
            length_values,
            occurrence_cap,
            term_cap,
        )
        peak_bytes = _reusable_workspace_bytes(
            length_values,
            ranges,
            source_rows,
            embedding_dim,
            aggregate_rows,
        )
    return ranges, peak_bytes / 1e6


def build_packed_membership_matrix(
    packed: PackedTermAxis,
    n_genes: int,
    gene_universe: ArrayLike | None = None,
) -> _CsrFloatMatrix:
    """Build a CSR term-by-gene membership matrix."""
    n_genes = int(n_genes)
    universe: IntArray | None
    if gene_universe is None:
        universe = None
    else:
        universe = np.asarray(gene_universe, dtype=np.int32)
        if universe.ndim != 1:
            raise ValueError("gene_universe must be one-dimensional")
        if universe.size and (universe.min() < 0 or universe.max() >= n_genes):
            raise IndexError("gene_universe contains an out-of-range gene index")
        if universe.size > 1 and np.any(universe[1:] <= universe[:-1]):
            raise ValueError("gene_universe must be sorted and unique")

    n_terms = len(packed.terms)
    if n_terms == 0:
        column_count = n_genes if universe is None else int(universe.size)
        return cast(
            _CsrFloatMatrix,
            cast(object, sparse.csr_matrix((0, column_count), dtype=np.float32)),
        )
    if packed.members.size and (
        packed.members.min() < 0 or packed.members.max() >= n_genes
    ):
        raise IndexError("term membership contains an out-of-range gene index")

    rows = np.repeat(
        np.arange(n_terms, dtype=np.int32),
        packed.sizes.astype(np.int64),
    )
    if universe is None:
        columns = packed.members
        n_columns = n_genes
    else:
        global_to_local = np.full(n_genes, -1, dtype=np.int32)
        global_to_local[universe] = np.arange(
            universe.size,
            dtype=np.int32,
        )
        columns = global_to_local[packed.members]
        if np.any(columns < 0):
            raise ValueError("gene_universe does not contain every term member")
        n_columns = universe.size

    values = np.ones(columns.size, dtype=np.float32)
    return cast(
        _CsrFloatMatrix,
        cast(
            object,
            sparse.csr_matrix(
                (values, (rows, columns)),
                shape=(n_terms, n_columns),
                dtype=np.float32,
            ),
        ),
    )


def _iter_gene_to_term_best_match_chunks(
    source_embeddings: ArrayLike,
    all_embeddings: ArrayLike,
    packed: PackedTermAxis,
    max_workspace_mb: float | None = 128,
    aggregate_rows: int = 0,
) -> Iterator[tuple[int, int, FloatArray]]:
    """Yield exact gene-to-term best-match chunks from reusable storage.

    A yielded array is valid until the iterator advances. Consumers must copy
    it if they need to retain a chunk after requesting the next one.
    """
    source_values: FloatArray = np.ascontiguousarray(
        source_embeddings,
        dtype=np.float32,
    )
    all_values: FloatArray = np.asarray(all_embeddings, dtype=np.float32)
    if source_values.ndim != 2 or all_values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if source_values.shape[1] != all_values.shape[1]:
        raise ValueError("source and target embedding dimensions differ")
    if packed.members.size and (
        packed.members.min() < 0 or packed.members.max() >= all_values.shape[0]
    ):
        raise IndexError("packed target contains an out-of-range gene index")

    ranges, _ = bestmatch_term_ranges(
        packed.sizes,
        source_values.shape[0],
        source_values.shape[1],
        max_workspace_mb,
        aggregate_rows=aggregate_rows,
    )
    if not ranges:
        return

    max_occurrences = max(
        int(cast(np.int64, packed.offsets[end] - packed.offsets[start]))
        for start, end in ranges
    )
    max_terms = max(end - start for start, end in ranges)
    embedding_dim = source_values.shape[1]
    source_rows = source_values.shape[0]
    target_storage = np.empty(
        max_occurrences * embedding_dim,
        dtype=np.float32,
    )
    similarity_storage = np.empty(
        source_rows * max_occurrences,
        dtype=np.float32,
    )
    best_storage = np.empty(
        source_rows * max_terms,
        dtype=np.float32,
    )

    for start, end in ranges:
        flat_start = _int64_at(packed.offsets, start)
        flat_end = _int64_at(packed.offsets, end)
        target_indices = packed.members[flat_start:flat_end]
        occurrences = flat_end - flat_start
        terms = end - start
        target_embeddings = target_storage[: occurrences * embedding_dim].reshape(
            occurrences, embedding_dim
        )
        _ = np.take(
            all_values,
            target_indices,
            axis=0,
            out=target_embeddings,
        )
        similarities = similarity_storage[: source_rows * occurrences].reshape(
            source_rows, occurrences
        )
        _ = np.matmul(
            source_values,
            target_embeddings.T,
            out=similarities,
        )
        local_offsets = packed.offsets[start:end] - flat_start
        best = best_storage[: source_rows * terms].reshape(source_rows, terms)
        _ = np.maximum.reduceat(
            similarities,
            local_offsets,
            axis=1,
            out=best,
        )
        yield start, end, best


def gene_to_term_best_match_matrix(
    embeddings: ArrayLike,
    packed: PackedTermAxis,
    max_workspace_mb: float | None = 128,
) -> tuple[FloatArray, float]:
    """Materialize the reusable ``B[gene, term]`` matrix."""
    embedding_values: FloatArray = np.asarray(embeddings, dtype=np.float32)
    if embedding_values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    bestmatch = np.empty(
        (embedding_values.shape[0], len(packed.terms)),
        dtype=np.float32,
    )
    for start, end, chunk in _iter_gene_to_term_best_match_chunks(
        embedding_values,
        embedding_values,
        packed,
        max_workspace_mb=max_workspace_mb,
    ):
        bestmatch[:, start:end] = chunk
    _, workspace_mb = bestmatch_term_ranges(
        packed.sizes,
        embedding_values.shape[0],
        embedding_values.shape[1],
        max_workspace_mb,
    )
    return bestmatch, workspace_mb


def score_bma_matrix_bestmatch(
    embeddings: ArrayLike,
    packed1: PackedTermAxis,
    packed2: PackedTermAxis,
    *,
    symmetric: bool = False,
    max_workspace_mb: float | None = 128,
) -> tuple[FloatArray, ScoreMetadata]:
    """Compute exact all-vs-all BMA with streamed best-match chunks."""
    embedding_values: FloatArray = np.asarray(embeddings, dtype=np.float32)
    if embedding_values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    n_genes = embedding_values.shape[0]
    same_input_representation = bool(symmetric) and packed1 is packed2
    if same_input_representation:
        packed2 = packed1
        symmetric_reuse = True
    else:
        symmetric_reuse = bool(symmetric) and packed_terms_equal(
            packed1,
            packed2,
        )
        packed2 = packed1 if symmetric_reuse else packed2

    def directed_scores(
        source: PackedTermAxis,
        target: PackedTermAxis,
    ) -> tuple[FloatArray, _DirectionStats]:
        source_genes = np.unique(source.members)
        membership = build_packed_membership_matrix(
            source,
            n_genes,
            gene_universe=source_genes,
        )
        source_embeddings = np.ascontiguousarray(
            embedding_values[source_genes],
            dtype=np.float32,
        )
        directed = np.empty(
            (len(source.terms), len(target.terms)),
            dtype=np.float32,
        )
        for start, end, chunk in _iter_gene_to_term_best_match_chunks(
            source_embeddings,
            embedding_values,
            target,
            max_workspace_mb=max_workspace_mb,
            aggregate_rows=len(source.terms),
        ):
            directed[:, start:end] = membership @ chunk

        _, chunk_workspace_mb = bestmatch_term_ranges(
            target.sizes,
            source_genes.size,
            embedding_values.shape[1],
            max_workspace_mb,
            aggregate_rows=len(source.terms),
        )
        membership_mb = (
            membership.data.nbytes
            + membership.indices.nbytes
            + membership.indptr.nbytes
        ) / 1e6
        source_embedding_mb = source_embeddings.nbytes / 1e6
        directed_mb = directed.nbytes / 1e6
        fixed_mb = source_genes.nbytes / 1e6 + membership_mb + source_embedding_mb
        return directed, {
            "source_genes": int(source_genes.size),
            "chunk_workspace_mb": float(chunk_workspace_mb),
            "logical_bestmatch_mb": (source_genes.size * len(target.terms) * 4 / 1e6),
            "directed_matrix_mb": float(directed_mb),
            "fixed_direction_mb": float(fixed_mb),
            "direction_peak_mb": float(fixed_mb + directed_mb + chunk_workspace_mb),
        }

    if symmetric_reuse:
        directed, direction = directed_scores(packed1, packed1)
        scores = combine_directed_scores(
            directed,
            directed,
            packed1.sizes,
            packed1.sizes,
        )
        output_mb = scores.nbytes / 1e6
        peak_mb = max(
            direction["direction_peak_mb"],
            direction["directed_matrix_mb"] + output_mb,
        )
        return scores, {
            "workspace_mb": float(peak_mb),
            "estimated_peak_mb": float(peak_mb),
            "chunk_workspace_mb": direction["chunk_workspace_mb"],
            "directed_matrix_mb": direction["directed_matrix_mb"],
            "output_matrix_mb": float(output_mb),
            "fixed_direction_mb": direction["fixed_direction_mb"],
            "bestmatch1_mb": direction["logical_bestmatch_mb"],
            "bestmatch2_mb": direction["logical_bestmatch_mb"],
            "bestmatch_materialized_mb": 0.0,
            "source_genes1": direction["source_genes"],
            "source_genes2": direction["source_genes"],
            "membership_occurrences1": int(packed1.members.size),
            "membership_occurrences2": int(packed1.members.size),
            "symmetric_reuse": True,
        }

    directed12, direction12 = directed_scores(packed1, packed2)
    directed21, direction21 = directed_scores(packed2, packed1)
    scores = combine_directed_scores(
        directed12,
        directed21,
        packed1.sizes,
        packed2.sizes,
    )
    directed_mb = directed12.nbytes / 1e6
    output_mb = scores.nbytes / 1e6
    peak_mb = max(
        direction12["direction_peak_mb"],
        directed_mb + direction21["direction_peak_mb"],
        2.0 * directed_mb + output_mb,
    )
    return scores, {
        "workspace_mb": float(peak_mb),
        "estimated_peak_mb": float(peak_mb),
        "chunk_workspace_mb": max(
            direction12["chunk_workspace_mb"],
            direction21["chunk_workspace_mb"],
        ),
        "directed_matrix_mb": float(directed_mb),
        "output_matrix_mb": float(output_mb),
        "fixed_direction_mb": max(
            direction12["fixed_direction_mb"],
            direction21["fixed_direction_mb"],
        ),
        "bestmatch1_mb": direction21["logical_bestmatch_mb"],
        "bestmatch2_mb": direction12["logical_bestmatch_mb"],
        "bestmatch_materialized_mb": 0.0,
        "source_genes1": direction12["source_genes"],
        "source_genes2": direction21["source_genes"],
        "membership_occurrences1": int(packed1.members.size),
        "membership_occurrences2": int(packed2.members.size),
        "symmetric_reuse": False,
    }


def build_prefix_null(
    embeddings: ArrayLike,
    population1: ArrayLike,
    size_pairs: Iterable[NullSizePair],
    *,
    iterations: int = 1000,
    seed: int = 12345,
    population2: ArrayLike | None = None,
    existing: Mapping[NullSizePair, NullStatistic] | None = None,
) -> NullStatistics:
    """Return prefix-coupled BMA null statistics keyed by size pair.

    The null resolver validates ``existing`` before passing it here.
    """
    iterations = int(iterations)
    if iterations < 2:
        raise ValueError("iterations must be at least two when ddof=1")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    embedding_values: FloatArray = np.asarray(embeddings, dtype=np.float32)
    population1_values: IntArray = np.asarray(population1, dtype=np.int32)
    population2_values: IntArray = np.asarray(
        population1_values if population2 is None else population2,
        dtype=np.int32,
    )
    values: NullStatistics = dict(existing or {})
    pairs = sorted({(int(m), int(k)) for m, k in size_pairs})
    if not pairs or any(m < 1 or k < 1 for m, k in pairs):
        raise ValueError("null sizes must contain positive size pairs")
    missing = [pair for pair in pairs if pair not in values]
    if not missing:
        return values
    max_m = max(m for m, _ in missing)
    max_k = max(k for _, k in missing)
    if max_m > len(population1_values) or max_k > len(population2_values):
        raise ValueError("requested null size exceeds background population")

    rng = np.random.default_rng(seed)
    means: NDArray[np.float64] = np.zeros((max_m, max_k), dtype=np.float64)
    second_moments: NDArray[np.float64] = np.zeros(
        (max_m, max_k),
        dtype=np.float64,
    )
    denominator: NDArray[np.float64] = (
        np.arange(1, max_m + 1, dtype=np.float64)[:, None]
        + np.arange(1, max_k + 1, dtype=np.float64)[None, :]
    )

    for iteration in range(iterations):
        left = population1_values[rng.permutation(len(population1_values))[:max_m]]
        right = population2_values[rng.permutation(len(population2_values))[:max_k]]
        similarities: FloatArray = embedding_values[left] @ embedding_values[right].T
        row_sums: NDArray[np.float64] = np.cumsum(
            np.maximum.accumulate(similarities, axis=1),
            axis=0,
            dtype=np.float64,
        )
        column_sums: NDArray[np.float64] = np.cumsum(
            np.maximum.accumulate(similarities, axis=0),
            axis=1,
            dtype=np.float64,
        )
        scores: NDArray[np.float64] = (row_sums + column_sums) / denominator
        delta: NDArray[np.float64] = scores - means
        means += delta / (iteration + 1)
        second_moments += delta * (scores - means)

    stds: NDArray[np.float64] = (
        np.sqrt(second_moments / (iterations - 1))
        if iterations > 1
        else np.zeros_like(second_moments)
    )
    for m, k in missing:
        values[(m, k)] = (
            _float64_at(means, m - 1, k - 1),
            _float64_at(stds, m - 1, k - 1),
        )
    return values
