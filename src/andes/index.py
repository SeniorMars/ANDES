"""
Persistent exact ANDES database indexes and index command-line operations.

Builds a reusable index for one embedding plus one GMT database:

  B[g, t] = max_{x in term_t} sim(g, x)
  M[t, g] = 1 if gene g belongs to term t

Then one query gene set Q can be scored exactly against every indexed term:

  query_to_terms = B[Q, :].sum(axis=0)
  best_to_query = max_{q in Q} sim(g, q) for every gene g
  terms_to_query = M @ best_to_query
  true = (query_to_terms + terms_to_query) / (|Q| + term_sizes)

Artifact construction and numerical consumers live here. Command-line parsing
and result serialization live in ``andes.index_cli``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from . import artifacts
from . import bma as func
from . import data as ld
from .nulls import BmaNullModel
from .scoring import ScoreResult, ScoreStats

INDEX_VERSION = 5
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]

_INDEX_FILES = (
    "metadata.json",
    "genes.json",
    "terms.json",
    "embedding.npy",
    "bestmatch.npy",
    "sizes.npy",
    "background.npy",
    "membership.npz",
)

_METADATA_KEYS = {
    "index_version",
    "n_genes",
    "embedding_dim",
    "n_terms",
    "embedding_hash",
    "gene_list_hash",
    "terms_hash",
    "sizes_hash",
    "members_hash",
    "background_hash",
    "membership_hash",
    "bestmatch_hash",
    "embedding_fingerprint",
    "database_fingerprint",
    "background_policy",
    "runtime",
}


def _invalid_index(index_dir, message):
    return ValueError(f"invalid ANDES index at {Path(index_dir)}: {message}")


def _close_memmap(value: NDArray[np.generic]) -> None:
    """Flush and close a NumPy memory map when one backs ``value``."""
    if isinstance(value, np.memmap):
        value.flush()
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _validated_int_indices(name, value, upper_bound, allow_empty=False):
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index array")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must contain at least one index")
    # Validate in the source integer width. Casting first could wrap a large
    # uint64/int64 value into an apparently valid int32 embedding row.
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= int(upper_bound)):
        raise ValueError(f"{name} contains an index outside [0, {int(upper_bound)})")
    return arr.astype(np.int32, copy=False)


def _canonical_int_indices(name, value, upper_bound, allow_empty=False):
    validated = _validated_int_indices(
        name,
        value,
        upper_bound,
        allow_empty=allow_empty,
    )
    return np.unique(validated)


def _ordered_unique_int_indices(name, value, upper_bound):
    validated = _validated_int_indices(
        name,
        value,
        upper_bound,
        allow_empty=False,
    )
    if np.unique(validated).size != validated.size:
        raise ValueError(f"{name} must not contain duplicate positions")
    return validated


def _validate_null_model_axes(null_model, index1, index2):
    """Require a BMA null model with the exact index-axis identity."""
    if not isinstance(null_model, BmaNullModel):
        raise TypeError("null_model must be a BmaNullModel")
    expected = {
        "embedding_hash": index1.metadata["embedding_hash"],
        "population1_hash": artifacts.hash_array(
            np.asarray(index1.background, dtype=np.int32)
        ),
        "population2_hash": artifacts.hash_array(
            np.asarray(index2.background, dtype=np.int32)
        ),
    }
    actual = {
        "embedding_hash": null_model.spec.embedding_hash,
        "population1_hash": null_model.spec.population_hashes[0],
        "population2_hash": null_model.spec.population_hashes[1],
    }
    labels = {
        "embedding_hash": "embedding",
        "population1_hash": "row/background-1",
        "population2_hash": "column/background-2",
    }
    for key, wanted in expected.items():
        if actual[key] != wanted:
            raise ValueError(f"null model {labels[key]} is incompatible")
    return null_model


def _build_andes_index_payload(
    embedding: ld.EmbeddingSpace,
    database: ld.GeneSetDatabase,
    out_dir: str | Path,
    max_workspace_mb: float = 128,
    runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write one complete index payload into an existing private directory."""
    payload_started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(embedding, ld.EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    if not isinstance(database, ld.GeneSetDatabase):
        raise TypeError("database must be a GeneSetDatabase")
    if database.embedding_fingerprint != embedding.fingerprint:
        raise ValueError("database was built for a different embedding or gene order")

    E_unit = embedding.vectors
    gene_list = embedding.genes
    terms = database.terms
    background = database.background
    packed = database.packed_axis
    membership = func.build_packed_membership_matrix(packed, E_unit.shape[0])
    membership.sort_indices()
    sizes = np.asarray(database.sizes, dtype=np.int32)

    np.save(out_dir / "embedding.npy", E_unit)
    np.save(out_dir / "sizes.npy", sizes)
    np.save(out_dir / "background.npy", background)
    sparse.save_npz(out_dir / "membership.npz", membership)
    with open(out_dir / "genes.json", "w") as fh:
        json.dump(list(gene_list), fh)
    with open(out_dir / "terms.json", "w") as fh:
        json.dump(terms, fh)

    _, planned_chunk_workspace_mb = func.bestmatch_term_ranges(
        packed.sizes,
        E_unit.shape[0],
        E_unit.shape[1],
        max_workspace_mb,
    )
    bestmatch_path = out_dir / "bestmatch.npy"
    B = np.lib.format.open_memmap(
        bestmatch_path,
        mode="w+",
        dtype=np.float32,
        shape=(E_unit.shape[0], len(terms)),
    )
    bestmatch_started = time.perf_counter()
    next_start = 0
    peak_chunk_mb = 0.0
    try:
        for start, end, B_chunk in func._iter_gene_to_term_best_match_chunks(
            E_unit,
            E_unit,
            packed,
            max_workspace_mb=max_workspace_mb,
        ):
            start = int(start)
            end = int(end)
            B_chunk = np.asarray(B_chunk)
            if start != next_start or end <= start or end > len(terms):
                raise RuntimeError(
                    "best-match chunk iterator yielded a non-contiguous term range "
                    f"({start}, {end}) after column {next_start}"
                )
            expected_shape = (E_unit.shape[0], end - start)
            if B_chunk.shape != expected_shape or B_chunk.dtype != np.float32:
                raise RuntimeError(
                    "best-match chunk iterator yielded "
                    f"shape={B_chunk.shape}, dtype={B_chunk.dtype}; "
                    f"expected shape={expected_shape}, dtype=float32"
                )
            B[:, start:end] = B_chunk
            peak_chunk_mb = max(peak_chunk_mb, B_chunk.nbytes / 1e6)
            next_start = end
        if next_start != len(terms):
            raise RuntimeError(
                f"best-match chunk iterator stopped at column "
                f"{next_start} of {len(terms)}"
            )
        B.flush()
        bestmatch_construction_seconds = time.perf_counter() - bestmatch_started
        bestmatch_hash = artifacts.hash_array(B)
        bestmatch_mb = B.nbytes / 1e6
    finally:
        _close_memmap(B)

    metadata: dict[str, object] = {
        "index_version": INDEX_VERSION,
        "n_genes": int(E_unit.shape[0]),
        "embedding_dim": int(E_unit.shape[1]),
        "n_terms": len(terms),
        "embedding_hash": artifacts.hash_array(E_unit),
        "gene_list_hash": artifacts.hash_strings(gene_list),
        "terms_hash": artifacts.hash_strings(terms),
        "sizes_hash": artifacts.hash_array(sizes),
        "members_hash": artifacts.hash_array(database.members),
        "background_hash": artifacts.hash_array(np.sort(background)),
        "membership_hash": artifacts.hash_sparse_csr(membership),
        "bestmatch_hash": bestmatch_hash,
        "embedding_fingerprint": embedding.fingerprint,
        "database_fingerprint": database.fingerprint,
        "background_policy": database.background_policy,
        "workspace_mb": float(planned_chunk_workspace_mb),
        "chunk_workspace_mb": float(planned_chunk_workspace_mb),
        "bestmatch_write_chunk_mb": float(peak_chunk_mb),
        "workspace_limit_mb": float(max_workspace_mb),
        "bestmatch_mb": float(bestmatch_mb),
        "bestmatch_construction_seconds": bestmatch_construction_seconds,
        "payload_build_seconds": time.perf_counter() - payload_started,
        "runtime": dict(runtime or {}),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return metadata


def build_andes_index(
    embedding: ld.EmbeddingSpace,
    database: ld.GeneSetDatabase,
    out_dir: str | Path,
    max_workspace_mb: float = 128,
    overwrite: bool = False,
    runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build, validate, and atomically publish an exact ANDES index."""
    final_dir = Path(out_dir)
    with (
        artifacts.artifact_lock(final_dir),
        artifacts.atomic_artifact_directory(
            final_dir,
            overwrite=bool(overwrite),
        ) as building_dir,
    ):
        metadata = _build_andes_index_payload(
            embedding,
            database,
            building_dir,
            max_workspace_mb=max_workspace_mb,
            runtime=runtime,
        )
        # Validate the private artifact with the same complete contract
        # used by consumers before making the final directory visible.
        audited = _load_andes_index_unlocked(
            building_dir,
            mmap=True,
            verify="full",
        )
        audited.close()
    return metadata


@dataclass(frozen=True, slots=True)
class AndesIndex:
    index_dir: Path
    E_unit: FloatArray
    gene_list: tuple[str, ...]
    gene_to_index: Mapping[str, int]
    terms: tuple[str, ...]
    term_to_index: Mapping[str, int]
    sizes: IntArray
    background: IntArray
    membership: sparse.csr_matrix
    bestmatch: FloatArray
    metadata: Mapping[str, object]

    def close(self) -> None:
        """Close memory-mapped payloads when this index is no longer in use."""
        _close_memmap(self.E_unit)
        _close_memmap(self.bestmatch)

    @property
    def embedding_fingerprint(self) -> str:
        return str(self.metadata["embedding_fingerprint"])

    @property
    def database_fingerprint(self) -> str:
        return str(self.metadata["database_fingerprint"])

    @property
    def background_policy(self) -> str:
        return str(self.metadata["background_policy"])

    def embedding_space(self) -> ld.EmbeddingSpace:
        """Return a zero-copy canonical embedding view from validated metadata."""
        return ld.EmbeddingSpace(
            vectors=self.E_unit,
            genes=self.gene_list,
            gene_to_index=self.gene_to_index,
            vector_hash=str(self.metadata["embedding_hash"]),
            gene_order_hash=str(self.metadata["gene_list_hash"]),
            fingerprint=self.embedding_fingerprint,
        )

    def gene_set_database(self) -> ld.GeneSetDatabase:
        """Return a canonical view sharing large membership and size arrays."""
        members = cast(IntArray, self.membership.indices)
        offsets = np.asarray(self.membership.indptr, dtype=np.int64)
        offsets.setflags(write=False)
        return ld.GeneSetDatabase(
            n_genes=len(self.gene_list),
            embedding_fingerprint=self.embedding_fingerprint,
            terms=self.terms,
            members=members,
            offsets=offsets,
            sizes=self.sizes,
            background=self.background,
            background_policy=str(self.metadata["background_policy"]),
            term_to_index=self.term_to_index,
            background_hash=str(self.metadata["background_hash"]),
            fingerprint=self.database_fingerprint,
        )

    def members_at(self, position: int) -> IntArray:
        """Return the read-only member indices for one term position."""
        if isinstance(position, (bool, np.bool_)) or not isinstance(
            position, (int, np.integer)
        ):
            raise TypeError("term position must be an integer")
        position = int(position)
        if position < 0 or position >= len(self.terms):
            raise IndexError(f"term position {position} is out of range")
        start = int(self.membership.indptr[position])
        end = int(self.membership.indptr[position + 1])
        return cast(IntArray, self.membership.indices[start:end])

    def members_for(self, term: str) -> IntArray:
        """Return the read-only member indices for one term identifier."""
        if not isinstance(term, str):
            raise TypeError("term must be a string identifier")
        try:
            position = self.term_to_index[term]
        except KeyError as exc:
            raise KeyError(f"term {term!r} is not present in the index") from exc
        return self.members_at(position)

    def ranked_bestmatch_artifact(self):
        """Return the persistent matrix with its complete scoring identity."""
        from .scoring import IndexedBestMatch

        return IndexedBestMatch(
            values=self.bestmatch,
            embedding_fingerprint=self.embedding_fingerprint,
            database_fingerprint=self.database_fingerprint,
        )

    def map_genes(self, genes: Iterable[str]) -> tuple[IntArray, list[str]]:
        idx: list[int] = []
        missing: list[str] = []
        for gene in genes:
            if gene in self.gene_to_index:
                idx.append(self.gene_to_index[gene])
            else:
                missing.append(gene)
        if not idx:
            raise ValueError("query has no genes present in the indexed embedding")
        return np.asarray(sorted(set(idx)), dtype=np.int32), missing

    def _canonical_query_indices(self, query_idx: ArrayLike) -> IntArray:
        return _canonical_int_indices(
            "query_idx", query_idx, self.E_unit.shape[0], allow_empty=False
        )

    def validate_query_background(self, query_idx: ArrayLike) -> None:
        """Raise if query genes cannot be normalized by the index null cache."""
        query_idx = self._canonical_query_indices(query_idx)
        missing = np.setdiff1d(query_idx, self.background, assume_unique=False)
        if missing.size:
            genes = [self.gene_list[int(i)] for i in missing[:5]]
            extra = "" if missing.size <= 5 else f" and {missing.size - 5} more"
            raise ValueError(
                "z-score queries must use genes from the indexed background; "
                f"outside-background genes: {', '.join(genes)}{extra}"
            )

    def assert_compatible_with(self, other: AndesIndex) -> None:
        """Require the same normalized embedding and exact gene row order."""
        if not isinstance(other, AndesIndex):
            raise TypeError("other must be an AndesIndex")
        if self.gene_list != other.gene_list:
            raise ValueError(
                "index gene order mismatch; indexes must use identical genes "
                "in identical row order"
            )
        if self.metadata["gene_list_hash"] != other.metadata["gene_list_hash"]:
            raise ValueError("index gene-order hashes do not match")
        if self.E_unit.shape != other.E_unit.shape:
            raise ValueError(
                "index embedding shape mismatch: "
                f"{self.E_unit.shape} != {other.E_unit.shape}"
            )
        if self.metadata["embedding_hash"] != other.metadata["embedding_hash"]:
            raise ValueError(
                "index embedding mismatch; indexes must be built from the "
                "same normalized embedding"
            )

    def _query_chunk_size(
        self,
        query_size,
        max_workspace_mb,
        *,
        include_similarity,
    ):
        """Plan a bounded query-gene chunk for the largest temporary matrix."""
        query_size = int(query_size)
        if max_workspace_mb is None or float(max_workspace_mb) <= 0.0:
            return query_size

        float_bytes = np.dtype(np.float32).itemsize
        n_genes, dimensions = self.E_unit.shape
        n_terms = len(self.terms)
        budget_bytes = max(float_bytes, int(float(max_workspace_mb) * 1e6))
        if include_similarity:
            fixed_bytes = (2 * n_genes + 3 * n_terms) * float_bytes
            bytes_per_query_gene = max(n_terms, n_genes + dimensions) * float_bytes
        else:
            fixed_bytes = 4 * n_terms * float_bytes
            bytes_per_query_gene = n_terms * float_bytes
        available = max(0, budget_bytes - fixed_bytes)
        return min(query_size, max(1, available // max(1, bytes_per_query_gene)))

    def _sum_bestmatch_rows(self, query_idx, chunk_size):
        """Sum indexed query-to-term affinities without a full query slice."""
        totals = np.zeros(len(self.terms), dtype=np.float64)
        for start in range(0, query_idx.size, chunk_size):
            rows = query_idx[start : start + chunk_size]
            totals += self.bestmatch[rows, :].sum(axis=0, dtype=np.float64)
        return totals.astype(np.float32)

    def _best_to_query(self, query_idx, chunk_size):
        """Compute gene-to-query maxima while bounding the similarity matrix."""
        maxima = np.full(self.E_unit.shape[0], -np.inf, dtype=np.float32)
        for start in range(0, query_idx.size, chunk_size):
            rows = query_idx[start : start + chunk_size]
            query_vectors = np.ascontiguousarray(
                self.E_unit[rows],
                dtype=np.float32,
            )
            similarities = self.E_unit @ query_vectors.T
            chunk_maxima = similarities.max(axis=1)
            np.maximum(maxima, chunk_maxima, out=maxima)
        return maxima

    def score_query(
        self,
        query_idx: ArrayLike,
        max_workspace_mb: float | None = 128,
    ) -> ScoreResult:
        """Return exact scores for one query and measured planning metadata."""
        query_idx = self._canonical_query_indices(query_idx)
        chunk_size = self._query_chunk_size(
            query_idx.size,
            max_workspace_mb,
            include_similarity=True,
        )
        query_to_terms = self._sum_bestmatch_rows(query_idx, chunk_size)
        best_to_query = self._best_to_query(query_idx, chunk_size)
        terms_to_query = np.asarray(self.membership @ best_to_query, dtype=np.float32)

        true_scores = (query_to_terms + terms_to_query) / (
            float(query_idx.size) + self.sizes.astype(np.float32)
        )
        n_genes, dimensions = self.E_unit.shape
        n_terms = len(self.terms)
        float_bytes = np.dtype(np.float32).itemsize
        fixed_bytes = (2 * n_genes + 3 * n_terms) * float_bytes
        dynamic_bytes = chunk_size * max(n_terms, n_genes + dimensions) * float_bytes
        requested_bytes = (
            0
            if max_workspace_mb is None or float(max_workspace_mb) <= 0.0
            else int(float(max_workspace_mb) * 1e6)
        )
        return ScoreResult(
            scores=np.asarray(true_scores, dtype=np.float32),
            stats=ScoreStats.create(
                "indexed_query",
                workspace_bytes=fixed_bytes + dynamic_bytes,
                details={
                    "query_size": int(query_idx.size),
                    "query_chunk_size": int(chunk_size),
                    "requested_workspace_bytes": requested_bytes,
                },
            ),
        )

    def score_indexed_term(
        self,
        term: str | int,
        max_workspace_mb: float | None = 128,
    ) -> ScoreResult:
        """Score one indexed term using its already-persisted best-match column."""
        if isinstance(term, (bool, np.bool_)):
            raise TypeError("term must be a string identifier or integer position")
        if isinstance(term, (int, np.integer)):
            term_pos = int(term)
            if term_pos < 0 or term_pos >= len(self.terms):
                raise IndexError(f"indexed term position {term_pos} is out of range")
            term_name = self.terms[term_pos]
        else:
            if not isinstance(term, str):
                raise TypeError("term must be a string identifier or integer position")
            term_name = term
            try:
                term_pos = self.term_to_index[term_name]
            except KeyError as exc:
                raise KeyError(
                    f"term {term_name!r} is not present in the index"
                ) from exc

        query_idx = self.members_at(term_pos)
        chunk_size = self._query_chunk_size(
            query_idx.size,
            max_workspace_mb,
            include_similarity=False,
        )
        query_to_terms = self._sum_bestmatch_rows(query_idx, chunk_size)
        best_to_query = np.asarray(self.bestmatch[:, term_pos], dtype=np.float32)
        terms_to_query = np.asarray(self.membership @ best_to_query, dtype=np.float32)
        true_scores = (query_to_terms + terms_to_query) / (
            float(query_idx.size) + self.sizes.astype(np.float32)
        )
        true_scores = np.asarray(true_scores, dtype=np.float32)
        n_terms = len(self.terms)
        float_bytes = np.dtype(np.float32).itemsize
        fixed_bytes = 4 * n_terms * float_bytes
        dynamic_bytes = chunk_size * n_terms * float_bytes
        requested_bytes = (
            0
            if max_workspace_mb is None or float(max_workspace_mb) <= 0.0
            else int(float(max_workspace_mb) * 1e6)
        )
        return ScoreResult(
            scores=true_scores,
            stats=ScoreStats.create(
                "indexed_term",
                workspace_bytes=fixed_bytes + dynamic_bytes,
                details={
                    "query_size": int(query_idx.size),
                    "query_chunk_size": int(chunk_size),
                    "term": term_name,
                    "term_position": int(term_pos),
                    "requested_workspace_bytes": requested_bytes,
                },
            ),
        )

    def score_queries(
        self,
        query_sets: Iterable[ArrayLike] | Mapping[object, ArrayLike],
        max_workspace_mb: float | None = 128,
    ) -> ScoreResult:
        """Score many query sets with stable forward sums and shared reverse GEMMs."""
        if isinstance(query_sets, Mapping):
            query_sets = list(query_sets.values())
        else:
            query_sets = list(query_sets)
        if not query_sets:
            empty = np.empty((0, len(self.terms)), dtype=np.float32)
            return ScoreResult(
                scores=empty,
                stats=ScoreStats.create(
                    "indexed_batch",
                    details={
                        "query_sizes": (),
                        "query_batch_ranges": (),
                        "requested_workspace_bytes": 0,
                    },
                ),
            )

        queries = [self._canonical_query_indices(query) for query in query_sets]
        query_sizes = np.asarray([query.size for query in queries], dtype=np.int32)
        float_bytes = np.dtype(np.float32).itemsize
        double_bytes = np.dtype(np.float64).itemsize
        retained_index_bytes = int(sum(query.nbytes for query in queries))
        query_to_terms = np.empty(
            (len(queries), len(self.terms)),
            dtype=np.float32,
        )
        matrix_bytes = int(query_to_terms.nbytes)
        peak_bytes = matrix_bytes + retained_index_bytes
        for position, query in enumerate(queries):
            chunk_size = self._query_chunk_size(
                query.size,
                max_workspace_mb,
                include_similarity=True,
            )
            query_to_terms[position] = self._sum_bestmatch_rows(query, chunk_size)
            forward_dynamic_bytes = (
                chunk_size * len(self.terms) * float_bytes
                + 2 * len(self.terms) * double_bytes
            )
            peak_bytes = max(
                peak_bytes,
                matrix_bytes + retained_index_bytes + forward_dynamic_bytes,
            )
        terms_to_queries = np.empty((len(queries), len(self.terms)), dtype=np.float32)
        peak_bytes = max(
            peak_bytes,
            2 * matrix_bytes + retained_index_bytes,
        )

        budget_bytes = (
            float(max_workspace_mb) * 1e6
            if max_workspace_mb and max_workspace_mb > 0
            else float("inf")
        )
        ranges = []
        start = 0
        packed_genes = 0
        for end, size in enumerate(query_sizes, start=1):
            candidate_genes = packed_genes + int(size)
            candidate_queries = end - start
            candidate_bytes = (
                self.E_unit.shape[0]
                * (candidate_genes + candidate_queries)
                * np.dtype(np.float32).itemsize
            )
            if end - start > 1 and candidate_bytes > budget_bytes:
                ranges.append((start, end - 1))
                start = end - 1
                packed_genes = int(size)
            else:
                packed_genes = candidate_genes
        ranges.append((start, len(queries)))

        for start, end in ranges:
            local_queries = queries[start:end]
            local_sizes = query_sizes[start:end]
            if len(local_queries) == 1:
                query = local_queries[0]
                chunk_size = self._query_chunk_size(
                    query.size,
                    max_workspace_mb,
                    include_similarity=True,
                )
                if chunk_size < query.size:
                    best_to_query = self._best_to_query(query, chunk_size)
                    dynamic_bytes = (
                        int(best_to_query.nbytes)
                        + chunk_size
                        * max(
                            len(self.terms),
                            self.E_unit.shape[0] + self.E_unit.shape[1],
                        )
                        * float_bytes
                    )
                    peak_bytes = max(
                        peak_bytes,
                        2 * matrix_bytes + retained_index_bytes + dynamic_bytes,
                    )
                    terms_to_queries[start, :] = np.asarray(
                        self.membership @ best_to_query,
                        dtype=np.float32,
                    )
                    continue
            flat = np.concatenate(local_queries).astype(np.int32, copy=False)
            offsets = np.empty(len(local_queries), dtype=np.int64)
            offsets[0] = 0
            if len(local_queries) > 1:
                np.cumsum(local_sizes[:-1], out=offsets[1:])

            sims = self.E_unit @ self.E_unit[flat].T
            best_to_queries = np.maximum.reduceat(sims, offsets, axis=1)
            dynamic_bytes = int(
                flat.nbytes
                + flat.size * self.E_unit.shape[1] * float_bytes
                + sims.nbytes
                + best_to_queries.nbytes
            )
            peak_bytes = max(
                peak_bytes,
                2 * matrix_bytes + retained_index_bytes + dynamic_bytes,
            )
            terms_to_queries[start:end, :] = np.asarray(
                self.membership @ best_to_queries, dtype=np.float32
            ).T

        true_scores = (query_to_terms + terms_to_queries) / (
            query_sizes[:, None].astype(np.float32)
            + self.sizes[None, :].astype(np.float32)
        )
        true_scores = np.asarray(true_scores, dtype=np.float32)
        peak_bytes = max(
            peak_bytes,
            3 * matrix_bytes + retained_index_bytes,
        )
        requested_bytes = (
            0
            if max_workspace_mb is None or float(max_workspace_mb) <= 0.0
            else int(float(max_workspace_mb) * 1e6)
        )
        return ScoreResult(
            scores=true_scores,
            stats=ScoreStats.create(
                "indexed_batch",
                workspace_bytes=peak_bytes,
                details={
                    "query_sizes": tuple(int(size) for size in query_sizes),
                    "query_batch_ranges": tuple(
                        (int(start), int(end)) for start, end in ranges
                    ),
                    "requested_workspace_bytes": requested_bytes,
                },
            ),
        )

    def compare(
        self,
        other: AndesIndex,
        *,
        row_positions: ArrayLike | None = None,
        column_positions: ArrayLike | None = None,
        max_workspace_mb: float | None = 128,
    ) -> ScoreResult:
        """Compute exact BMA scores for selected axes of compatible indexes."""
        self.assert_compatible_with(other)
        rows = (
            np.arange(len(self.terms), dtype=np.int32)
            if row_positions is None
            else _ordered_unique_int_indices(
                "row_positions",
                row_positions,
                len(self.terms),
            )
        )
        columns = (
            np.arange(len(other.terms), dtype=np.int32)
            if column_positions is None
            else _ordered_unique_int_indices(
                "column_positions",
                column_positions,
                len(other.terms),
            )
        )
        same_term_axis = (
            self.terms == other.terms
            and np.array_equal(self.sizes, other.sizes)
            and np.array_equal(self.membership.indptr, other.membership.indptr)
            and np.array_equal(self.membership.indices, other.membership.indices)
        )
        symmetric_reuse = same_term_axis and np.array_equal(rows, columns)

        float_bytes = np.dtype(np.float32).itemsize
        scores = np.empty((rows.size, columns.size), dtype=np.float32)
        output_bytes = int(scores.nbytes)
        requested_bytes = (
            0
            if max_workspace_mb is None or float(max_workspace_mb) <= 0.0
            else int(float(max_workspace_mb) * 1e6)
        )
        if requested_bytes == 0:
            block_size = max(rows.size, columns.size)
        else:
            available = max(0, requested_bytes - output_bytes)
            low = 1
            high = max(rows.size, columns.size)
            block_size = 1
            while low <= high:
                candidate = (low + high) // 2
                row_count = min(rows.size, candidate)
                column_count = min(columns.size, candidate)
                dynamic = (
                    self.E_unit.shape[0] * (row_count + column_count) * float_bytes
                    + 2 * row_count * column_count * float_bytes
                )
                if dynamic <= available:
                    block_size = candidate
                    low = candidate + 1
                else:
                    high = candidate - 1

        peak_bytes = output_bytes
        for row_start in range(0, rows.size, block_size):
            row_end = min(rows.size, row_start + block_size)
            row_block = rows[row_start:row_end]
            row_membership = self.membership[row_block, :]
            left_bestmatch = np.asarray(
                self.bestmatch[:, row_block],
                dtype=np.float32,
                order="C",
            )
            column_start = row_start if symmetric_reuse else 0
            for start in range(column_start, columns.size, block_size):
                end = min(columns.size, start + block_size)
                column_block = columns[start:end]
                column_membership = other.membership[column_block, :]
                right_bestmatch = np.asarray(
                    other.bestmatch[:, column_block],
                    dtype=np.float32,
                    order="C",
                )
                block_scores = np.asarray(
                    row_membership @ right_bestmatch,
                    dtype=np.float32,
                )
                reverse = np.asarray(
                    column_membership @ left_bestmatch,
                    dtype=np.float32,
                )
                block_scores += reverse.T
                column_sizes = other.sizes[column_block].astype(
                    np.float32,
                    copy=False,
                )
                for local_row, row_position in enumerate(row_block):
                    block_scores[local_row] /= (
                        np.float32(self.sizes[row_position]) + column_sizes
                    )
                scores[row_start:row_end, start:end] = block_scores
                if symmetric_reuse and start != row_start:
                    scores[start:end, row_start:row_end] = block_scores.T
                peak_bytes = max(
                    peak_bytes,
                    output_bytes
                    + int(left_bestmatch.nbytes)
                    + int(right_bestmatch.nbytes)
                    + int(block_scores.nbytes)
                    + int(reverse.nbytes),
                )

        return ScoreResult(
            scores=scores,
            stats=ScoreStats.create(
                "index_to_index",
                workspace_bytes=peak_bytes,
                symmetric_reuse=symmetric_reuse,
                details={
                    "rows": int(rows.size),
                    "columns": int(columns.size),
                    "block_size": int(block_size),
                    "requested_workspace_bytes": requested_bytes,
                },
            ),
        )


def calibrate_index_scores(
    result: ScoreResult,
    row_sizes: ArrayLike,
    left_index: AndesIndex,
    right_index: AndesIndex,
    null_model: BmaNullModel,
    *,
    in_place: bool = False,
) -> ScoreResult:
    """Calibrate one exact indexed result against an axis-compatible BMA null."""
    if not isinstance(result, ScoreResult):
        raise TypeError("result must be a ScoreResult")
    if not isinstance(left_index, AndesIndex) or not isinstance(
        right_index, AndesIndex
    ):
        raise TypeError("left_index and right_index must be AndesIndex objects")
    _validate_null_model_axes(null_model, left_index, right_index)

    row_sizes = np.asarray(row_sizes)
    if row_sizes.ndim != 1 or row_sizes.dtype.kind not in "iu":
        raise TypeError("row_sizes must be a one-dimensional integer array")
    if row_sizes.size and (
        int(row_sizes.min()) < 1 or int(row_sizes.max()) > np.iinfo(np.int32).max
    ):
        raise ValueError("row_sizes contains a value outside the int32 size range")
    row_sizes = row_sizes.astype(np.int32, copy=False)
    values = np.asarray(result.scores, dtype=np.float32)
    squeeze = values.ndim == 1
    matrix = values[None, :] if squeeze else values
    expected = (row_sizes.size, len(right_index.terms))
    if matrix.shape != expected:
        raise ValueError(
            f"indexed score shape {matrix.shape} does not match {expected}"
        )

    output = matrix if in_place else None
    calibrated = null_model.standardize_matrix(
        matrix,
        row_sizes,
        right_index.sizes,
        out=output,
    )
    if squeeze:
        calibrated = calibrated[0]
    return ScoreResult(scores=calibrated, stats=result.stats)


def _load_json(index_dir, filename):
    try:
        with open(index_dir / filename, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid_index(index_dir, f"cannot read {filename}: {exc}") from exc


def _validate_loaded_index(
    index_dir,
    metadata,
    gene_list,
    terms,
    E_unit,
    bestmatch,
    sizes,
    background,
    membership,
    verify,
):
    if not isinstance(metadata, dict):
        raise _invalid_index(index_dir, "metadata.json must contain an object")
    version = metadata.get("index_version")
    if version != INDEX_VERSION:
        raise _invalid_index(
            index_dir,
            f"unsupported index_version {version!r}; expected {INDEX_VERSION}",
        )
    missing_metadata = sorted(_METADATA_KEYS - set(metadata))
    if missing_metadata:
        raise _invalid_index(
            index_dir,
            "metadata.json is missing keys: " + ", ".join(missing_metadata),
        )

    if not isinstance(gene_list, list) or any(
        not isinstance(gene, str) or not gene for gene in gene_list
    ):
        raise _invalid_index(index_dir, "genes.json must contain non-empty strings")
    if len(set(gene_list)) != len(gene_list):
        raise _invalid_index(index_dir, "genes.json contains duplicate gene names")
    if not isinstance(terms, list) or any(
        not isinstance(term, str) or not term for term in terms
    ):
        raise _invalid_index(index_dir, "terms.json must contain non-empty strings")
    if len(set(terms)) != len(terms):
        raise _invalid_index(index_dir, "terms.json contains duplicate term names")

    n_genes = len(gene_list)
    n_terms = len(terms)
    embedding_dim = int(metadata.get("embedding_dim", -1))
    expected_shapes = {
        "embedding.npy": (n_genes, embedding_dim),
        "bestmatch.npy": (n_genes, n_terms),
        "sizes.npy": (n_terms,),
    }
    arrays = {
        "embedding.npy": E_unit,
        "bestmatch.npy": bestmatch,
        "sizes.npy": sizes,
    }
    for filename, expected_shape in expected_shapes.items():
        if arrays[filename].shape != expected_shape:
            raise _invalid_index(
                index_dir,
                f"{filename} has shape {arrays[filename].shape}; "
                f"expected {expected_shape}",
            )
    if E_unit.dtype != np.float32:
        raise _invalid_index(
            index_dir, f"embedding.npy has dtype {E_unit.dtype}; expected float32"
        )
    if bestmatch.dtype != np.float32:
        raise _invalid_index(
            index_dir, f"bestmatch.npy has dtype {bestmatch.dtype}; expected float32"
        )
    if sizes.dtype != np.int32:
        raise _invalid_index(
            index_dir, f"sizes.npy has dtype {sizes.dtype}; expected int32"
        )
    if background.ndim != 1 or background.dtype != np.int32:
        raise _invalid_index(
            index_dir,
            "background.npy must be a one-dimensional int32 array",
        )
    if membership.shape != (n_terms, n_genes):
        raise _invalid_index(
            index_dir,
            f"membership.npz has shape {membership.shape}; "
            f"expected {(n_terms, n_genes)}",
        )
    if membership.dtype != np.float32:
        raise _invalid_index(
            index_dir,
            f"membership.npz has dtype {membership.dtype}; expected float32",
        )
    if membership.indices.dtype != np.int32:
        raise _invalid_index(
            index_dir,
            "membership.npz column indices must use int32",
        )

    scalar_metadata = {
        "n_genes": n_genes,
        "embedding_dim": E_unit.shape[1],
        "n_terms": n_terms,
    }
    for key, expected in scalar_metadata.items():
        if metadata.get(key) != expected:
            raise _invalid_index(
                index_dir,
                f"metadata {key}={metadata.get(key)!r}; expected {expected}",
            )
    background_policy = metadata.get("background_policy")
    if not isinstance(background_policy, str) or not background_policy:
        raise _invalid_index(index_dir, "background_policy must be a non-empty string")
    if not isinstance(metadata.get("runtime"), dict):
        raise _invalid_index(index_dir, "runtime metadata must be an object")

    if background.size == 0:
        raise _invalid_index(index_dir, "background.npy is empty")
    if np.any(background < 0) or np.any(background >= n_genes):
        raise _invalid_index(index_dir, "background.npy contains out-of-range indices")
    if not np.array_equal(background, np.unique(background)):
        raise _invalid_index(
            index_dir, "background.npy must contain sorted unique indices"
        )

    membership = membership.tocsr(copy=False)
    membership.sort_indices()
    if membership.nnz and not np.all(membership.data == 1.0):
        raise _invalid_index(index_dir, "membership.npz must contain binary values")
    for i, term in enumerate(terms):
        row_start = int(membership.indptr[i])
        row_end = int(membership.indptr[i + 1])
        idx = membership.indices[row_start:row_end]
        if idx.size == 0:
            raise _invalid_index(index_dir, f"term {term!r} has no gene indices")
        if int(idx[0]) < 0 or int(idx[-1]) >= n_genes:
            raise _invalid_index(
                index_dir, f"term {term!r} contains an out-of-range gene index"
            )
        if not np.array_equal(idx, np.unique(idx)):
            raise _invalid_index(
                index_dir, f"term {term!r} indices must be sorted and unique"
            )
        if np.setdiff1d(idx, background, assume_unique=True).size:
            raise _invalid_index(
                index_dir,
                f"background.npy omits members of term {term!r}",
            )
        if int(sizes[i]) != idx.size:
            raise _invalid_index(
                index_dir,
                f"sizes.npy disagrees with term {term!r}: "
                f"{int(sizes[i])} != {idx.size}",
            )

    if verify != "none":
        expected_hashes = {
            "gene_list_hash": artifacts.hash_strings(gene_list),
            "terms_hash": artifacts.hash_strings(terms),
            "sizes_hash": artifacts.hash_array(sizes),
            "members_hash": artifacts.hash_array(membership.indices),
            "background_hash": artifacts.hash_array(background),
            "membership_hash": artifacts.hash_sparse_csr(membership),
        }
        expected_embedding_fingerprint = artifacts.combine_fingerprints(
            "andes_embedding_v1",
            {
                "genes": expected_hashes["gene_list_hash"],
                "vectors": str(metadata["embedding_hash"]),
            },
        )
        expected_database_fingerprint = artifacts.combine_fingerprints(
            "andes_genesets_v1",
            {
                "background": expected_hashes["background_hash"],
                "embedding": expected_embedding_fingerprint,
                "members": expected_hashes["members_hash"],
                "offsets": artifacts.hash_array(
                    np.asarray(membership.indptr, dtype=np.int64)
                ),
                "terms": expected_hashes["terms_hash"],
            },
        )
        expected_hashes.update(
            {
                "embedding_fingerprint": expected_embedding_fingerprint,
                "database_fingerprint": expected_database_fingerprint,
            }
        )
        if verify == "full":
            for block in artifacts.iter_array_row_chunks(E_unit):
                if not np.isfinite(block).all():
                    raise _invalid_index(
                        index_dir,
                        "embedding.npy contains non-finite values",
                    )
                if not np.allclose(
                    np.linalg.norm(block, axis=1),
                    1.0,
                    rtol=2e-5,
                    atol=2e-6,
                ):
                    raise _invalid_index(
                        index_dir,
                        "embedding.npy rows are not unit-normalized",
                    )
            for block in artifacts.iter_array_row_chunks(bestmatch):
                if not np.isfinite(block).all():
                    raise _invalid_index(
                        index_dir,
                        "bestmatch.npy contains non-finite values",
                    )
            expected_hashes.update(
                {
                    "embedding_hash": artifacts.hash_array(E_unit),
                    "bestmatch_hash": artifacts.hash_array(bestmatch),
                }
            )
        for key, expected in expected_hashes.items():
            if metadata.get(key) != expected:
                raise _invalid_index(
                    index_dir,
                    f"{key} mismatch; the index payload may be corrupt or stale",
                )
    return membership


def _load_andes_index_unlocked(
    index_dir: str | Path,
    *,
    mmap: bool = False,
    verify: Literal["none", "metadata", "full"] = "metadata",
) -> AndesIndex:
    """Load one index while the caller holds its publication lock."""
    if verify not in {"none", "metadata", "full"}:
        raise ValueError("verify must be 'none', 'metadata', or 'full'")
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        raise _invalid_index(index_dir, "index directory does not exist")
    missing = [name for name in _INDEX_FILES if not (index_dir / name).is_file()]
    if missing:
        raise _invalid_index(index_dir, "missing required files: " + ", ".join(missing))

    metadata = _load_json(index_dir, "metadata.json")
    gene_list = _load_json(index_dir, "genes.json")
    terms = _load_json(index_dir, "terms.json")
    mmap_mode = "r" if mmap else None
    try:
        E_unit = np.load(
            index_dir / "embedding.npy",
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        bestmatch = np.load(
            index_dir / "bestmatch.npy",
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        sizes = np.load(index_dir / "sizes.npy", allow_pickle=False)
        background = np.load(index_dir / "background.npy", allow_pickle=False)
        membership = sparse.load_npz(index_dir / "membership.npz").tocsr()
    except ValueError as exc:
        if str(exc).startswith("invalid ANDES index"):
            raise
        raise _invalid_index(index_dir, f"cannot load array payload: {exc}") from exc
    except OSError as exc:
        raise _invalid_index(index_dir, f"cannot load array payload: {exc}") from exc

    try:
        membership = _validate_loaded_index(
            index_dir,
            metadata,
            gene_list,
            terms,
            E_unit,
            bestmatch,
            sizes,
            background,
            membership,
            verify,
        )
    except BaseException:
        _close_memmap(E_unit)
        _close_memmap(bestmatch)
        raise
    for value in (E_unit, bestmatch, sizes, background):
        value.setflags(write=False)
    for value in (membership.data, membership.indices, membership.indptr):
        value.setflags(write=False)

    gene_names = tuple(gene_list)
    term_names = tuple(terms)
    gene_to_index = MappingProxyType({gene: i for i, gene in enumerate(gene_names)})
    term_to_index = MappingProxyType({term: i for i, term in enumerate(term_names)})

    return AndesIndex(
        index_dir=index_dir,
        E_unit=E_unit,
        gene_list=gene_names,
        gene_to_index=gene_to_index,
        terms=term_names,
        term_to_index=term_to_index,
        sizes=sizes,
        background=background,
        membership=membership,
        bestmatch=bestmatch,
        metadata=MappingProxyType(metadata),
    )


def load_andes_index(
    index_dir: str | Path,
    *,
    mmap: bool = False,
    verify: Literal["none", "metadata", "full"] = "metadata",
) -> AndesIndex:
    """Load a consistently published index under its shared artifact lock.

    ``metadata`` checks shapes, dtypes, and compact identity data. The large
    embedding and best-match arrays remain lazy. ``full`` also scans and hashes
    those arrays and checks their values are finite. Index construction runs
    the full audit before publication.

    The shared lock is released after every payload has been opened and
    validated. Existing memory maps keep their original file descriptors, so
    a later controlled replacement cannot mix payload generations within the
    returned object.
    """
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        raise _invalid_index(index_dir, "index directory does not exist")
    with artifacts.artifact_lock(
        index_dir,
        shared=True,
        create=False,
    ):
        return _load_andes_index_unlocked(
            index_dir,
            mmap=mmap,
            verify=verify,
        )
