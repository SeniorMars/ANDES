"""Typed, calibration-free public scoring API for ANDES."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from . import artifacts
from . import bma as bma_core
from . import ranked as ranked_core
from .data import EmbeddingSpace, GeneSetDatabase
from .nulls import BmaNullModel, RankedNullModel


@dataclass(frozen=True, slots=True)
class ScoreStats:
    """Execution metadata that does not affect score alignment."""

    engine: str
    workspace_bytes: int = 0
    symmetric_reuse: bool = False
    details: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def create(
        cls,
        engine,
        *,
        workspace_bytes=0,
        symmetric_reuse=False,
        details=None,
    ):
        return cls(
            engine=str(engine),
            workspace_bytes=max(0, int(workspace_bytes)),
            symmetric_reuse=bool(symmetric_reuse),
            details=MappingProxyType(dict(details or {})),
        )


@dataclass(slots=True)
class ScoreResult:
    """An aligned exact or calibrated score array plus execution metadata."""

    scores: np.ndarray
    stats: ScoreStats

    def __post_init__(self):
        values = np.asarray(self.scores)
        if values.dtype != np.float32:
            values = values.astype(np.float32)
        self.scores = values


@dataclass(frozen=True, slots=True)
class IndexedBestMatch:
    """A best-match matrix bound to its embedding and term database."""

    values: np.ndarray
    embedding_fingerprint: str
    database_fingerprint: str

    def __post_init__(self):
        values = np.asanyarray(self.values)
        if values.ndim != 2 or values.dtype != np.float32:
            raise TypeError(
                "indexed bestmatch must be a two-dimensional float32 matrix"
            )
        if not self.embedding_fingerprint or not self.database_fingerprint:
            raise ValueError("indexed bestmatch fingerprints must not be empty")
        object.__setattr__(self, "values", values)


def _validate_database_alignment(
    embedding: EmbeddingSpace,
    database: GeneSetDatabase,
    name: str,
):
    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    if not isinstance(database, GeneSetDatabase):
        raise TypeError(f"{name} must be a GeneSetDatabase")
    if database.embedding_fingerprint != embedding.fingerprint:
        raise ValueError(f"{name} was built for a different embedding or gene order")
    if database.members.size and int(database.members.max()) >= len(embedding.genes):
        raise ValueError(f"{name} is not aligned to the embedding")


def validate_bma_null_compatibility(
    null_model: BmaNullModel,
    embedding: EmbeddingSpace,
    left: GeneSetDatabase,
    right: GeneSetDatabase,
):
    """Reject a BMA null model built for different numerical inputs."""
    if not isinstance(null_model, BmaNullModel):
        raise TypeError("null_model must be a BmaNullModel")
    _validate_database_alignment(embedding, left, "left")
    _validate_database_alignment(embedding, right, "right")
    expected = (
        embedding.vector_hash,
        left.background_hash,
        right.background_hash,
    )
    actual = (
        null_model.spec.embedding_hash,
        *null_model.spec.population_hashes,
    )
    labels = ("embedding", "left background", "right background")
    for label, observed, wanted in zip(labels, actual, expected, strict=True):
        if observed != wanted:
            raise ValueError(f"null model {label} is incompatible")


def validate_ranked_null_compatibility(
    null_model: RankedNullModel,
    embedding: EmbeddingSpace,
    database: GeneSetDatabase,
    ranked_indices,
    *,
    ranked_hash=None,
):
    """Reject a ranked null model built for different numerical inputs."""
    if not isinstance(null_model, RankedNullModel):
        raise TypeError("null_model must be a RankedNullModel")
    _validate_database_alignment(embedding, database, "database")
    ranked = _canonical_ranked_indices(embedding, ranked_indices)
    if ranked_hash is None:
        ranked_embeddings = ranked_core.compute_ranked_emb(
            embedding.vectors,
            ranked,
        )
        ranked_hash = artifacts.hash_array(ranked_embeddings)
    expected = (
        embedding.vector_hash,
        database.background_hash,
        ranked_hash,
    )
    actual = (
        null_model.spec.embedding_hash,
        null_model.spec.population_hashes[0],
        null_model.spec.ranked_hash,
    )
    labels = ("embedding", "background", "ranking")
    for label, observed, wanted in zip(labels, actual, expected, strict=True):
        if observed != wanted:
            raise ValueError(f"null model {label} is incompatible")


def score_bma_matrix(
    embedding: EmbeddingSpace,
    left: GeneSetDatabase,
    right: GeneSetDatabase,
    *,
    engine: str = "bestmatch",
    workers: int = 1,
    numba_threshold: int = 400,
    workspace_mb: float = 1024,
    show_progress: bool = False,
) -> ScoreResult:
    """Compute exact BMA scores without consulting a null model."""
    _validate_database_alignment(embedding, left, "left")
    _validate_database_alignment(embedding, right, "right")
    engine = str(engine)
    if engine not in {"bestmatch", "batched", "pairwise"}:
        raise ValueError(f"unknown BMA engine {engine!r}")
    workers = max(1, int(workers))

    left_indices = left.as_index_mapping()
    right_indices = right.as_index_mapping()
    symmetric = left.fingerprint == right.fingerprint
    details = {}

    if engine == "bestmatch":
        scores, raw_stats = bma_core.score_bma_matrix_bestmatch(
            embedding.vectors,
            left.terms,
            right.terms,
            left_indices,
            right_indices,
            symmetric=symmetric,
            max_workspace_mb=workspace_mb,
            show_progress=show_progress,
        )
        details.update(raw_stats)
        workspace_bytes = int(float(raw_stats.get("estimated_peak_mb", 0.0)) * 1e6)
        symmetric_reuse = bool(raw_stats.get("symmetric_reuse", False))
    elif engine == "batched":
        left_blocks = bma_core.precompute_term_embedding_blocks(
            embedding.vectors, left_indices
        )
        right_blocks = (
            left_blocks
            if symmetric
            else bma_core.precompute_term_embedding_blocks(
                embedding.vectors, right_indices
            )
        )
        scores, effective_workers, per_worker_mb = bma_core.score_bma_matrix_batched(
            left.terms,
            right.terms,
            left_blocks,
            right_blocks,
            symmetric=symmetric,
            n_workers=workers,
            max_workspace_mb=workspace_mb,
            show_progress=show_progress,
        )
        block_bytes = sum(block.nbytes for block in left_blocks.values())
        if right_blocks is not left_blocks:
            block_bytes += sum(block.nbytes for block in right_blocks.values())
        workspace_bytes = int(per_worker_mb * 1e6 * effective_workers + block_bytes)
        symmetric_reuse = symmetric
        details.update(
            {
                "effective_workers": int(effective_workers),
                "per_worker_workspace_mb": float(per_worker_mb),
                "term_block_bytes": int(block_bytes),
            }
        )
    else:
        scores = bma_core.score_bma_matrix_pairwise(
            embedding.vectors,
            left.terms,
            right.terms,
            left_indices,
            right_indices,
            symmetric=symmetric,
            n_workers=workers,
            numba_threshold=numba_threshold,
            show_progress=show_progress,
        )
        workspace_bytes = 0
        symmetric_reuse = symmetric
        details["effective_workers"] = workers

    return ScoreResult(
        scores=np.asarray(scores, dtype=np.float32),
        stats=ScoreStats.create(
            engine,
            workspace_bytes=workspace_bytes,
            symmetric_reuse=symmetric_reuse,
            details=details,
        ),
    )


def calibrate_bma(
    result: ScoreResult,
    null_model: BmaNullModel,
    embedding: EmbeddingSpace,
    left: GeneSetDatabase,
    right: GeneSetDatabase,
    *,
    in_place: bool = False,
) -> ScoreResult:
    """Apply BMA null calibration, optionally reusing the exact-score array."""
    validate_bma_null_compatibility(null_model, embedding, left, right)
    output = result.scores if in_place else None
    calibrated = null_model.standardize_matrix(
        result.scores,
        left.sizes,
        right.sizes,
        out=output,
    )
    return ScoreResult(scores=calibrated, stats=result.stats)


def _canonical_ranked_indices(embedding, ranked_indices):
    ranked = np.asarray(ranked_indices)
    if ranked.ndim != 1 or ranked.size == 0:
        raise ValueError("ranked_indices must be a non-empty vector")
    if ranked.dtype.kind not in "iu":
        raise TypeError("ranked_indices must contain integers")
    ranked = ranked.astype(np.int32, copy=False)
    if int(ranked.min()) < 0 or int(ranked.max()) >= len(embedding.genes):
        raise IndexError("ranked_indices contains an out-of-range embedding row")
    if np.unique(ranked).size != ranked.size:
        raise ValueError("ranked_indices must not contain duplicate genes")
    return ranked


def score_ranked(
    embedding: EmbeddingSpace,
    database: GeneSetDatabase,
    ranked_indices,
    *,
    engine: str = "batched",
    bestmatch=None,
    workspace_mb: float = 128,
) -> ScoreResult:
    """Compute exact ranked ANDES scores aligned with ``database.terms``."""
    _validate_database_alignment(embedding, database, "database")
    ranked = _canonical_ranked_indices(embedding, ranked_indices)
    engine = str(engine)
    if engine not in {"batched", "bestmatch", "indexed"}:
        raise ValueError(f"unknown ranked engine {engine!r}")

    term_indices = database.as_index_mapping()
    ranked_embeddings = ranked_core.compute_ranked_emb(embedding.vectors, ranked)
    if engine == "batched":
        scores = ranked_core.score_terms_batched_exact(
            embedding.vectors,
            term_indices,
            database.terms,
            np.ascontiguousarray(ranked_embeddings.T, dtype=np.float32),
            batch_bytes=max(4, int(float(workspace_mb) * 1e6)),
        )
        details = {}
        workspace_bytes = max(4, int(float(workspace_mb) * 1e6))
    elif engine == "bestmatch":
        scores, raw_stats = ranked_core.score_terms_bestmatch_exact(
            embedding.vectors,
            term_indices,
            database.terms,
            ranked_embeddings,
            max_workspace_mb=workspace_mb,
        )
        details = dict(raw_stats)
        workspace_bytes = int(float(raw_stats["workspace_mb"]) * 1e6)
    else:
        if not isinstance(bestmatch, IndexedBestMatch):
            raise TypeError(
                "indexed ranked scoring requires an IndexedBestMatch artifact"
            )
        if bestmatch.embedding_fingerprint != embedding.fingerprint:
            raise ValueError("indexed bestmatch embedding is incompatible")
        if bestmatch.database_fingerprint != database.fingerprint:
            raise ValueError("indexed bestmatch database is incompatible")
        expected_shape = (len(embedding.genes), len(database.terms))
        if bestmatch.values.shape != expected_shape:
            raise ValueError(
                f"indexed bestmatch shape {bestmatch.values.shape} != {expected_shape}"
            )
        scores, _, raw_stats = ranked_core.score_terms_indexed(
            bestmatch.values,
            ranked,
            database.sizes,
            cache=None,
            max_workspace_mb=workspace_mb,
        )
        details = dict(raw_stats)
        workspace_bytes = int(float(raw_stats["workspace_mb"]) * 1e6)

    details["ranked_hash"] = artifacts.hash_array(ranked_embeddings)
    return ScoreResult(
        scores=np.asarray(scores, dtype=np.float32),
        stats=ScoreStats.create(
            engine,
            workspace_bytes=workspace_bytes,
            details=details,
        ),
    )


def calibrate_ranked(
    result: ScoreResult,
    null_model: RankedNullModel,
    embedding: EmbeddingSpace,
    database: GeneSetDatabase,
    ranked_indices,
    *,
    in_place: bool = False,
) -> ScoreResult:
    """Apply ranked null calibration, optionally reusing the score array."""
    validate_ranked_null_compatibility(
        null_model,
        embedding,
        database,
        ranked_indices,
        ranked_hash=result.stats.details.get("ranked_hash"),
    )
    output = result.scores if in_place else None
    calibrated = null_model.standardize(
        result.scores,
        database.sizes,
        out=output,
    )
    return ScoreResult(scores=calibrated, stats=result.stats)
