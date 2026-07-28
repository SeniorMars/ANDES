"""ANDES: exact embedding-based gene-set comparison and enrichment."""

from .data import EmbeddingSpace, GeneSetDatabase
from .nulls import BmaNullModel, NullSpec, RankedNullModel
from .scoring import (
    IndexedBestMatch,
    ScoreResult,
    ScoreStats,
    calibrate_bma,
    calibrate_ranked,
    score_bma_matrix,
    score_ranked,
)
from .runtime import (
    blas_runtime_info,
    format_blas_runtime,
    numba_warmup_requirements,
    resolve_query_blas_limit,
)


__version__ = "0.1.0"

__all__ = [
    "BmaNullModel",
    "EmbeddingSpace",
    "GeneSetDatabase",
    "IndexedBestMatch",
    "NullSpec",
    "RankedNullModel",
    "ScoreResult",
    "ScoreStats",
    "blas_runtime_info",
    "calibrate_bma",
    "calibrate_ranked",
    "format_blas_runtime",
    "numba_warmup_requirements",
    "resolve_query_blas_limit",
    "score_bma_matrix",
    "score_ranked",
]
