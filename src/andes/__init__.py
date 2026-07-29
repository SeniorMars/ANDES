"""ANDES: exact embedding-based gene-set comparison and enrichment."""

from .data import EmbeddingSpace, GeneSetDatabase
from .nulls import BmaNullModel, NullSpec, RankedNullModel
from .runtime import (
    blas_runtime_info,
    format_blas_runtime,
)
from .scoring import (
    IndexedBestMatch,
    ScoreResult,
    ScoreStats,
    calibrate_bma,
    calibrate_ranked,
    score_bma_matrix,
    score_ranked,
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
    "score_bma_matrix",
    "score_ranked",
]
