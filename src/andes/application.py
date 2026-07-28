"""Testable application services above the ANDES numerical kernels."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from . import artifacts
from .nulls import BmaNullModel, RankedNullModel
from .provenance import RunProvenance, write_result_sidecar
from .scoring import (
    IndexedBestMatch,
    ScoreResult,
    calibrate_bma,
    calibrate_ranked,
    score_bma_matrix,
    score_ranked,
)
from .data import EmbeddingSpace, GeneSetDatabase


@dataclass(frozen=True, slots=True)
class CompareRequest:
    embedding: EmbeddingSpace
    left: GeneSetDatabase
    right: GeneSetDatabase
    engine: str = "bestmatch"
    workers: int = 1
    numba_threshold: int = 400
    workspace_mb: float = 1024
    show_progress: bool = False
    null_model: BmaNullModel | None = None
    runtime: dict | None = None


@dataclass(slots=True)
class CompareResult:
    scores: ScoreResult
    row_terms: tuple[str, ...]
    column_terms: tuple[str, ...]
    score_kind: str
    provenance: RunProvenance


def run_compare(request: CompareRequest) -> CompareResult:
    """Execute exact BMA scoring and optional null calibration."""
    exact = score_bma_matrix(
        request.embedding,
        request.left,
        request.right,
        engine=request.engine,
        workers=request.workers,
        numba_threshold=request.numba_threshold,
        workspace_mb=request.workspace_mb,
        show_progress=request.show_progress,
    )
    if request.null_model is None:
        result = exact
        score_kind = "true_score"
        null_spec = None
    else:
        result = calibrate_bma(
            exact,
            request.null_model,
            request.embedding,
            request.left,
            request.right,
            in_place=True,
        )
        score_kind = "z_score"
        null_spec = request.null_model.spec.to_dict()

    provenance = RunProvenance(
        method="andes_bma",
        score_engine=request.engine,
        score_kind=score_kind,
        numeric_dtype=str(result.scores.dtype),
        accumulator_dtype="float64",
        embedding_fingerprint=request.embedding.fingerprint,
        left_database_fingerprint=request.left.fingerprint,
        right_database_fingerprint=request.right.fingerprint,
        symmetric_reuse=result.stats.symmetric_reuse,
        null_spec=null_spec,
        runtime=request.runtime or {},
        extra={
            "workspace_bytes": result.stats.workspace_bytes,
            "rows": len(request.left.terms),
            "columns": len(request.right.terms),
        },
    )
    return CompareResult(
        scores=result,
        row_terms=request.left.terms,
        column_terms=request.right.terms,
        score_kind=score_kind,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class RankedRequest:
    embedding: EmbeddingSpace
    database: GeneSetDatabase
    ranked_indices: np.ndarray
    engine: str = "batched"
    bestmatch: IndexedBestMatch | None = None
    workspace_mb: float = 128
    null_model: RankedNullModel | None = None
    runtime: dict | None = None


@dataclass(slots=True)
class RankedResult:
    scores: ScoreResult
    true_scores: np.ndarray
    terms: tuple[str, ...]
    score_kind: str
    provenance: RunProvenance


def run_ranked(request: RankedRequest) -> RankedResult:
    """Execute exact ranked scoring and optional null calibration."""
    exact = score_ranked(
        request.embedding,
        request.database,
        request.ranked_indices,
        engine=request.engine,
        bestmatch=request.bestmatch,
        workspace_mb=request.workspace_mb,
    )
    true_scores = exact.scores.copy()
    if request.null_model is None:
        result = exact
        score_kind = "true_score"
        null_spec = None
    else:
        result = calibrate_ranked(
            exact,
            request.null_model,
            request.embedding,
            request.database,
            request.ranked_indices,
            in_place=True,
        )
        score_kind = "z_score"
        null_spec = request.null_model.spec.to_dict()

    provenance = RunProvenance(
        method="andes_ranked",
        score_engine=request.engine,
        score_kind=score_kind,
        numeric_dtype=str(result.scores.dtype),
        accumulator_dtype="float64",
        embedding_fingerprint=request.embedding.fingerprint,
        left_database_fingerprint=request.database.fingerprint,
        null_spec=null_spec,
        runtime=request.runtime or {},
        extra={
            "workspace_bytes": result.stats.workspace_bytes,
            "ranked_genes": int(np.asarray(request.ranked_indices).size),
            "terms": len(request.database.terms),
        },
    )
    return RankedResult(
        scores=result,
        true_scores=true_scores,
        terms=request.database.terms,
        score_kind=score_kind,
        provenance=provenance,
    )


def write_dataframe_atomic(dataframe, output, **to_csv_kwargs):
    """Atomically publish a pandas CSV payload."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataframe.to_csv(temporary, **to_csv_kwargs)
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_compare_result(result: CompareResult, output_path):
    """Atomically write a labeled matrix and its scientific provenance."""
    output = Path(output_path)
    values = np.asarray(result.scores.scores, dtype=np.float32)
    expected = (len(result.row_terms), len(result.column_terms))
    if values.shape != expected:
        raise ValueError(f"comparison result shape {values.shape} != {expected}")
    if output.suffix.lower() == ".npy":
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".npy",
            dir=output.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.save(temporary, values, allow_pickle=False)
            os.replace(temporary, output)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        artifacts.write_json_atomic(
            output.with_suffix(".rows.json"),
            result.row_terms,
        )
        artifacts.write_json_atomic(
            output.with_suffix(".columns.json"),
            result.column_terms,
        )
    else:
        frame = pd.DataFrame(
            values,
            index=result.row_terms,
            columns=result.column_terms,
        )
        write_dataframe_atomic(frame, output)
    return write_result_sidecar(output, result.provenance)


def write_ranked_result(result: RankedResult, output_path):
    """Atomically write one aligned ranked-score table and provenance."""
    frame = pd.DataFrame(
        {
            "term": result.terms,
            result.score_kind: np.asarray(result.scores.scores, dtype=np.float32),
        }
    ).set_index("term")
    write_dataframe_atomic(frame, output_path)
    return write_result_sidecar(output_path, result.provenance)
