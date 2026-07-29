"""Testable application services above the ANDES numerical kernels."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from . import artifacts
from .data import EmbeddingSpace, FloatArray, GeneSetDatabase, IntArray
from .nulls import BmaNullModel, RankedNullModel
from .provenance import RunProvenance, write_result_sidecar
from .ranked import RANKED_ES_TIE_POLICY
from .scoring import (
    IndexedBestMatch,
    ScoreResult,
    calibrate_bma,
    calibrate_ranked,
    score_bma_matrix,
    score_ranked,
)


@dataclass(frozen=True, slots=True)
class CompareResult:
    scores: ScoreResult
    row_terms: tuple[str, ...]
    column_terms: tuple[str, ...]
    score_kind: str
    provenance: RunProvenance


def run_compare(
    *,
    embedding: EmbeddingSpace,
    left: GeneSetDatabase,
    right: GeneSetDatabase,
    workspace_mb: float = 128,
    null_model: BmaNullModel | None = None,
    runtime: Mapping[str, object] | None = None,
) -> CompareResult:
    """Execute exact BMA scoring and optional null calibration."""
    exact = score_bma_matrix(
        embedding,
        left,
        right,
        workspace_mb=workspace_mb,
    )
    if null_model is None:
        result = exact
        score_kind = "true_score"
        null_spec = None
    else:
        result = calibrate_bma(
            exact,
            null_model,
            embedding,
            left,
            right,
            in_place=True,
        )
        score_kind = "z_score"
        null_spec = null_model.spec.to_dict()

    provenance = RunProvenance(
        method="andes_bma",
        score_engine=result.stats.engine,
        score_kind=score_kind,
        similarity_dtype="float32",
        score_accumulator_dtype="float32",
        null_accumulator_dtype=(
            "float64" if null_model is not None else "not_applicable"
        ),
        output_dtype=str(result.scores.dtype),
        tie_policy="not_applicable",
        embedding_fingerprint=embedding.fingerprint,
        left_database_fingerprint=left.fingerprint,
        right_database_fingerprint=right.fingerprint,
        symmetric_reuse=result.stats.symmetric_reuse,
        null_spec=null_spec,
        runtime=runtime or {},
        extra={
            "workspace_bytes": result.stats.workspace_bytes,
            "rows": len(left.terms),
            "columns": len(right.terms),
            "left_background_policy": left.background_policy,
            "right_background_policy": right.background_policy,
        },
    )
    return CompareResult(
        scores=result,
        row_terms=left.terms,
        column_terms=right.terms,
        score_kind=score_kind,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class RankedResult:
    scores: ScoreResult
    true_scores: FloatArray
    terms: tuple[str, ...]
    sizes: IntArray
    null_means: NDArray[np.float64] | None
    null_stds: NDArray[np.float64] | None
    score_kind: str
    provenance: RunProvenance


def run_ranked(
    *,
    embedding: EmbeddingSpace,
    database: GeneSetDatabase,
    ranked_indices: ArrayLike,
    bestmatch: IndexedBestMatch | None = None,
    workspace_mb: float = 128,
    null_model: RankedNullModel | None = None,
    runtime: Mapping[str, object] | None = None,
) -> RankedResult:
    """Execute exact ranked scoring and optional null calibration."""
    exact = score_ranked(
        embedding,
        database,
        ranked_indices,
        bestmatch=bestmatch,
        workspace_mb=workspace_mb,
    )
    true_scores = exact.scores.copy()
    if null_model is None:
        result = exact
        score_kind = "true_score"
        null_spec = None
        null_means = None
        null_stds = None
    else:
        result = calibrate_ranked(
            exact,
            null_model,
            embedding,
            database,
            ranked_indices,
            in_place=True,
        )
        score_kind = "z_score"
        null_spec = null_model.spec.to_dict()
        null_means = np.asarray(
            null_model.means[database.sizes],
            dtype=np.float64,
        )
        null_stds = np.asarray(
            null_model.stds[database.sizes],
            dtype=np.float64,
        )

    provenance = RunProvenance(
        method="andes_ranked",
        score_engine=result.stats.engine,
        score_kind=score_kind,
        similarity_dtype="float32",
        score_accumulator_dtype="float64",
        null_accumulator_dtype=(
            "float64" if null_model is not None else "not_applicable"
        ),
        output_dtype=str(result.scores.dtype),
        tie_policy=RANKED_ES_TIE_POLICY,
        embedding_fingerprint=embedding.fingerprint,
        left_database_fingerprint=database.fingerprint,
        null_spec=null_spec,
        runtime=runtime or {},
        extra={
            "workspace_bytes": result.stats.workspace_bytes,
            "ranked_genes": int(np.asarray(ranked_indices).size),
            "terms": len(database.terms),
            "background_policy": database.background_policy,
            "bestmatch_source": (
                "persistent_index" if bestmatch is not None else "transient_stream"
            ),
        },
    )
    return RankedResult(
        scores=result,
        true_scores=true_scores,
        terms=database.terms,
        sizes=database.sizes,
        null_means=null_means,
        null_stds=null_stds,
        score_kind=score_kind,
        provenance=provenance,
    )


def write_dataframe_atomic(
    dataframe: pd.DataFrame,
    output: str | Path,
    *,
    index: bool = True,
    index_label: str | None = None,
) -> None:
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
        dataframe.to_csv(temporary, index=index, index_label=index_label)
        os.replace(temporary, output)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def write_compare_result(result: CompareResult, output_path):
    """Atomically write a labeled matrix and its scientific provenance."""
    output = Path(output_path)
    companion_paths = ()
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
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise
        rows_path = output.with_suffix(".rows.json")
        columns_path = output.with_suffix(".columns.json")
        artifacts.write_json_atomic(rows_path, result.row_terms)
        artifacts.write_json_atomic(columns_path, result.column_terms)
        companion_paths = (rows_path, columns_path)
    else:
        frame = pd.DataFrame(
            values,
            index=result.row_terms,
            columns=result.column_terms,
        )
        write_dataframe_atomic(frame, output)
    return write_result_sidecar(
        output,
        result.provenance,
        companion_paths=companion_paths,
    )


def ranked_result_frame(result: RankedResult) -> pd.DataFrame:
    """Return the canonical aligned ranked-result table."""
    data: dict[str, object] = {
        "term": result.terms,
        "size": np.asarray(result.sizes, dtype=np.int32),
        "true_score": np.asarray(result.true_scores, dtype=np.float32),
    }
    if result.null_means is not None:
        data["null_mu"] = np.asarray(result.null_means, dtype=np.float64)
        data["null_sigma"] = np.asarray(result.null_stds, dtype=np.float64)
        data["z_score"] = np.asarray(result.scores.scores, dtype=np.float32)
    return pd.DataFrame(data).set_index("term")


def write_ranked_result(result: RankedResult, output_path, *, provenance_extra=None):
    """Atomically write one aligned ranked-score table and provenance."""
    frame = ranked_result_frame(result)
    write_dataframe_atomic(frame, output_path)
    provenance = result.provenance.for_tabular_output(
        {
            "term": "string",
            **{str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        },
        extra=provenance_extra,
    )
    return write_result_sidecar(output_path, provenance)
