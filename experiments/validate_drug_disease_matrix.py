"""Validate a DrugBank-by-OMIM BMA matrix against a frozen paper artifact.

This experiment checks a frozen matrix produced by the package. The paper's
ranked-expression workflow and figures are outside its scope.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from andes import application, artifacts, data
from andes.nulls import BmaNullModel, resolve_bma_null, resolve_null_seed


@dataclass(frozen=True, slots=True)
class AcceptanceEnvelope:
    """Checked-in release thresholds for the frozen matrix comparison."""

    minimum_drug_coverage: float = 0.99
    minimum_disease_coverage: float = 0.99
    minimum_jointly_finite_fraction: float = 1.0
    minimum_pearson: float = 0.99
    minimum_spearman: float = 0.99
    maximum_mae: float = 0.1
    maximum_absolute_difference: float = 1.0


DEFAULT_ACCEPTANCE_ENVELOPE = AcceptanceEnvelope()


def _subset_database(path, embedding, requested_terms, min_size, max_size):
    requested_terms = tuple(str(term) for term in requested_terms)
    database = data.load_gene_set_database(
        path,
        embedding,
        min_size=min_size,
        max_size=max_size,
    )
    terms = tuple(term for term in requested_terms if term in database.term_to_index)
    if not terms:
        raise ValueError(f"no requested terms from {path} passed the size filters")
    dropped = [term for term in requested_terms if term not in database.term_to_index]
    return database.select_terms(terms), dropped


def _load_or_build_null(
    path,
    embedding,
    left,
    right,
    *,
    iterations,
    seed,
    blas_threads,
    rebuild,
    no_build,
):
    model = resolve_bma_null(
        embedding,
        left.background,
        right.background,
        left.sizes,
        right.sizes,
        path=path,
        iterations=iterations,
        seed=seed,
        no_build=no_build,
        rebuild=rebuild,
        blas_threads=blas_threads,
    ).model
    if not isinstance(model, BmaNullModel):
        raise TypeError("BMA null resolver returned the wrong model kind")
    return model


def _comparison_metrics(reference, candidate):
    reference_values = reference.to_numpy(dtype=np.float64).ravel()
    candidate_values = candidate.to_numpy(dtype=np.float64).ravel()
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate matrices must have one shape")
    reference_finite = np.isfinite(reference_values)
    candidate_finite = np.isfinite(candidate_values)
    bad_candidate = reference_finite & ~candidate_finite
    if np.any(bad_candidate):
        raise ValueError(
            f"candidate contains {int(bad_candidate.sum())} non-finite values "
            "where the reference is finite"
        )
    finite = np.isfinite(reference_values) & np.isfinite(candidate_values)
    reference_finite_count = int(reference_finite.sum())
    jointly_finite_count = int(finite.sum())
    reference_values = reference_values[finite]
    candidate_values = candidate_values[finite]
    if not reference_values.size:
        raise ValueError("reference and candidate have no jointly finite values")
    difference = candidate_values - reference_values
    has_variable_scores = (
        reference_values.size > 1
        and np.ptp(reference_values) > 0.0
        and np.ptp(candidate_values) > 0.0
    )
    pearson = (
        float(np.corrcoef(reference_values, candidate_values)[0, 1])
        if has_variable_scores
        else np.nan
    )
    spearman = (
        float(
            pd.Series(reference_values).corr(
                pd.Series(candidate_values),
                method="spearman",
            )
        )
        if has_variable_scores
        else np.nan
    )
    return {
        "reference_finite_values": reference_finite_count,
        "jointly_finite_values": jointly_finite_count,
        "jointly_finite_fraction": (
            jointly_finite_count / reference_finite_count
            if reference_finite_count
            else 0.0
        ),
        "pearson": pearson if np.isfinite(pearson) else None,
        "spearman": spearman if np.isfinite(spearman) else None,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "median_abs_diff": float(np.median(np.abs(difference))),
        "max_abs_diff": float(np.max(np.abs(difference))),
    }


def _acceptance_failures(metrics, envelope):
    failures = []

    def require_minimum(name, value, minimum):
        if value is None or float(value) < float(minimum):
            failures.append(f"{name} {value!r} is below {minimum}")

    require_minimum(
        "drug coverage",
        metrics["drug_coverage"],
        envelope.minimum_drug_coverage,
    )
    require_minimum(
        "disease coverage",
        metrics["disease_coverage"],
        envelope.minimum_disease_coverage,
    )
    require_minimum(
        "jointly finite fraction",
        metrics["jointly_finite_fraction"],
        envelope.minimum_jointly_finite_fraction,
    )
    require_minimum("Pearson correlation", metrics["pearson"], envelope.minimum_pearson)
    require_minimum(
        "Spearman correlation",
        metrics["spearman"],
        envelope.minimum_spearman,
    )
    mae = metrics["mae"]
    if mae is None or float(mae) > envelope.maximum_mae:
        failures.append(f"MAE {mae!r} exceeds {envelope.maximum_mae}")
    maximum_difference = metrics["max_abs_diff"]
    if (
        maximum_difference is None
        or float(maximum_difference) > envelope.maximum_absolute_difference
    ):
        failures.append(
            "maximum absolute difference "
            f"{maximum_difference!r} exceeds "
            f"{envelope.maximum_absolute_difference}"
        )
    return failures


def _validation_report(
    reference,
    candidate,
    *,
    requested_drugs,
    requested_diseases,
    envelope=DEFAULT_ACCEPTANCE_ENVELOPE,
) -> dict[str, object]:
    retained_drugs, retained_diseases = candidate.shape
    report: dict[str, object] = {
        "requested_drugs": int(requested_drugs),
        "retained_drugs": int(retained_drugs),
        "drug_coverage": retained_drugs / requested_drugs,
        "requested_diseases": int(requested_diseases),
        "retained_diseases": int(retained_diseases),
        "disease_coverage": retained_diseases / requested_diseases,
        "acceptance_envelope": asdict(envelope),
    }
    try:
        report.update(_comparison_metrics(reference, candidate))
        failures = _acceptance_failures(report, envelope)
    except ValueError as exc:
        reference_values = reference.to_numpy(dtype=np.float64)
        candidate_values = candidate.to_numpy(dtype=np.float64)
        reference_finite = np.isfinite(reference_values)
        jointly_finite = reference_finite & np.isfinite(candidate_values)
        reference_finite_count = int(reference_finite.sum())
        report.update(
            {
                "reference_finite_values": reference_finite_count,
                "jointly_finite_values": int(jointly_finite.sum()),
                "jointly_finite_fraction": (
                    float(jointly_finite.sum()) / reference_finite_count
                    if reference_finite_count
                    else 0.0
                ),
                "pearson": None,
                "spearman": None,
                "mae": None,
                "rmse": None,
                "median_abs_diff": None,
                "max_abs_diff": None,
            }
        )
        failures = [str(exc)]
    report["passed"] = not failures
    report["failures"] = failures
    return report


def _average_precision(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size:
        raise ValueError("labels and scores must be same-length vectors")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("ground-truth labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("AUPRC scores must be finite")
    positives = int(labels.sum())
    if positives == 0:
        return np.nan

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    group_ends = np.r_[
        np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), labels.size - 1
    ]
    true_positives = np.cumsum(ranked_labels, dtype=np.int64)[group_ends]
    retrieved = group_ends + 1
    recall = true_positives / float(positives)
    precision = true_positives / retrieved
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def _write_auprc(reference, candidate, truth, output):
    common_rows = reference.index.intersection(truth.index, sort=False)
    common_columns = reference.columns.intersection(truth.columns, sort=False)
    if common_rows.empty or common_columns.empty:
        raise ValueError("ground truth does not overlap reference terms")
    rows = []
    for disease in common_columns:
        labels = truth.loc[common_rows, disease].to_numpy()
        rows.append(
            {
                "disease": disease,
                "reference_auprc": _average_precision(
                    labels,
                    reference.loc[common_rows, disease].to_numpy(),
                ),
                "optimized_auprc": _average_precision(
                    labels,
                    candidate.loc[common_rows, disease].to_numpy(),
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("disease")
    application.write_dataframe_atomic(frame, output)
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emb",
        default="data/embedding/node2vec_consensus.csv",
    )
    parser.add_argument(
        "--genelist",
        default="data/embedding/consensus_node.txt",
    )
    parser.add_argument(
        "--drug-gmt",
        default="paper/data/gmt/drugbank.2301.gmt",
    )
    parser.add_argument(
        "--disease-gmt",
        default="paper/data/gmt/omim.20231030.gmt",
    )
    parser.add_argument(
        "--reference-zscores",
        default="paper/results/drug_disease/drug_disease_fixed_seed.csv",
    )
    parser.add_argument(
        "--truth",
        default="",
        help="optional binary drug-by-disease CSV for AUPRC validation",
    )
    parser.add_argument("--out-dir", default="reports/drug_disease_validation")
    parser.add_argument("--cache", default="")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=300)
    parser.add_argument("--ite", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--workspace-mb", type=float, default=128.0)
    parser.add_argument("--limit-drugs", type=int, default=0)
    parser.add_argument("--limit-diseases", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write failed validation reports without returning a failure status",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("build", "require"),
        default="build",
    )
    args = parser.parse_args(argv)
    if args.min_size < 1 or args.max_size < args.min_size:
        parser.error("invalid gene-set size range")
    if args.ite < 2:
        parser.error("--ite must be at least 2")
    if args.blas_threads < 1:
        parser.error("--blas-threads must be positive")
    if args.workspace_mb <= 0:
        parser.error("--workspace-mb must be positive")
    if args.limit_drugs < 0 or args.limit_diseases < 0:
        parser.error("term limits must be non-negative")
    if args.cache_policy == "require" and args.rebuild_cache:
        parser.error("--rebuild-cache requires --cache-policy build")
    args.seed = resolve_null_seed(args.seed)
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(args.reference_zscores, index_col=0)
    reference.index = reference.index.astype(str)
    reference.columns = reference.columns.astype(str)
    if args.limit_drugs:
        reference = reference.iloc[: args.limit_drugs]
    if args.limit_diseases:
        reference = reference.iloc[:, : args.limit_diseases]
    requested_drugs, requested_diseases = reference.shape
    if requested_drugs == 0 or requested_diseases == 0:
        raise ValueError("reference matrix must contain at least one drug and disease")

    embedding = data.load_embedding_space(args.emb, args.genelist)
    left, dropped_left = _subset_database(
        args.drug_gmt,
        embedding,
        reference.index,
        args.min_size,
        args.max_size,
    )
    right, dropped_right = _subset_database(
        args.disease_gmt,
        embedding,
        reference.columns,
        args.min_size,
        args.max_size,
    )
    reference = reference.loc[list(left.terms), list(right.terms)]
    cache_path = Path(args.cache) if args.cache else output / "drug_disease_bma.null"

    with threadpool_limits(limits=args.blas_threads, user_api="blas"):
        null_model = _load_or_build_null(
            cache_path,
            embedding,
            left,
            right,
            iterations=args.ite,
            seed=args.seed,
            blas_threads=args.blas_threads,
            rebuild=args.rebuild_cache,
            no_build=args.cache_policy == "require",
        )
        result = application.run_compare(
            embedding=embedding,
            left=left,
            right=right,
            workspace_mb=args.workspace_mb,
            null_model=null_model,
        )

    candidate = pd.DataFrame(
        result.scores.scores,
        index=result.row_terms,
        columns=result.column_terms,
    )
    application.write_dataframe_atomic(
        candidate,
        output / "drug_disease_optimized_zscores.csv",
    )
    application.write_compare_result(
        result,
        output / "drug_disease_optimized_zscores.npy",
    )

    difference = candidate - reference
    difference_series = cast("pd.Series[float]", difference.stack())
    difference_long = difference_series.to_frame(
        name="optimized_minus_reference",
    )
    application.write_dataframe_atomic(
        difference_long,
        output / "drug_disease_zscore_differences.csv",
    )
    metrics: dict[str, object] = _validation_report(
        reference,
        candidate,
        requested_drugs=requested_drugs,
        requested_diseases=requested_diseases,
    )
    metrics.update(
        {
            "dropped_drugs": len(dropped_left),
            "dropped_diseases": len(dropped_right),
            "iterations": args.ite,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )

    if args.truth:
        truth = pd.read_csv(args.truth, index_col=0)
        truth.index = truth.index.astype(str)
        truth.columns = truth.columns.astype(str)
        auprc = _write_auprc(
            reference,
            candidate,
            truth,
            output / "drug_disease_auprc.csv",
        )
        reference_mean = float(auprc["reference_auprc"].mean())
        optimized_mean = float(auprc["optimized_auprc"].mean())
        metrics["mean_reference_auprc"] = (
            reference_mean if np.isfinite(reference_mean) else None
        )
        metrics["mean_optimized_auprc"] = (
            optimized_mean if np.isfinite(optimized_mean) else None
        )

    artifacts.write_json_atomic(
        output / "drug_disease_validation.json",
        metrics,
    )
    passed = metrics["passed"]
    if not isinstance(passed, bool):
        raise TypeError("validation report has an invalid passed field")
    status = "passed" if passed else "failed"
    mae = metrics["mae"]
    if mae is not None and not isinstance(mae, (int, float)):
        raise TypeError("validation report has an invalid MAE field")
    mae_display = "unavailable" if mae is None else f"{mae:.6f}"
    print(
        f"Validation {status}: {metrics['retained_drugs']} drugs x "
        f"{metrics['retained_diseases']} diseases, "
        f"r={metrics['pearson']}, MAE={mae_display}"
    )
    failures = metrics["failures"]
    if not isinstance(failures, list) or not all(
        isinstance(failure, str) for failure in failures
    ):
        raise TypeError("validation report has an invalid failures field")
    if failures:
        print("Acceptance failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    print(f"Wrote validation artifacts to {output}")
    return 0 if passed or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
