#!/usr/bin/env python3
"""Benchmark complete ANDES CLI workflows on real input artifacts.

``benchmark_optimized.py`` isolates numerical kernels. This benchmark starts a
fresh Python process for every sample and includes input loading, artifact
validation, scoring, provenance generation, and result serialization.

Cold samples start without a null artifact. Warm samples reuse the artifact
produced by the cold sample. The benchmark also builds a
persistent index and verifies that index-to-index and indexed-ranked results
agree with their one-shot counterparts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING = ROOT / "data" / "embedding" / "node2vec_consensus.csv"
DEFAULT_GENES = ROOT / "data" / "embedding" / "consensus_node.txt"
DEFAULT_GENE_SETS = (
    ROOT / "data" / "gene_sets" / "hsa_experimental_eval_BP_propagated.gmt"
)
DEFAULT_RANKING = ROOT / "data" / "expression" / "GSE3467_rank.txt"
_RUSAGE_MARKER = "__ANDES_BENCH_RUSAGE__"
_SUBPROCESS_RUNNER = f"""
import json
import resource
import sys

from andes.cli import main

try:
    exit_code = int(main(sys.argv[1:]) or 0)
finally:
    scale = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(
        {_RUSAGE_MARKER!r}
        + json.dumps({{
            "process_peak_rss_bytes": int(own.ru_maxrss) * scale,
            "child_peak_rss_bytes": int(children.ru_maxrss) * scale,
        }}),
        file=sys.stderr,
    )
raise SystemExit(exit_code)
"""


@dataclass(frozen=True, slots=True)
class CommandSample:
    label: str
    seconds: float
    process_peak_rss_bytes: int
    child_peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class CompareArtifacts:
    output: Path
    cache: Path
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EnrichArtifacts:
    output: Path
    cache: Path
    report: Mapping[str, object]


class RssGate(TypedDict):
    metric: str
    allowed_regression_percent: float
    passed: bool
    regressions: list[str]
    comparisons: dict[str, dict[str, float | int | bool]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        choices=("all", "compare", "enrich"),
        default="all",
        help="workflow to benchmark (default: all)",
    )
    parser.add_argument("--emb", default=str(DEFAULT_EMBEDDING))
    parser.add_argument("--genelist", default=str(DEFAULT_GENES))
    parser.add_argument("--geneset", default=str(DEFAULT_GENE_SETS))
    parser.add_argument("--rankedlist", default=str(DEFAULT_RANKING))
    parser.add_argument("--min", dest="min_size", type=int, default=10)
    parser.add_argument("--max", dest="max_size", type=int, default=300)
    parser.add_argument(
        "--ite",
        type=int,
        default=100,
        help="Monte Carlo iterations for each cold null build (default: 100)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="warm samples per scoring path (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--query-blas-threads", type=int, default=0)
    parser.add_argument(
        "--null-workers",
        type=int,
        default=0,
        help="ranked-null workers; 0 selects a memory-aware count (default: 0)",
    )
    parser.add_argument("--worker-blas-threads", type=int, default=1)
    parser.add_argument("--workspace-mb", type=float, default=128.0)
    parser.add_argument("--null-memory-mb", type=float, default=128.0)
    parser.add_argument(
        "--baseline-json",
        default="",
        help="fail when measured peak RSS regresses against this report",
    )
    parser.add_argument(
        "--max-rss-regression-percent",
        type=float,
        default=5.0,
        help="largest allowed measured peak-RSS increase (default: 5)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="retain benchmark artifacts in a new directory; defaults to /tmp",
    )
    parser.add_argument("--json-out", default="", help="write the JSON report")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print stdout and stderr from every measured command",
    )
    args = parser.parse_args(argv)

    if args.min_size < 1 or args.max_size < args.min_size:
        parser.error("invalid gene-set size range")
    if args.ite < 2:
        parser.error("--ite must be at least 2")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.query_blas_threads < 0:
        parser.error("--query-blas-threads must be non-negative")
    if args.null_workers < 0:
        parser.error("--null-workers must be non-negative")
    if args.worker_blas_threads < 1:
        parser.error("--worker-blas-threads must be positive")
    if args.workspace_mb <= 0:
        parser.error("--workspace-mb must be positive")
    if args.null_memory_mb <= 0:
        parser.error("--null-memory-mb must be positive")
    if args.max_rss_regression_percent < 0:
        parser.error("--max-rss-regression-percent must be non-negative")

    required_paths = ["emb", "genelist", "geneset"]
    if args.workflow in {"all", "enrich"}:
        required_paths.append("rankedlist")
    for attribute in required_paths:
        path = Path(getattr(args, attribute)).expanduser()
        if not path.is_file():
            parser.error(f"--{attribute} does not exist: {path}")
        setattr(args, attribute, str(path.resolve()))

    if args.artifacts_dir:
        artifact_path = Path(args.artifacts_dir).expanduser().resolve()
        if artifact_path.exists():
            parser.error("--artifacts-dir must name a new directory")
        args.artifacts_dir = str(artifact_path)
    if args.json_out:
        args.json_out = str(Path(args.json_out).expanduser().resolve())
    if args.baseline_json:
        baseline = Path(args.baseline_json).expanduser().resolve()
        if not baseline.is_file():
            parser.error(f"--baseline-json does not exist: {baseline}")
        args.baseline_json = str(baseline)
    return args


def _hash_file(path):
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_snapshot(path):
    """Return content and timestamp identity for an immutable artifact."""
    root = Path(path)
    if not root.exists():
        raise AssertionError(f"expected artifact does not exist: {root}")
    files = (
        [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    )
    return tuple(
        (
            file.relative_to(root).as_posix() if root.is_dir() else file.name,
            file.stat().st_size,
            file.stat().st_mtime_ns,
            _hash_file(file),
        )
        for file in files
    )


def _run_cli(label, arguments, *, verbose):
    command = [sys.executable, "-c", _SUBPROCESS_RUNNER, *map(str, arguments)]
    display_command = [
        sys.executable,
        "-m",
        "andes.cli",
        *map(str, arguments),
    ]
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    usage = None
    stderr_lines = []
    for line in completed.stderr.splitlines(keepends=True):
        if line.startswith(_RUSAGE_MARKER):
            usage = json.loads(line[len(_RUSAGE_MARKER) :])
        else:
            stderr_lines.append(line)
    visible_stderr = "".join(stderr_lines)
    if verbose:
        print(f"\n[{label}] {shlex.join(display_command)}")
        if completed.stdout:
            print(completed.stdout, end="")
        if visible_stderr:
            print(visible_stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"command: {shlex.join(display_command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{visible_stderr}"
        )
    if usage is None:
        raise RuntimeError(f"{label} did not report peak RSS")
    return CommandSample(
        label=label,
        seconds=elapsed,
        process_peak_rss_bytes=int(usage["process_peak_rss_bytes"]),
        child_peak_rss_bytes=int(usage["child_peak_rss_bytes"]),
    )


def _read_provenance(output, *, method, engine, companion_paths=()):
    output = Path(output)
    sidecar = output.with_suffix(".metadata.json")
    if not output.is_file() or not sidecar.is_file():
        raise AssertionError(f"missing output or provenance for {output}")
    companions = tuple(Path(path) for path in companion_paths)
    files = (output, *companions)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise AssertionError(f"missing result companion(s): {missing}")
    with sidecar.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "hash": _hash_file(path),
        }
        for path in files
    }
    expected = {
        "manifest_version": 1,
        "method": method,
        "score_engine": engine,
        "output_file": output.name,
        "output_size_bytes": output.stat().st_size,
        "output_hash": _hash_file(output),
        "files": expected_files,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"invalid provenance for {output}: {mismatches}")
    query_seconds = metadata.get("runtime", {}).get("query_seconds")
    if not isinstance(query_seconds, (int, float)) or query_seconds < 0:
        raise AssertionError(f"{output} provenance omits a valid query_seconds value")
    return metadata


def _load_compare_output(output, *, method, engine):
    output = Path(output)
    rows_path = output.with_suffix(".rows.json")
    columns_path = output.with_suffix(".columns.json")
    metadata = _read_provenance(
        output,
        method=method,
        engine=engine,
        companion_paths=(rows_path, columns_path),
    )
    scores = np.load(output, allow_pickle=False)
    with rows_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    with columns_path.open(encoding="utf-8") as handle:
        columns = json.load(handle)
    if scores.dtype != np.float32:
        raise AssertionError(f"{output} has dtype {scores.dtype}; expected float32")
    if scores.shape != (len(rows), len(columns)):
        raise AssertionError(f"{output} labels do not match shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise AssertionError(f"{output} contains non-finite scores")
    return scores, tuple(rows), tuple(columns), metadata


def _load_enrich_output(output, *, engine):
    metadata = _read_provenance(
        output,
        method="andes_ranked",
        engine=engine,
    )
    frame = pd.read_csv(output, index_col=0)
    expected_columns = {
        "size",
        "true_score",
        "null_mu",
        "null_sigma",
        "z_score",
    }
    if set(frame.columns) != expected_columns:
        raise AssertionError(
            f"{output} columns {set(frame.columns)} != {expected_columns}"
        )
    if not frame.index.is_unique:
        raise AssertionError(f"{output} contains duplicate terms")
    if frame.empty or not np.all(np.isfinite(frame.to_numpy())):
        raise AssertionError(f"{output} is empty or contains non-finite values")
    return frame, metadata


def _seconds(samples):
    values = [sample.seconds for sample in samples]
    return {
        "median": statistics.median(values),
        "samples": values,
        "process_peak_rss_bytes": [sample.process_peak_rss_bytes for sample in samples],
        "child_peak_rss_bytes": [sample.child_peak_rss_bytes for sample in samples],
    }


def _measured_rss_metrics(report):
    """Return comparable CLI-process peak RSS from an end-to-end report."""
    metrics = {}

    def add(label, process):
        metrics[label] = int(process)

    def add_repeated(label, summary):
        add(
            label,
            statistics.median(summary["process_peak_rss_bytes"]),
        )

    index_report = report["artifacts"]["index"]
    add(
        "index.build",
        index_report["build_process_peak_rss_bytes"],
    )
    add_repeated("index.metadata_load", index_report["metadata_load"])
    add_repeated("index.full_audit", index_report["full_audit_load"])

    for workflow_name, workflow in report["workflows"].items():
        add(
            f"{workflow_name}.cold",
            workflow["cold_process_peak_rss_bytes"],
        )
        add_repeated(f"{workflow_name}.warm", workflow["warm_seconds"])
        add_repeated(f"{workflow_name}.indexed", workflow["indexed"])
    return metrics


def _assert_comparable_reports(current, baseline):
    """Reject RSS comparisons across different workloads or runtimes."""
    config_keys = (
        "workflow",
        "embedding",
        "gene_list",
        "gene_sets",
        "ranked_list",
        "min_size",
        "max_size",
        "null_iterations",
        "repeats",
        "seed",
        "query_blas_threads",
        "null_workers",
        "worker_blas_threads",
        "workspace_mb",
        "null_memory_mb",
    )
    system_keys = (
        "python",
        "numpy",
        "scipy",
        "platform",
        "processor",
        "logical_cpu_count",
        "blas_pools",
    )
    differences = [
        f"config.{key}"
        for key in config_keys
        if current["config"].get(key) != baseline["config"].get(key)
    ]
    differences.extend(
        f"system.{key}"
        for key in system_keys
        if current["system"].get(key) != baseline["system"].get(key)
    )
    if differences:
        raise ValueError(
            "RSS baseline is not comparable; differing fields: "
            + ", ".join(differences)
        )


def _rss_gate(current, baseline, allowed_percent) -> RssGate:
    """Compare measured subprocess RSS and describe any regressions."""
    _assert_comparable_reports(current, baseline)
    current_metrics = _measured_rss_metrics(current)
    baseline_metrics = _measured_rss_metrics(baseline)
    if current_metrics.keys() != baseline_metrics.keys():
        raise ValueError("RSS baseline does not contain the same benchmark stages")

    comparisons: dict[str, dict[str, float | int | bool]] = {}
    regressions: list[str] = []
    factor = 1.0 + float(allowed_percent) / 100.0
    for label, current_bytes in current_metrics.items():
        baseline_bytes = baseline_metrics[label]
        allowed_bytes = int(baseline_bytes * factor)
        passed = current_bytes <= allowed_bytes
        comparisons[label] = {
            "baseline_bytes": baseline_bytes,
            "current_bytes": current_bytes,
            "allowed_bytes": allowed_bytes,
            "change_percent": (
                100.0 * (current_bytes - baseline_bytes) / baseline_bytes
                if baseline_bytes
                else 0.0
            ),
            "passed": passed,
        }
        if not passed:
            regressions.append(label)
    return {
        "metric": "measured_cli_process_peak_rss",
        "allowed_regression_percent": float(allowed_percent),
        "passed": not regressions,
        "regressions": regressions,
        "comparisons": comparisons,
    }


def _base_compare_arguments(args, output, cache, *, cache_policy="build"):
    return [
        "compare",
        "--emb",
        args.emb,
        "--genelist",
        args.genelist,
        "--geneset1",
        args.geneset,
        "--geneset2",
        args.geneset,
        "--out",
        output,
        "--cache",
        cache,
        "--cache-policy",
        cache_policy,
        "--min",
        args.min_size,
        "--max",
        args.max_size,
        "--ite",
        args.ite,
        "--seed",
        args.seed,
        "--query-blas-threads",
        args.query_blas_threads,
        "--null-blas-threads",
        args.worker_blas_threads,
        "--query-memory-mb",
        args.workspace_mb,
    ]


def _benchmark_compare_one_shot(args, work):
    cache = work / "compare.null"
    cold_output = work / "compare_cold.npy"
    cold = _run_cli(
        "compare cold",
        _base_compare_arguments(
            args,
            cold_output,
            cache,
            cache_policy="build",
        ),
        verbose=args.verbose,
    )
    cold_scores, rows, columns, provenance = _load_compare_output(
        cold_output,
        method="andes_bma",
        engine="bestmatch",
    )
    cache_snapshot = _artifact_snapshot(cache)

    warm_samples = []
    for repeat in range(args.repeats):
        output = work / f"compare_warm_{repeat + 1}.npy"
        arguments = _base_compare_arguments(
            args,
            output,
            cache,
            cache_policy="require",
        )
        warm_samples.append(
            _run_cli(
                f"compare warm {repeat + 1}",
                arguments,
                verbose=args.verbose,
            )
        )
        warm_scores, warm_rows, warm_columns, _ = _load_compare_output(
            output,
            method="andes_bma",
            engine="bestmatch",
        )
        if warm_rows != rows or warm_columns != columns:
            raise AssertionError("warm comparison term axes changed")
        np.testing.assert_allclose(warm_scores, cold_scores, rtol=1e-6, atol=1e-6)
        if _artifact_snapshot(cache) != cache_snapshot:
            raise AssertionError("warm comparison mutated its null artifact")

    report = {
        "terms": len(rows),
        "matrix_entries": int(cold_scores.size),
        "output_bytes": cold_output.stat().st_size,
        "cold_seconds": cold.seconds,
        "cold_process_peak_rss_bytes": cold.process_peak_rss_bytes,
        "cold_child_peak_rss_bytes": cold.child_peak_rss_bytes,
        "warm_seconds": _seconds(warm_samples),
        "cold_to_warm_ratio": cold.seconds
        / statistics.median(sample.seconds for sample in warm_samples),
        "null_spec": provenance["null_spec"],
        "runtime": provenance["runtime"],
    }
    return CompareArtifacts(cold_output, cache, report)


def _base_enrich_arguments(args, output, cache, *, cache_policy="build"):
    return [
        "enrich",
        "--emb",
        args.emb,
        "--genelist",
        args.genelist,
        "--geneset",
        args.geneset,
        "--rankedlist",
        args.rankedlist,
        "--out",
        output,
        "--cache",
        cache,
        "--cache-policy",
        cache_policy,
        "--min",
        args.min_size,
        "--max",
        args.max_size,
        "--ite",
        args.ite,
        "--seed",
        args.seed,
        "--workers",
        args.null_workers,
        "--query-blas-threads",
        args.query_blas_threads,
        "--worker-blas-threads",
        args.worker_blas_threads,
        "--workspace-mb",
        args.workspace_mb,
        "--null-memory-mb",
        args.null_memory_mb,
    ]


def _benchmark_enrich_one_shot(args, work):
    cache = work / "ranked.null"
    cold_output = work / "enrich_cold.csv"
    cold = _run_cli(
        "enrich cold",
        _base_enrich_arguments(
            args,
            cold_output,
            cache,
            cache_policy="build",
        ),
        verbose=args.verbose,
    )
    cold_frame, provenance = _load_enrich_output(cold_output, engine="bestmatch")
    cache_snapshot = _artifact_snapshot(cache)

    warm_samples = []
    for repeat in range(args.repeats):
        output = work / f"enrich_warm_{repeat + 1}.csv"
        warm_samples.append(
            _run_cli(
                f"enrich warm {repeat + 1}",
                _base_enrich_arguments(
                    args,
                    output,
                    cache,
                    cache_policy="require",
                ),
                verbose=args.verbose,
            )
        )
        warm_frame, _ = _load_enrich_output(output, engine="bestmatch")
        pd.testing.assert_frame_equal(
            warm_frame,
            cold_frame,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        if _artifact_snapshot(cache) != cache_snapshot:
            raise AssertionError("warm enrichment mutated its null artifact")

    report = {
        "terms": len(cold_frame),
        "ranked_genes": provenance["extra"]["ranked_genes"],
        "output_bytes": cold_output.stat().st_size,
        "cold_seconds": cold.seconds,
        "cold_process_peak_rss_bytes": cold.process_peak_rss_bytes,
        "cold_child_peak_rss_bytes": cold.child_peak_rss_bytes,
        "warm_seconds": _seconds(warm_samples),
        "cold_to_warm_ratio": cold.seconds
        / statistics.median(sample.seconds for sample in warm_samples),
        "null_spec": provenance["null_spec"],
        "runtime": provenance["runtime"],
    }
    return EnrichArtifacts(cold_output, cache, report)


def _build_index(args, work):
    index_dir = work / "index"
    sample = _run_cli(
        "index build",
        [
            "index",
            "build",
            "--emb",
            args.emb,
            "--genelist",
            args.genelist,
            "--geneset",
            args.geneset,
            "--out",
            index_dir,
            "--min",
            args.min_size,
            "--max",
            args.max_size,
            "--query-memory-mb",
            args.workspace_mb,
            "--query-blas-threads",
            args.query_blas_threads,
        ],
        verbose=args.verbose,
    )
    metadata_path = index_dir / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    bestmatch = np.load(index_dir / "bestmatch.npy", mmap_mode="r")
    expected_shape = (metadata["n_genes"], metadata["n_terms"])
    if bestmatch.shape != expected_shape or bestmatch.dtype != np.float32:
        raise AssertionError(
            f"index bestmatch {bestmatch.shape}/{bestmatch.dtype} != "
            f"{expected_shape}/float32"
        )
    del bestmatch
    metadata_loads = [
        _run_cli(
            f"index metadata load {repeat + 1}",
            ["index", "verify", "--index", index_dir],
            verbose=args.verbose,
        )
        for repeat in range(args.repeats)
    ]
    full_audits = [
        _run_cli(
            f"index full audit {repeat + 1}",
            ["index", "verify", "--index", index_dir, "--full"],
            verbose=args.verbose,
        )
        for repeat in range(args.repeats)
    ]
    return (
        index_dir,
        _artifact_snapshot(index_dir),
        {
            "build_seconds": sample.seconds,
            "build_process_peak_rss_bytes": sample.process_peak_rss_bytes,
            "build_child_peak_rss_bytes": sample.child_peak_rss_bytes,
            "numerical_construction_seconds": metadata[
                "bestmatch_construction_seconds"
            ],
            "payload_build_seconds": metadata["payload_build_seconds"],
            "non_payload_cli_and_publication_seconds": max(
                0.0,
                sample.seconds - metadata["payload_build_seconds"],
            ),
            "metadata_load": _seconds(metadata_loads),
            "full_audit_load": _seconds(full_audits),
            "genes": metadata["n_genes"],
            "terms": metadata["n_terms"],
            "bestmatch_mb": metadata["bestmatch_mb"],
            "build_workspace_mb": metadata["chunk_workspace_mb"],
        },
    )


def _benchmark_index_compare(
    args,
    work,
    index_dir,
    index_snapshot,
    one_shot,
):
    expected, expected_rows, expected_columns, _ = _load_compare_output(
        one_shot.output,
        method="andes_bma",
        engine="bestmatch",
    )
    cache_snapshot = _artifact_snapshot(one_shot.cache)
    samples = []
    max_abs_difference = 0.0
    for repeat in range(args.repeats):
        output = work / f"compare_indexed_{repeat + 1}.npy"
        samples.append(
            _run_cli(
                f"compare indexed {repeat + 1}",
                [
                    "index",
                    "compare",
                    "--index1",
                    index_dir,
                    "--index2",
                    index_dir,
                    "--out",
                    output,
                    "--cache",
                    one_shot.cache,
                    "--ite",
                    args.ite,
                    "--seed",
                    args.seed,
                    "--cache-policy",
                    "require",
                    "--query-blas-threads",
                    args.query_blas_threads,
                    "--query-memory-mb",
                    args.workspace_mb,
                    "--null-blas-threads",
                    args.worker_blas_threads,
                ],
                verbose=args.verbose,
            )
        )
        indexed, rows, columns, _ = _load_compare_output(
            output,
            method="andes_index_compare",
            engine="index_to_index",
        )
        if rows != expected_rows or columns != expected_columns:
            raise AssertionError("indexed comparison term axes changed")
        np.testing.assert_allclose(indexed, expected, rtol=1e-5, atol=1e-5)
        max_abs_difference = max(
            max_abs_difference,
            float(np.max(np.abs(indexed - expected))),
        )
        if _artifact_snapshot(one_shot.cache) != cache_snapshot:
            raise AssertionError("indexed comparison mutated its null artifact")
        if _artifact_snapshot(index_dir) != index_snapshot:
            raise AssertionError("indexed comparison mutated the persistent index")

    timing = _seconds(samples)
    return {
        **timing,
        "speedup_vs_warm_one_shot": (
            one_shot.report["warm_seconds"]["median"] / timing["median"]
        ),
        "max_abs_difference": max_abs_difference,
    }


def _benchmark_index_enrich(
    args,
    work,
    index_dir,
    index_snapshot,
    one_shot,
):
    cache_snapshot = _artifact_snapshot(one_shot.cache)
    expected, _ = _load_enrich_output(one_shot.output, engine="bestmatch")
    samples = []
    maximum_signed_difference = 0.0

    for repeat in range(args.repeats):
        output = work / f"enrich_indexed_{repeat + 1}.csv"
        samples.append(
            _run_cli(
                f"enrich indexed {repeat + 1}",
                [
                    "enrich",
                    "--index",
                    index_dir,
                    "--rankedlist",
                    args.rankedlist,
                    "--out",
                    output,
                    "--cache",
                    one_shot.cache,
                    "--ite",
                    args.ite,
                    "--seed",
                    args.seed,
                    "--cache-policy",
                    "require",
                    "--query-blas-threads",
                    args.query_blas_threads,
                    "--worker-blas-threads",
                    args.worker_blas_threads,
                    "--workspace-mb",
                    args.workspace_mb,
                    "--null-memory-mb",
                    args.null_memory_mb,
                ],
                verbose=args.verbose,
            )
        )
        observed, _ = _load_enrich_output(output, engine="indexed")
        pd.testing.assert_index_equal(observed.index, expected.index)
        expected_true = expected["true_score"].to_numpy()
        observed_true = observed["true_score"].to_numpy()
        np.testing.assert_allclose(
            observed_true,
            expected_true,
            rtol=2e-5,
            atol=2e-5,
        )
        maximum_signed_difference = max(
            maximum_signed_difference,
            float(np.max(np.abs(observed_true - expected_true))),
        )
        if _artifact_snapshot(one_shot.cache) != cache_snapshot:
            raise AssertionError("indexed enrichment mutated its null artifact")
        if _artifact_snapshot(index_dir) != index_snapshot:
            raise AssertionError("indexed enrichment mutated the persistent index")

    timing = _seconds(samples)
    return {
        **timing,
        "speedup_vs_warm_standalone": (
            one_shot.report["warm_seconds"]["median"] / timing["median"]
        ),
        "true_score_signed_max_abs_difference": maximum_signed_difference,
    }


@contextmanager
def _benchmark_directory(requested):
    if requested:
        directory = Path(requested)
        directory.mkdir(parents=True)
        yield directory
        return
    with tempfile.TemporaryDirectory(prefix="andes-e2e-") as temporary:
        yield Path(temporary)


def _write_json_atomic(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def main(argv=None):
    args = parse_args(argv)
    run_compare = args.workflow in {"all", "compare"}
    run_enrich = args.workflow in {"all", "enrich"}

    with _benchmark_directory(args.artifacts_dir) as work:
        compare_artifacts = (
            _benchmark_compare_one_shot(args, work) if run_compare else None
        )
        enrich_artifacts = (
            _benchmark_enrich_one_shot(args, work) if run_enrich else None
        )
        index_dir, index_snapshot, index_report = _build_index(args, work)

        workflows = {}
        if compare_artifacts is not None:
            compare_report = dict(compare_artifacts.report)
            compare_report["indexed"] = _benchmark_index_compare(
                args,
                work,
                index_dir,
                index_snapshot,
                compare_artifacts,
            )
            workflows["compare"] = compare_report
        if enrich_artifacts is not None:
            enrich_report = dict(enrich_artifacts.report)
            enrich_report["indexed"] = _benchmark_index_enrich(
                args,
                work,
                index_dir,
                index_snapshot,
                enrich_artifacts,
            )
            workflows["enrich"] = enrich_report

        report: dict[str, object] = {
            "config": {
                "workflow": args.workflow,
                "embedding": args.emb,
                "gene_list": args.genelist,
                "gene_sets": args.geneset,
                "ranked_list": args.rankedlist,
                "min_size": args.min_size,
                "max_size": args.max_size,
                "null_iterations": args.ite,
                "repeats": args.repeats,
                "seed": args.seed,
                "query_blas_threads": args.query_blas_threads,
                "null_workers": args.null_workers,
                "worker_blas_threads": args.worker_blas_threads,
                "workspace_mb": args.workspace_mb,
                "null_memory_mb": args.null_memory_mb,
            },
            "system": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
                "logical_cpu_count": os.cpu_count(),
                "blas_pools": [
                    {
                        "internal_api": pool.get("internal_api", "unknown"),
                        "prefix": pool.get("prefix", "unknown"),
                        "num_threads": int(pool.get("num_threads", 0)),
                        "version": pool.get("version", "unknown"),
                    }
                    for pool in threadpool_info()
                    if pool.get("user_api") == "blas"
                ],
            },
            "artifacts": {
                "retained": bool(args.artifacts_dir),
                "directory": str(work) if args.artifacts_dir else None,
                "index": index_report,
            },
            "workflows": workflows,
        }
        rss_gate = None
        if args.baseline_json:
            with Path(args.baseline_json).open(encoding="utf-8") as handle:
                baseline = json.load(handle)
            rss_gate = _rss_gate(
                report,
                baseline,
                args.max_rss_regression_percent,
            )
            report["rss_gate"] = rss_gate
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.json_out:
            _write_json_atomic(args.json_out, report)
            print(f"\nWrote JSON report to {args.json_out}")
    if rss_gate is not None and not rss_gate["passed"]:
        print(
            "\nPeak-RSS regression gate failed: " + ", ".join(rss_gate["regressions"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
