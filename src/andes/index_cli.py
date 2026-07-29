"""Command-line orchestration for persistent ANDES indexes."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import application, artifacts
from . import data as load_data
from . import index as index_core
from .nulls import BmaNullModel, null_cache_dir, resolve_bma_null
from .provenance import RunProvenance, write_result_sidecar
from .runtime import blas_runtime_info, format_blas_runtime, query_blas_context
from .scoring import ScoreResult

FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]
OrderArray: TypeAlias = NDArray[np.intp]
BlasPool: TypeAlias = Mapping[str, object]
CommandName: TypeAlias = Literal[
    "build",
    "verify",
    "query",
    "batch-query",
    "compare",
]


def _missing_command(_args: argparse.Namespace) -> None:
    raise RuntimeError("index command parser did not install a handler")


class IndexArgs(argparse.Namespace):
    """Typed boundary around values populated by ``argparse``."""

    cmd: CommandName = "build"
    func: Callable[[IndexArgs], None] = staticmethod(_missing_command)

    emb: str = ""
    genelist: str = ""
    geneset: str = ""
    out: str = ""
    min_size: int = 10
    max_size: int = 300
    query_memory_mb: float = 128.0
    query_blas_threads: int = 0
    overwrite: bool = False

    index: str = ""
    full: bool = False
    genes: str | None = None
    term: str | None = None
    queries: str = ""
    top_k: int = 0
    cache: str = ""
    cache_root: str = ""
    cache_policy: Literal["build", "require"] = "require"
    ite: int = 1000
    seed: int = 12345
    null_blas_threads: int = 1
    no_zscore: bool = False
    no_mmap: bool = False

    index1: str = ""
    index2: str = ""


def _null_payload(model: BmaNullModel | None) -> dict[str, object] | None:
    return None if model is None else model.spec.to_dict()


def _runtime_payload(
    args: IndexArgs,
    pools: Sequence[BlasPool],
    null_pools: Sequence[BlasPool] = (),
    *,
    null_blas_threads: int | None = None,
) -> dict[str, object]:
    requested = int(args.query_blas_threads)
    return {
        "query_blas_threads_requested": requested,
        "query_blas_threads_limit": None if requested == 0 else requested,
        "null_blas_threads": null_blas_threads,
        "blas_pools": list(pools),
        "null_blas_pools": list(null_pools),
    }


def _print_blas(
    label: str,
    requested_threads: int,
    pools: Sequence[BlasPool],
) -> None:
    requested = "runtime default" if requested_threads == 0 else str(requested_threads)
    print(f"{label} BLAS: {requested} ({format_blas_runtime(pools)})")


def _query_fingerprint(query_indices: IntArray) -> str:
    return artifacts.combine_fingerprints(
        "andes_query_v1",
        {"indices": artifacts.hash_array(np.asarray(query_indices, dtype=np.int32))},
    )


def _load_query_genes(path: str | Path) -> list[str]:
    genes: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tokens = line.split("\t")
            if len(tokens) >= 3:
                genes.extend(token for token in tokens[2:] if token)
            else:
                genes.extend(line.split())
    return genes


def _load_query_sets(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Load one query set per GMT row, preserving file order."""
    names: list[str] = []
    gene_sets: list[list[str]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            tokens = line.split("\t")
            if len(tokens) < 3:
                raise ValueError(
                    f"{path}:{line_number}: batch queries must use GMT format "
                    + "(name, description, genes...)"
                )
            name = tokens[0].strip()
            genes = [gene for gene in tokens[2:] if gene]
            if not name:
                raise ValueError(f"{path}:{line_number}: query name is empty")
            if name in seen:
                raise ValueError(f"{path}:{line_number}: duplicate query name {name!r}")
            if not genes:
                raise ValueError(f"{path}:{line_number}: query {name!r} has no genes")
            seen.add(name)
            names.append(name)
            gene_sets.append(genes)
    if not names:
        raise ValueError(f"{path}: no query sets found")
    return names, gene_sets


def _ranked_order(values: FloatArray, top_k: int = 0) -> OrderArray:
    order = np.asarray(np.argsort(-values, kind="stable"), dtype=np.intp)
    if top_k > 0:
        order = order[:top_k]
    return order


def _write_query_results(
    out_path: str | Path,
    index: index_core.AndesIndex,
    true_scores: FloatArray,
    *,
    zscores: FloatArray | None = None,
    top_k: int = 0,
) -> pd.DataFrame:
    sort_values = true_scores if zscores is None else zscores
    order = _ranked_order(sort_values, top_k)
    term_positions = cast(list[int], order.tolist())
    data: dict[str, object] = {
        "term": [index.terms[position] for position in term_positions],
        "size": index.sizes[order],
        "true_score": true_scores[order],
    }
    if zscores is not None:
        data["z_score"] = zscores[order]
    frame = pd.DataFrame(data)
    application.write_dataframe_atomic(frame, out_path, index=False)
    return frame


def _write_batch_query_results(
    out_path: str | Path,
    index: index_core.AndesIndex,
    query_names: Sequence[str],
    true_scores: FloatArray,
    *,
    zscores: FloatArray | None = None,
    top_k: int = 0,
) -> pd.DataFrame:
    expected_shape = (len(query_names), len(index.terms))
    if true_scores.shape != expected_shape:
        raise ValueError(
            f"true_scores has shape {true_scores.shape}; expected {expected_shape}"
        )
    if zscores is not None and zscores.shape != expected_shape:
        raise ValueError(
            f"zscores has shape {zscores.shape}; expected {expected_shape}"
        )

    frames: list[pd.DataFrame] = []
    for query_position, name in enumerate(query_names):
        sort_values = np.asarray(
            true_scores[query_position] if zscores is None else zscores[query_position],
            dtype=np.float32,
        )
        order = _ranked_order(sort_values, top_k)
        term_positions = cast(list[int], order.tolist())
        data: dict[str, object] = {
            "query": [name] * len(order),
            "term": [index.terms[position] for position in term_positions],
            "size": index.sizes[order],
            "true_score": true_scores[query_position, order],
        }
        if zscores is not None:
            data["z_score"] = zscores[query_position, order]
        frames.append(pd.DataFrame(data))

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    application.write_dataframe_atomic(frame, out_path, index=False)
    return frame


def _write_comparison_results(
    out_path: str | Path,
    left: index_core.AndesIndex,
    right: index_core.AndesIndex,
    true_scores: FloatArray,
    *,
    zscores: FloatArray | None = None,
) -> None:
    out_path = Path(out_path)
    values = np.asarray(
        true_scores if zscores is None else zscores,
        dtype=np.float32,
    )
    expected_shape = (len(left.terms), len(right.terms))
    if values.shape != expected_shape:
        raise ValueError(
            f"comparison matrix has shape {values.shape}; expected {expected_shape}"
        )
    if out_path.suffix.lower() == ".npy":
        artifacts.save_npy_atomic(out_path, values)
        artifacts.write_json_atomic(out_path.with_suffix(".rows.json"), left.terms)
        artifacts.write_json_atomic(out_path.with_suffix(".columns.json"), right.terms)
        return

    frame = pd.DataFrame(values, index=left.terms, columns=right.terms)
    application.write_dataframe_atomic(frame, out_path, index_label="term")


def cmd_build(args: IndexArgs) -> None:
    embedding = load_data.load_embedding_space(args.emb, args.genelist)
    database = load_data.load_gene_set_database(
        args.geneset,
        embedding,
        min_size=args.min_size,
        max_size=args.max_size,
        sort_terms=True,
    )
    print(
        f"Building ANDES index: {len(embedding.genes)} genes x "
        + f"{len(database.terms)} terms"
    )
    with query_blas_context(args.query_blas_threads):
        pools = blas_runtime_info()
        _print_blas("Query", args.query_blas_threads, pools)
        metadata = index_core.build_andes_index(
            embedding,
            database,
            args.out,
            max_workspace_mb=args.query_memory_mb,
            overwrite=args.overwrite,
            runtime=_runtime_payload(args, pools),
        )
    print(f"Wrote index to {args.out}")
    bestmatch_mb = cast(float, metadata["bestmatch_mb"])
    chunk_workspace_mb = cast(float, metadata["chunk_workspace_mb"])
    print(f"Best-match matrix: {bestmatch_mb:.1f} MB")
    print(f"Peak temporary build chunk: ~{chunk_workspace_mb:.1f} MB")


def cmd_verify(args: IndexArgs) -> None:
    mode = "full" if args.full else "metadata"
    started = time.perf_counter()
    index = index_core.load_andes_index(args.index, mmap=True, verify=mode)
    elapsed = time.perf_counter() - started
    print(
        f"Verified {args.index} ({mode}): {len(index.gene_list)} genes, "
        + f"{len(index.terms)} terms in {elapsed:.2f}s"
    )


def _resolve_index_null(
    left: index_core.AndesIndex,
    right: index_core.AndesIndex,
    row_sizes: Sequence[int] | IntArray,
    column_sizes: Sequence[int] | IntArray,
    args: IndexArgs,
) -> tuple[BmaNullModel | None, Sequence[BlasPool], dict[str, object]]:
    if args.no_zscore:
        return (
            None,
            [],
            {
                "null_cache_policy": "disabled",
                "null_cache_built": False,
                "null_cache_added_entries": 0,
                "null_cache_resolution_seconds": 0.0,
            },
        )
    started = time.perf_counter()
    with query_blas_context(args.null_blas_threads):
        pools = blas_runtime_info()
        _print_blas("Null", args.null_blas_threads, pools)
        resolved = resolve_bma_null(
            left.embedding_space(),
            left.background,
            right.background,
            row_sizes,
            column_sizes,
            path=args.cache or None,
            base_dir=(
                None if args.cache else null_cache_dir("bma", args.cache_root or None)
            ),
            iterations=args.ite,
            seed=args.seed,
            no_build=args.cache_policy == "require",
            blas_threads=args.null_blas_threads,
        )
        if not isinstance(resolved.model, BmaNullModel):
            raise TypeError("BMA null resolution returned a ranked null model")
        model = resolved.model
    elapsed = time.perf_counter() - started
    return (
        model,
        pools,
        {
            "null_cache_policy": args.cache_policy,
            "null_cache_built": bool(resolved.built),
            "null_cache_added_entries": int(resolved.added_entries),
            "null_cache_resolution_seconds": elapsed,
        },
    )


def _query_null(
    index: index_core.AndesIndex,
    query_sizes: Sequence[int] | IntArray,
    args: IndexArgs,
) -> tuple[BmaNullModel | None, Sequence[BlasPool], dict[str, object]]:
    return _resolve_index_null(
        index,
        index,
        query_sizes,
        index.sizes,
        args,
    )


def _calibrated_values(
    exact: ScoreResult,
    row_sizes: Sequence[int] | IntArray,
    left: index_core.AndesIndex,
    right: index_core.AndesIndex,
    null_model: BmaNullModel | None,
) -> FloatArray | None:
    if null_model is None:
        return None
    calibrated = index_core.calibrate_index_scores(
        exact,
        np.asarray(row_sizes, dtype=np.int32),
        left,
        right,
        null_model,
    )
    return calibrated.scores


def cmd_query(args: IndexArgs) -> None:
    load_started = time.perf_counter()
    index = index_core.load_andes_index(args.index, mmap=not args.no_mmap)
    index_load_seconds = time.perf_counter() - load_started
    if args.term is not None:
        if args.term not in index.term_to_index:
            raise ValueError(f"term {args.term!r} is not present in the index")
        query_idx = index.members_for(args.term)
        missing: list[str] = []
        print(f"Indexed-term query: {args.term} ({len(query_idx)} genes)")
    else:
        if args.genes is None:
            raise RuntimeError("query requires --genes or --term")
        genes = _load_query_genes(args.genes)
        query_idx, missing = index.map_genes(genes)
        print(
            f"Query genes: {len(query_idx)} matched"
            + (f", {len(missing)} missing" if missing else "")
        )

    if not args.no_zscore:
        index.validate_query_background(query_idx)
    null_model, null_pools, cache_metrics = _query_null(
        index,
        [len(query_idx)],
        args,
    )

    query_started = time.perf_counter()
    with query_blas_context(args.query_blas_threads):
        pools = blas_runtime_info()
        _print_blas("Query", args.query_blas_threads, pools)
        if args.term is not None:
            exact = index.score_indexed_term(
                args.term,
                max_workspace_mb=args.query_memory_mb,
            )
        else:
            exact = index.score_query(
                query_idx,
                max_workspace_mb=args.query_memory_mb,
            )
    query_seconds = time.perf_counter() - query_started
    runtime = _runtime_payload(
        args,
        pools,
        null_pools,
        null_blas_threads=args.null_blas_threads,
    )
    runtime.update(cache_metrics)
    runtime["index_load_seconds"] = index_load_seconds
    runtime["query_seconds"] = query_seconds
    zscores = _calibrated_values(
        exact,
        [len(query_idx)],
        index,
        index,
        null_model,
    )
    frame = _write_query_results(
        args.out,
        index,
        exact.scores,
        zscores=zscores,
        top_k=args.top_k,
    )
    provenance = RunProvenance(
        method="andes_index_query",
        score_engine=exact.stats.engine,
        score_kind="z_score" if zscores is not None else "true_score",
        similarity_dtype="float32",
        score_accumulator_dtype="float64_forward_float32_reverse",
        null_accumulator_dtype=(
            "float64" if null_model is not None else "not_applicable"
        ),
        output_dtype="float32",
        tie_policy="not_applicable",
        embedding_fingerprint=index.embedding_fingerprint,
        left_database_fingerprint=_query_fingerprint(query_idx),
        right_database_fingerprint=index.database_fingerprint,
        null_spec=_null_payload(null_model),
        runtime=runtime,
        extra={
            "query_size": len(query_idx),
            "top_k": int(args.top_k),
            "missing_genes": len(missing),
            "background_policy": index.background_policy,
            "workspace_bytes": int(exact.stats.workspace_bytes),
            **dict(exact.stats.details),
        },
    )
    column_dtypes = {
        "term": "object",
        "size": "int32",
        "true_score": "float32",
    }
    if zscores is not None:
        column_dtypes["z_score"] = "float32"
    provenance = provenance.for_tabular_output(column_dtypes)
    _ = write_result_sidecar(args.out, provenance)
    print(f"Wrote {len(frame)} results to {args.out}")


def cmd_batch_query(args: IndexArgs) -> None:
    load_started = time.perf_counter()
    index = index_core.load_andes_index(args.index, mmap=not args.no_mmap)
    index_load_seconds = time.perf_counter() - load_started
    query_names, gene_sets = _load_query_sets(args.queries)
    query_indices: list[IntArray] = []
    missing_total = 0
    for name, genes in zip(query_names, gene_sets, strict=True):
        try:
            indices, missing = index.map_genes(genes)
        except ValueError as exc:
            raise ValueError(f"query {name!r}: {exc}") from exc
        query_indices.append(indices)
        missing_total += len(missing)
    if not args.no_zscore:
        for indices in query_indices:
            index.validate_query_background(indices)

    query_sizes = np.asarray(
        [len(indices) for indices in query_indices], dtype=np.int32
    )
    null_model, null_pools, cache_metrics = _query_null(
        index,
        query_sizes,
        args,
    )
    query_started = time.perf_counter()
    with query_blas_context(args.query_blas_threads):
        pools = blas_runtime_info()
        _print_blas("Query", args.query_blas_threads, pools)
        exact = index.score_queries(
            query_indices,
            max_workspace_mb=args.query_memory_mb,
        )
    query_seconds = time.perf_counter() - query_started
    runtime = _runtime_payload(
        args,
        pools,
        null_pools,
        null_blas_threads=args.null_blas_threads,
    )
    runtime.update(cache_metrics)
    runtime["index_load_seconds"] = index_load_seconds
    runtime["query_seconds"] = query_seconds
    zscores = _calibrated_values(
        exact,
        query_sizes,
        index,
        index,
        null_model,
    )
    frame = _write_batch_query_results(
        args.out,
        index,
        query_names,
        exact.scores,
        zscores=zscores,
        top_k=args.top_k,
    )
    batch_fingerprint = artifacts.combine_fingerprints(
        "andes_batch_query_v1",
        {
            str(position): artifacts.hash_array(indices)
            for position, indices in enumerate(query_indices)
        },
    )
    provenance = RunProvenance(
        method="andes_index_batch_query",
        score_engine=exact.stats.engine,
        score_kind="z_score" if zscores is not None else "true_score",
        similarity_dtype="float32",
        score_accumulator_dtype="float64_forward_float32_reverse",
        null_accumulator_dtype=(
            "float64" if null_model is not None else "not_applicable"
        ),
        output_dtype="float32",
        tie_policy="not_applicable",
        embedding_fingerprint=index.embedding_fingerprint,
        left_database_fingerprint=batch_fingerprint,
        right_database_fingerprint=index.database_fingerprint,
        null_spec=_null_payload(null_model),
        runtime=runtime,
        extra={
            "queries": len(query_names),
            "top_k": int(args.top_k),
            "missing_genes": int(missing_total),
            "background_policy": index.background_policy,
            "workspace_bytes": int(exact.stats.workspace_bytes),
            **dict(exact.stats.details),
        },
    )
    column_dtypes = {
        "query": "object",
        "term": "object",
        "size": "int32",
        "true_score": "float32",
    }
    if zscores is not None:
        column_dtypes["z_score"] = "float32"
    provenance = provenance.for_tabular_output(column_dtypes)
    _ = write_result_sidecar(args.out, provenance)
    print(f"Wrote {len(frame)} results to {args.out}")


def cmd_compare(args: IndexArgs) -> None:
    load_started = time.perf_counter()
    left = index_core.load_andes_index(args.index1, mmap=not args.no_mmap)
    if Path(args.index1).resolve() == Path(args.index2).resolve():
        right = left
    else:
        right = index_core.load_andes_index(args.index2, mmap=not args.no_mmap)
    index_load_seconds = time.perf_counter() - load_started
    left.assert_compatible_with(right)
    null_model, null_pools, cache_metrics = _resolve_index_null(
        left,
        right,
        left.sizes,
        right.sizes,
        args,
    )
    query_started = time.perf_counter()
    with query_blas_context(args.query_blas_threads):
        pools = blas_runtime_info()
        _print_blas("Query", args.query_blas_threads, pools)
        exact = left.compare(
            right,
            max_workspace_mb=args.query_memory_mb,
        )
    query_seconds = time.perf_counter() - query_started
    runtime = _runtime_payload(
        args,
        pools,
        null_pools,
        null_blas_threads=args.null_blas_threads,
    )
    runtime.update(cache_metrics)
    runtime["index_load_seconds"] = index_load_seconds
    runtime["query_seconds"] = query_seconds
    zscores = _calibrated_values(
        exact,
        left.sizes,
        left,
        right,
        null_model,
    )
    _write_comparison_results(
        args.out,
        left,
        right,
        exact.scores,
        zscores=zscores,
    )
    provenance = RunProvenance(
        method="andes_index_compare",
        score_engine=exact.stats.engine,
        score_kind="z_score" if zscores is not None else "true_score",
        similarity_dtype="float32",
        score_accumulator_dtype="float32",
        null_accumulator_dtype=(
            "float64" if null_model is not None else "not_applicable"
        ),
        output_dtype="float32",
        tie_policy="not_applicable",
        embedding_fingerprint=left.embedding_fingerprint,
        left_database_fingerprint=left.database_fingerprint,
        right_database_fingerprint=right.database_fingerprint,
        symmetric_reuse=exact.stats.symmetric_reuse,
        null_spec=_null_payload(null_model),
        runtime=runtime,
        extra={
            "rows": len(left.terms),
            "columns": len(right.terms),
            "left_background_policy": left.background_policy,
            "right_background_policy": right.background_policy,
            "workspace_bytes": int(exact.stats.workspace_bytes),
        },
    )
    output_path = Path(args.out)
    companion_paths = (
        (
            output_path.with_suffix(".rows.json"),
            output_path.with_suffix(".columns.json"),
        )
        if output_path.suffix.lower() == ".npy"
        else ()
    )
    _ = write_result_sidecar(
        output_path,
        provenance,
        companion_paths=companion_paths,
    )
    kind = "z-score" if zscores is not None else "true-score"
    print(f"Wrote {kind} matrix to {args.out}")


def parse_args(argv: Sequence[str] | None = None) -> IndexArgs:
    parser = argparse.ArgumentParser(
        prog="andes index",
        description="Build and query persistent ANDES indexes",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="build a reusable target database index")
    _ = build.add_argument("--emb", required=True, help="embedding CSV or NPY")
    _ = build.add_argument(
        "--genelist",
        required=True,
        help="embedding gene list",
    )
    _ = build.add_argument("--geneset", required=True, help="GMT database to index")
    _ = build.add_argument("--out", required=True, help="index directory")
    _ = build.add_argument(
        "--min",
        dest="min_size",
        type=int,
        default=10,
        help="minimum mapped term size (default: 10)",
    )
    _ = build.add_argument(
        "--max",
        dest="max_size",
        type=int,
        default=300,
        help="maximum mapped term size (default: 300)",
    )
    _ = build.add_argument(
        "--query-memory-mb",
        type=float,
        default=128.0,
        help="build workspace target in decimal MB (default: 128)",
    )
    _ = build.add_argument(
        "--query-blas-threads",
        type=int,
        default=0,
        help="BLAS threads for index construction; 0 keeps the runtime default",
    )
    _ = build.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing index directory",
    )
    build.set_defaults(func=cmd_build)

    verify = sub.add_parser("verify", help="audit a published index artifact")
    _ = verify.add_argument("--index", required=True, help="index directory")
    _ = verify.add_argument(
        "--full",
        action="store_true",
        help="hash and scan the dense embedding and best-match arrays",
    )
    verify.set_defaults(func=cmd_verify)

    query = sub.add_parser("query", help="score one query or indexed term")
    _ = query.add_argument("--index", required=True, help="index directory")
    query_source = query.add_mutually_exclusive_group(required=True)
    _ = query_source.add_argument(
        "--genes",
        help="text file with one query gene per line",
    )
    _ = query_source.add_argument("--term", help="indexed term identifier")
    _ = query.add_argument("--out", required=True, help="output CSV")
    _add_query_runtime(query)
    query.set_defaults(func=cmd_query)

    batch = sub.add_parser("batch-query", help="score GMT query sets")
    _ = batch.add_argument("--index", required=True, help="index directory")
    _ = batch.add_argument("--queries", required=True, help="query sets in GMT format")
    _ = batch.add_argument("--out", required=True, help="output CSV")
    _add_query_runtime(batch)
    batch.set_defaults(func=cmd_batch_query)

    compare = sub.add_parser("compare", help="compare two compatible indexes")
    _ = compare.add_argument("--index1", required=True, help="row index directory")
    _ = compare.add_argument("--index2", required=True, help="column index directory")
    _ = compare.add_argument("--out", required=True, help="output CSV or NPY")
    _add_cache_options(compare)
    _add_common_query_options(compare)
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv, namespace=IndexArgs())
    if (
        args.cmd in {"build", "query", "batch-query", "compare"}
        and args.query_memory_mb < 0
    ):
        parser.error("--query-memory-mb must be non-negative")
    if args.cmd != "verify" and args.query_blas_threads < 0:
        parser.error("--query-blas-threads must be non-negative")
    if args.cmd in {"query", "batch-query", "compare"} and args.null_blas_threads < 1:
        parser.error("--null-blas-threads must be at least 1")
    if args.cmd in {"query", "batch-query"} and args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.cmd in {"query", "batch-query", "compare"} and args.ite < 2:
        parser.error("--ite must be at least 2")
    if args.cmd == "build" and (args.min_size < 1 or args.max_size < args.min_size):
        parser.error("invalid gene-set size range")
    return args


def _add_query_runtime(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="keep the highest-scoring terms; 0 keeps all terms",
    )
    _add_cache_options(parser)
    _add_common_query_options(parser)


def _add_common_query_options(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--ite",
        type=int,
        default=1000,
        help="Monte Carlo null iterations (default: 1000)",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="random seed; -1 uses OS entropy",
    )
    _ = parser.add_argument(
        "--query-memory-mb",
        type=float,
        default=128.0,
        help="query workspace target in decimal MB (default: 128)",
    )
    _ = parser.add_argument(
        "--query-blas-threads",
        type=int,
        default=0,
        help="BLAS threads for scoring; 0 keeps the runtime default",
    )
    _ = parser.add_argument(
        "--null-blas-threads",
        type=int,
        default=1,
        help="BLAS threads for null construction (default: 1)",
    )
    _ = parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="write raw BMA scores",
    )
    _ = parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="load index arrays into memory",
    )


def _add_cache_options(parser: argparse.ArgumentParser) -> None:
    cache_group = parser.add_mutually_exclusive_group()
    _ = cache_group.add_argument(
        "--cache",
        default="",
        help="null artifact directory",
    )
    _ = cache_group.add_argument(
        "--cache-root",
        default="",
        help=(
            "content-addressed artifact root; defaults to ANDES_CACHE_ROOT or cache/"
        ),
    )
    _ = parser.add_argument(
        "--cache-policy",
        choices=("build", "require"),
        default="require",
        help=(
            "require a complete null artifact (default) or allow this command "
            "to build missing entries"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
