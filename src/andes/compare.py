"""All-vs-all ANDES application and command-line entrypoint."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from . import application, data
from .nulls import BmaNullModel, null_cache_dir, resolve_bma_null, resolve_null_seed
from .runtime import (
    blas_runtime_info,
    format_blas_runtime,
    query_blas_context,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="andes compare",
        description="ANDES gene-set comparison",
    )
    parser.add_argument("--emb", required=True, help="embedding CSV or NPY")
    parser.add_argument("--genelist", required=True, help="embedding gene list")
    parser.add_argument("--geneset1", required=True, help="first GMT database")
    parser.add_argument("--geneset2", required=True, help="second GMT database")
    parser.add_argument("--out", required=True, help="CSV or NPY result")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache",
        default="",
        help="null artifact directory",
    )
    cache_group.add_argument(
        "--cache-root",
        default="",
        help=(
            "content-addressed artifact root; defaults to ANDES_CACHE_ROOT or cache/"
        ),
    )
    parser.add_argument(
        "--cache-policy",
        choices=("build", "require"),
        default="build",
        help="build missing null entries or require a complete artifact",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="replace an existing null artifact",
    )
    parser.add_argument(
        "--min",
        dest="min_size",
        type=int,
        default=10,
        help="minimum mapped term size (default: 10)",
    )
    parser.add_argument(
        "--max",
        dest="max_size",
        type=int,
        default=300,
        help="maximum mapped term size (default: 300)",
    )
    parser.add_argument(
        "--ite",
        type=int,
        default=1000,
        help="Monte Carlo null iterations (default: 1000)",
    )
    parser.add_argument(
        "--query-blas-threads",
        dest="query_blas_threads",
        type=int,
        default=0,
        help="BLAS threads for scoring; 0 keeps the runtime default",
    )
    parser.add_argument(
        "--null-blas-threads",
        type=int,
        default=1,
        help="BLAS threads used while constructing the prefix null",
    )
    parser.add_argument(
        "--query-memory-mb",
        type=float,
        default=128.0,
        help="scoring workspace target in decimal MB (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="random seed; -1 uses OS entropy",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.query_blas_threads < 0:
        parser.error("--query-blas-threads must be non-negative")
    if args.null_blas_threads < 1:
        parser.error("--null-blas-threads must be at least 1")
    if args.min_size < 1 or args.max_size < args.min_size:
        parser.error("invalid gene-set size range")
    if args.ite < 2:
        parser.error("--ite must be at least 2")
    if args.query_memory_mb < 0:
        parser.error("--query-memory-mb must be non-negative")
    if args.cache_policy == "require" and args.rebuild_cache:
        parser.error("--rebuild-cache requires --cache-policy build")
    args.seed = resolve_null_seed(args.seed)
    return args


def _load_or_build_null(args, embedding, left, right):
    return resolve_bma_null(
        embedding,
        left.background,
        right.background,
        left.sizes,
        right.sizes,
        path=args.cache or None,
        base_dir=(
            None if args.cache else null_cache_dir("bma", args.cache_root or None)
        ),
        iterations=args.ite,
        seed=args.seed,
        no_build=args.cache_policy == "require",
        rebuild=args.rebuild_cache,
        blas_threads=args.null_blas_threads,
    )


def main(argv=None):
    args = parse_args(argv)
    print(f"Seed: {args.seed}")
    print("Loading and validating inputs...")
    embedding = data.load_embedding_space(args.emb, args.genelist)
    left = data.load_gene_set_database(
        args.geneset1,
        embedding,
        min_size=args.min_size,
        max_size=args.max_size,
        sort_terms=True,
    )
    right = data.load_gene_set_database(
        args.geneset2,
        embedding,
        min_size=args.min_size,
        max_size=args.max_size,
        sort_terms=True,
    )
    print(
        f"Embedding: {len(embedding.genes)} genes x "
        f"{embedding.vectors.shape[1]} dimensions"
    )
    print(f"Databases: {len(left.terms)} x {len(right.terms)} terms")

    null_started = time.perf_counter()
    resolution = _load_or_build_null(args, embedding, left, right)
    null_resolution_seconds = time.perf_counter() - null_started
    print(f"Null artifact: {resolution.path}")

    query_blas_limit = None if args.query_blas_threads == 0 else args.query_blas_threads
    with query_blas_context(args.query_blas_threads):
        pools = blas_runtime_info()
        runtime = {
            "query_blas_threads_requested": int(args.query_blas_threads),
            "query_blas_threads_limit": query_blas_limit,
            "null_blas_threads": int(args.null_blas_threads),
            "null_cache_policy": args.cache_policy,
            "null_cache_built": bool(resolution.built),
            "null_cache_added_entries": int(resolution.added_entries),
            "null_cache_resolution_seconds": null_resolution_seconds,
            "blas_pools": pools,
        }
        print(
            "Query: engine=bestmatch, "
            f"BLAS={query_blas_limit or 'runtime default'}; "
            f"{format_blas_runtime(pools)}"
        )
        query_started = time.perf_counter()
        null_model = resolution.model
        if not isinstance(null_model, BmaNullModel):
            raise TypeError("BMA null resolver returned the wrong model kind")
        result = application.run_compare(
            embedding=embedding,
            left=left,
            right=right,
            workspace_mb=args.query_memory_mb,
            null_model=null_model,
            runtime=runtime,
        )
        runtime["query_seconds"] = time.perf_counter() - query_started
        result = replace(
            result,
            provenance=replace(
                result.provenance,
                runtime=runtime,
            ),
        )

    sidecar = application.write_compare_result(result, args.out)
    print(f"Saved {result.score_kind} matrix to {args.out}")
    print(f"Provenance: {sidecar}")
    if result.scores.stats.workspace_bytes:
        print(
            "Estimated query-owned workspace: "
            f"{result.scores.stats.workspace_bytes / 1e6:.1f} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
