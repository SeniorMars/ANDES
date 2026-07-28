"""All-vs-all ANDES application and command-line entrypoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import application
from . import bma as func
from . import data
from .nulls import BmaNullModel
from .runtime import (
    blas_runtime_info,
    format_blas_runtime,
    numba_warmup_requirements,
    resolve_query_blas_limit,
)


def build_parser():
    parser = argparse.ArgumentParser(description="ANDES gene-set comparison")
    parser.add_argument("--emb", required=True, help="embedding CSV or NPY")
    parser.add_argument("--genelist", required=True, help="embedding gene list")
    parser.add_argument("--geneset1", required=True, help="first GMT database")
    parser.add_argument("--geneset2", required=True, help="second GMT database")
    parser.add_argument("--out", required=True, help="CSV or NPY result")
    parser.add_argument(
        "--cache",
        default="",
        help="typed null artifact directory; auto-named when empty",
    )
    parser.add_argument(
        "--no-precompute",
        action="store_true",
        help="require an existing compatible null artifact",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="ignore an existing cache and rebuild it",
    )
    parser.add_argument("--min", dest="min_size", type=int, default=10)
    parser.add_argument("--max", dest="max_size", type=int, default=300)
    parser.add_argument("--ite", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--query-workers",
        type=int,
        default=0,
        help="0 reuses --workers; bestmatch always uses one Python thread",
    )
    parser.add_argument(
        "--query-blas-threads",
        "--blas-threads",
        dest="query_blas_threads",
        type=int,
        default=0,
        help="0 uses runtime default for bestmatch and one otherwise",
    )
    parser.add_argument("--worker-blas-threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument(
        "--no-term-block-cache",
        action="store_true",
        help="make legacy batched mode fall back to pairwise gathering",
    )
    parser.add_argument("--numba-threshold", type=int, default=400)
    parser.add_argument(
        "--query-mode",
        choices=["batched", "pairwise", "bestmatch"],
        default="bestmatch",
    )
    parser.add_argument(
        "--null-mode",
        choices=["pairwise", "prefix"],
        default="prefix",
    )
    parser.add_argument("--query-memory-mb", type=float, default=1024.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.query_workers < 0:
        parser.error("--query-workers must be non-negative")
    if args.query_blas_threads < 0:
        parser.error("--query-blas-threads must be non-negative")
    if args.worker_blas_threads < 1:
        parser.error("--worker-blas-threads must be at least 1")
    if args.min_size < 1 or args.max_size < args.min_size:
        parser.error("invalid gene-set size range")
    if args.ite < 1:
        parser.error("--ite must be at least 1")
    if args.query_memory_mb < 0:
        parser.error("--query-memory-mb must be non-negative")
    args.seed = func.BmaNullBuilder.resolve_seed(args.seed)
    return args


def _load_embedding(embedding_path, genes_path):
    return data.load_embedding_space(embedding_path, genes_path)


def _load_database(path, embedding, min_size, max_size):
    return data.load_gene_set_database(
        path,
        embedding,
        min_size=min_size,
        max_size=max_size,
        sort_terms=False,
    )


def _default_cache_path(
    embedding,
    left,
    right,
    *,
    iterations,
    seed,
    sampling,
    ddof=1,
):
    from .nulls import NullSpec

    spec = NullSpec(
        kind="bma",
        iterations=iterations,
        seed=seed,
        sampling=sampling,
        ddof=ddof,
        embedding_hash=embedding.vector_hash,
        population_hashes=(
            left.background_hash,
            right.background_hash,
        ),
    )
    return Path("cache") / f"bma_{spec.fingerprint}.null"


def _load_cache(path):
    cache = func.BmaNullBuilder()
    cache.load_artifact(path)
    return cache


def _save_cache(cache, path):
    path = Path(path)
    cache.save_artifact(path, overwrite=path.exists())


def _load_or_build_null(args, embedding, left, right):
    null_sampling = "prefix_coupled" if args.null_mode == "prefix" else "per_size_pair"
    cache_path = (
        Path(args.cache)
        if args.cache
        else _default_cache_path(
            embedding,
            left,
            right,
            iterations=args.ite,
            seed=args.seed,
            sampling=null_sampling,
        )
    )
    expected = func.BmaNullBuilder.build_metadata(
        embedding.vectors,
        left.background,
        right.background,
        args.ite,
        args.seed,
        null_sampling=null_sampling,
    )
    size_pairs = {
        (int(left_size), int(right_size))
        for left_size in np.unique(left.sizes)
        for right_size in np.unique(right.sizes)
    }

    cache = func.BmaNullBuilder()
    cache_ready = False
    if cache_path.exists() and not args.rebuild_cache:
        cache = _load_cache(cache_path)
        metadata_ok, reason = cache.metadata_matches(expected)
        missing = sorted(pair for pair in size_pairs if pair not in cache.cache)
        cache_ready = metadata_ok and not missing
        if not cache_ready and args.no_precompute:
            detail = reason if not metadata_ok else f"{len(missing)} missing pairs"
            raise ValueError(f"null cache is incompatible: {detail}")
    elif args.no_precompute:
        raise FileNotFoundError(f"null cache does not exist: {cache_path}")

    effective_query_mode = (
        "pairwise"
        if args.query_mode == "pairwise"
        or (args.query_mode == "batched" and args.no_term_block_cache)
        else args.query_mode
    )
    warm_bma, warm_fys = numba_warmup_requirements(
        effective_query_mode,
        args.null_mode,
        args.numba_threshold,
        size_pairs,
        needs_null_build=not cache_ready,
    )
    if warm_bma or warm_fys:
        func.warmup_numba(warm_bma=warm_bma, warm_fys=warm_fys)

    if not cache_ready:
        if args.null_mode == "prefix":
            with threadpool_limits(limits=args.worker_blas_threads, user_api="blas"):
                cache.precompute_prefix(
                    embedding.vectors,
                    left.background,
                    size_pairs,
                    ite=args.ite,
                    seed=args.seed,
                    verbose=args.verbose,
                    population_idx2=right.background,
                )
        else:
            cache.precompute_parallel(
                embedding.vectors,
                left.background,
                size_pairs,
                ite=args.ite,
                seed=args.seed,
                verbose=args.verbose,
                n_workers=args.workers,
                chunk_size=None if args.chunk_size <= 0 else args.chunk_size,
                population_idx2=right.background,
                blas_threads_per_worker=args.worker_blas_threads,
            )
        _save_cache(cache, cache_path)
    return cache, cache_path


def main(argv=None):
    args = parse_args(argv)
    print(f"Seed: {args.seed}")
    print("Loading and validating inputs...")
    embedding = _load_embedding(args.emb, args.genelist)
    left = _load_database(args.geneset1, embedding, args.min_size, args.max_size)
    right = _load_database(args.geneset2, embedding, args.min_size, args.max_size)
    print(
        f"Embedding: {len(embedding.genes)} genes x "
        f"{embedding.vectors.shape[1]} dimensions"
    )
    print(f"Databases: {len(left.terms)} x {len(right.terms)} terms")

    cache, cache_path = _load_or_build_null(args, embedding, left, right)
    null_model = BmaNullModel.from_builder(cache)
    print(f"Null artifact: {cache_path}")

    requested_workers = args.workers if args.query_workers <= 0 else args.query_workers
    effective_engine = (
        "pairwise"
        if args.query_mode == "batched" and args.no_term_block_cache
        else args.query_mode
    )
    effective_workers = (
        1
        if effective_engine == "bestmatch"
        else min(max(1, requested_workers), len(left.terms))
    )
    query_blas_limit = resolve_query_blas_limit(
        effective_engine, args.query_blas_threads
    )
    blas_context = (
        nullcontext()
        if query_blas_limit is None
        else threadpool_limits(limits=query_blas_limit, user_api="blas")
    )
    with blas_context:
        pools = blas_runtime_info()
        runtime = {
            "query_workers_requested": int(max(1, requested_workers)),
            "query_workers_effective": int(effective_workers),
            "query_blas_threads_requested": int(args.query_blas_threads),
            "query_blas_threads_limit": query_blas_limit,
            "worker_blas_threads": int(args.worker_blas_threads),
            "blas_pools": pools,
        }
        print(
            f"Query: engine={effective_engine}, workers={effective_workers}, "
            f"BLAS={query_blas_limit or 'runtime default'}; "
            f"{format_blas_runtime(pools)}"
        )
        result = application.run_compare(
            application.CompareRequest(
                embedding=embedding,
                left=left,
                right=right,
                engine=effective_engine,
                workers=effective_workers,
                numba_threshold=args.numba_threshold,
                workspace_mb=args.query_memory_mb,
                show_progress=args.verbose,
                null_model=null_model,
                runtime=runtime,
            )
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
