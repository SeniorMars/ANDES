#!/usr/bin/env python3
import argparse
import cProfile
from contextlib import nullcontext
import json
import os
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from andes import data as ld
from andes import bma as func
from andes.ranked import (
    RankedNullBuilder,
    compute_ranked_emb,
    score_terms_bestmatch,
    score_terms_batched,
    warmup_numba_es,
)
from experiments.legacy import set_analysis_func as func_old


class Timer:
    def __init__(self):
        self.rows = []

    def record(self, name, started):
        self.rows.append((name, time.perf_counter() - started))

    def print(self):
        total = sum(seconds for _, seconds in self.rows)
        print("\nStage timings")
        for name, seconds in self.rows:
            pct = 100.0 * seconds / total if total else 0.0
            print(f"  {name:18s} {seconds:9.3f}s  {pct:5.1f}%")
        print(f"  {'total':18s} {total:9.3f}s")

    def as_dict(self):
        total = sum(seconds for _, seconds in self.rows)
        return {
            "stages": {name: seconds for name, seconds in self.rows},
            "total_s": total,
        }


def load_embedding(path, genelist_path):
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    with open(genelist_path) as fh:
        genes = [line.strip() for line in fh]
    if len(genes) != raw.shape[0]:
        raise ValueError("embedding rows do not match gene-list length")
    return raw, genes


def maybe_limit(items, n):
    if n <= 0:
        return items
    return items[: min(n, len(items))]


def resolve_query_blas_limit(query_mode, requested_threads):
    """Resolve the scoped BLAS limit for main-process query scoring."""
    requested_threads = int(requested_threads)
    if requested_threads < 0:
        raise ValueError("query BLAS threads must be non-negative")
    if requested_threads > 0:
        return requested_threads
    return None if query_mode in {"bestmatch", "gsea"} else 1


def numba_warmup_requirements(
    query_mode,
    null_mode,
    numba_threshold,
    size_pairs,
    needs_null_build,
):
    """Return ``(warm_bma, warm_fys)`` for kernels this run can execute."""
    warm_bma = (
        query_mode == "pairwise"
        and int(numba_threshold) > 0
        and any(int(m) * int(k) < int(numba_threshold) for m, k in size_pairs)
    )
    warm_fys = (
        bool(needs_null_build)
        and null_mode == "pairwise"
        and bool(size_pairs)
    )
    return warm_bma, warm_fys


def blas_runtime_info():
    """Return compact, JSON-safe metadata for active BLAS thread pools."""
    pools = []
    for pool in threadpool_info():
        if pool.get("user_api") != "blas":
            continue
        pools.append(
            {
                "internal_api": pool.get("internal_api", "unknown"),
                "prefix": pool.get("prefix", "unknown"),
                "num_threads": int(pool.get("num_threads", 0)),
            }
        )
    return pools


def effective_blas_threads(pools):
    """Return the maximum active BLAS pool width, or zero if unavailable."""
    return max((pool["num_threads"] for pool in pools), default=0)


def run_andes(args):
    timer = Timer()
    args.seed = func.BmaNullBuilder.resolve_seed(args.seed)
    report = {"mode": "andes", "config": vars(args).copy()}

    t = time.perf_counter()
    raw, node_list = load_embedding(args.emb, args.genelist)
    timer.record("load", t)

    t = time.perf_counter()
    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    del raw
    timer.record("normalize", t)

    t = time.perf_counter()
    node_set = set(node_list)
    g_node2index = {g: i for i, g in enumerate(node_list)}
    geneset1 = ld.load_gmt(args.geneset1)
    geneset2 = ld.load_gmt(args.geneset2)
    idx1 = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset1, g_node2index, upper=args.max_size, lower=args.min_size)
    )
    idx2 = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset2, g_node2index, upper=args.max_size, lower=args.min_size)
    )
    terms1 = maybe_limit(list(idx1.keys()), args.limit_terms1)
    terms2 = maybe_limit(list(idx2.keys()), args.limit_terms2)
    if not terms1 or not terms2:
        raise ValueError("no terms passed size filters")
    bg1 = np.asarray(func.get_background_indices(geneset1, node_set, g_node2index), dtype=np.int32)
    bg2 = np.asarray(func.get_background_indices(geneset2, node_set, g_node2index), dtype=np.int32)
    timer.record("gene_sets", t)

    sizes1 = {len(idx1[t]) for t in terms1}
    sizes2 = {len(idx2[t]) for t in terms2}
    size_pairs = {(m, k) for m in sizes1 for k in sizes2}
    warm_bma, _ = numba_warmup_requirements(
        args.query_mode,
        args.null_mode,
        args.numba_threshold,
        size_pairs,
        needs_null_build=False,
    )
    if warm_bma:
        t = time.perf_counter()
        func.warmup_numba(warm_bma=True, warm_fys=False)
        timer.record("numba_bma_warmup", t)

    t = time.perf_counter()
    cache = func.BmaNullBuilder()
    null_sampling = (
        "prefix_coupled" if args.null_mode == "prefix" else "per_size_pair"
    )
    cache_loaded = False
    null_built = False
    if args.skip_cache_build:
        for pair in size_pairs:
            cache.cache[pair] = (0.0, 1.0)
    else:
        # For benchmarking we only require the embedding and population to match;
        # a different ite is acceptable (we're timing query scoring, not null quality).
        _BENCH_REQUIRED_KEYS = {"kind", "version", "embedding_hash",
                                "population1_hash", "population2_hash",
                                "null_sampling"}
        expected = func.BmaNullBuilder.build_metadata(
            E_unit,
            bg1,
            bg2,
            args.ite,
            args.seed,
            null_sampling=null_sampling,
        )
        if args.cache and os.path.exists(args.cache):
            cache.load_artifact(args.cache)
            meta = cache.metadata
            core_ok = all(meta.get(k) == expected[k] for k in _BENCH_REQUIRED_KEYS)
            missing = [pair for pair in size_pairs if pair not in cache.cache]
            cache_loaded = core_ok and not missing
            if not cache_loaded and args.verbose:
                if not core_ok:
                    bad = [k for k in _BENCH_REQUIRED_KEYS if meta.get(k) != expected[k]]
                    print(f"Ignoring BMA cache: core metadata mismatch ({', '.join(bad)})")
                else:
                    print(f"Ignoring BMA cache: {len(missing)} missing pairs")
            elif cache_loaded and args.verbose and meta.get("ite") != args.ite:
                print(f"Using BMA cache built with ite={meta.get('ite')} "
                      f"(requested {args.ite}); sufficient for timing")

        if not cache_loaded:
            null_built = True
            _, warm_fys = numba_warmup_requirements(
                args.query_mode,
                args.null_mode,
                args.numba_threshold,
                size_pairs,
                needs_null_build=True,
            )
            if warm_fys:
                warm_started = time.perf_counter()
                func.warmup_numba(warm_bma=False, warm_fys=True)
                timer.record("numba_fys_warmup", warm_started)
                t = time.perf_counter()
            n_chunks_desc = "auto-cost" if args.chunk_size <= 0 else str(args.chunk_size)
            print(f"Building BMA null cache: {len(size_pairs)} size pairs, "
                  f"ite={args.ite}, workers={args.workers}, "
                  f"chunk={n_chunks_desc}, null_mode={args.null_mode}")
            if args.null_mode == "prefix":
                with threadpool_limits(
                    limits=args.worker_blas_threads, user_api="blas"
                ):
                    cache.precompute_prefix(
                        E_unit,
                        bg1,
                        size_pairs,
                        ite=args.ite,
                        seed=args.seed,
                        verbose=args.verbose,
                        population_idx2=bg2,
                    )
            elif args.workers > 1:
                cache.precompute_parallel(
                    E_unit,
                    bg1,
                    size_pairs,
                    ite=args.ite,
                    seed=args.seed,
                    verbose=args.verbose,
                    n_workers=args.workers,
                    chunk_size=None if args.chunk_size <= 0 else args.chunk_size,
                    population_idx2=bg2,
                    show_progress=True,
                    blas_threads_per_worker=args.worker_blas_threads,
                )
            else:
                with threadpool_limits(
                    limits=args.worker_blas_threads, user_api="blas"
                ):
                    cache.precompute(
                        E_unit,
                        bg1,
                        size_pairs,
                        ite=args.ite,
                        seed=args.seed,
                        verbose=args.verbose,
                        population_idx2=bg2,
                    )
            if args.cache:
                Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
                cache.save_artifact(args.cache, overwrite=os.path.exists(args.cache))
    timer.record("cache_build", t)

    symmetric = np.array_equal(bg1, bg2) and func.same_index_arrays_by_term(
        terms1, idx1, terms2, idx2
    )
    t = time.perf_counter()
    if args.query_mode == "bestmatch":
        blocks1 = {}
        blocks2 = {}
    else:
        blocks1 = func.precompute_term_embedding_blocks(
            E_unit, {term: idx1[term] for term in terms1}
        )
        if symmetric:
            blocks2 = blocks1
        else:
            blocks2 = func.precompute_term_embedding_blocks(
                E_unit, {term: idx2[term] for term in terms2}
            )
    timer.record("term_blocks", t)

    t = time.perf_counter()
    query_workers = args.workers if args.query_workers <= 0 else args.query_workers
    requested_query_workers = max(1, int(query_workers))
    effective_query_workers = (
        1
        if args.query_mode == "bestmatch"
        else min(requested_query_workers, len(terms1))
    )
    query_blas_limit = resolve_query_blas_limit(
        args.query_mode, args.query_blas_threads
    )
    workspace_mb = 0.0
    bestmatch_stats = {}
    query_blas_context = (
        nullcontext()
        if query_blas_limit is None
        else threadpool_limits(limits=query_blas_limit, user_api="blas")
    )
    with query_blas_context:
        query_blas_pools = blas_runtime_info()
        effective_workers_label = (
            "workspace-capped at runtime"
            if args.query_mode == "batched"
            else str(effective_query_workers)
        )
        print(
            "ANDES query execution: "
            f"Python workers requested={requested_query_workers}, "
            f"effective={effective_workers_label}; "
            f"BLAS requested={args.query_blas_threads}, "
            f"limit={query_blas_limit}, "
            f"effective={effective_blas_threads(query_blas_pools)}"
        )
        if args.query_mode == "bestmatch":
            zscores, bestmatch_stats = func.score_bma_zscore_matrix_bestmatch(
                E_unit,
                terms1,
                terms2,
                idx1,
                idx2,
                cache,
                symmetric=symmetric,
                max_workspace_mb=args.query_memory_mb,
                show_progress=True,
            )
            workspace_mb = float(bestmatch_stats.get("workspace_mb", 0.0))
        elif args.query_mode == "batched":
            zscores, effective_query_workers, workspace_mb = (
                func.score_bma_zscore_matrix_batched(
                    terms1,
                    terms2,
                    cache,
                    blocks1,
                    blocks2,
                    symmetric=symmetric,
                    n_workers=requested_query_workers,
                    max_workspace_mb=args.query_memory_mb,
                    show_progress=True,
                )
            )
            print(
                "ANDES query Python workers effective="
                f"{effective_query_workers}"
            )
        else:
            zscores = func.score_bma_zscore_matrix(
                E_unit,
                terms1,
                terms2,
                idx1,
                idx2,
                cache,
                blocks1=blocks1,
                blocks2=blocks2,
                symmetric=symmetric,
                n_workers=requested_query_workers,
                numba_threshold=args.numba_threshold,
                show_progress=True,
            )
    timer.record("query_scoring", t)

    t = time.perf_counter()
    if args.out:
        pd.DataFrame(zscores, index=terms1, columns=terms2).to_csv(args.out)
    timer.record("save", t)

    print(f"ANDES terms: {len(terms1)} x {len(terms2)}")
    print(f"Unique size pairs: {len(size_pairs)}")
    timer.print()
    report.update(
        {
            "n_terms1": len(terms1),
            "n_terms2": len(terms2),
            "n_pairs": len(terms1) * len(terms2),
            "n_scored_pairs": len(terms1) * (len(terms1) + 1) // 2 if symmetric else len(terms1) * len(terms2),
            "n_size_pairs": len(size_pairs),
            "symmetric": symmetric,
            "query_mode": args.query_mode,
            "null_mode": args.null_mode,
            "cache_loaded": cache_loaded,
            "cache_entries": len(cache.cache),
            "query_workers": int(effective_query_workers),
            "query_workers_requested": requested_query_workers,
            "query_workers_effective": int(effective_query_workers),
            "query_blas_threads_requested": args.query_blas_threads,
            "query_blas_threads_limit": query_blas_limit,
            "query_blas_threads_effective": effective_blas_threads(
                query_blas_pools
            ),
            "query_blas_pools": query_blas_pools,
            "worker_blas_threads": args.worker_blas_threads,
            "worker_blas_threads_requested": args.worker_blas_threads,
            "worker_blas_threads_effective": (
                args.worker_blas_threads if null_built else None
            ),
            "query_workspace_mb": workspace_mb,
            "query_chunk_workspace_mb": (
                float(bestmatch_stats.get("chunk_workspace_mb", 0.0))
                if args.query_mode == "bestmatch"
                else workspace_mb
            ),
            "bestmatch_stats": bestmatch_stats,
            "embedding_shape": list(E_unit.shape),
            "timing": timer.as_dict(),
        }
    )
    return report


def load_ranked(path, g_node2index, node_set):
    df = pd.read_csv(path, sep="\t", index_col=0, header=None)
    return np.asarray([g_node2index[str(g)] for g in df.index if str(g) in node_set], dtype=np.int32)


def run_gsea(args):
    timer = Timer()
    args.seed = RankedNullBuilder.resolve_seed(args.seed)
    report = {"mode": "gsea", "config": vars(args).copy()}

    t = time.perf_counter()
    raw, node_list = load_embedding(args.emb, args.genelist)
    timer.record("load", t)

    t = time.perf_counter()
    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    del raw
    timer.record("normalize", t)

    t = time.perf_counter()
    node_set = set(node_list)
    g_node2index = {g: i for i, g in enumerate(node_list)}
    geneset = ld.load_gmt(args.geneset)
    idx = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset, g_node2index, upper=args.max_size, lower=args.min_size)
    )
    terms = maybe_limit(sorted(idx.keys()), args.limit_terms)
    if not terms:
        raise ValueError("no terms passed size filters")
    pop = np.asarray(sorted(g_node2index[g] for g in (set().union(*geneset.values()) & node_set)), dtype=np.int32)
    ranked_idx = load_ranked(args.rankedlist, g_node2index, node_set)
    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    timer.record("gene_sets", t)

    es_warmed = False
    if args.score_mode == "batched":
        warm_started = time.perf_counter()
        warmup_numba_es()
        timer.record("numba_es_warmup", warm_started)
        es_warmed = True

    t = time.perf_counter()
    sizes = {len(idx[term]) for term in terms}
    cache = RankedNullBuilder()
    cache_loaded = False
    null_built = False
    if args.skip_cache_build:
        for m in sizes:
            cache.cache[m] = (0.0, 1.0)
    else:
        _ES_BENCH_REQUIRED_KEYS = {"kind", "version", "embedding_hash",
                                   "population_hash", "ranked_emb_hash"}
        expected = RankedNullBuilder.build_metadata(E_unit, pop, ranked_emb, args.ite, args.seed)
        if args.cache and os.path.exists(args.cache):
            cache = RankedNullBuilder.load_artifact(args.cache)
            meta = cache.metadata
            core_ok = all(meta.get(k) == expected[k] for k in _ES_BENCH_REQUIRED_KEYS)
            missing = cache.missing_sizes(sizes)
            cache_loaded = core_ok and not missing
            if not cache_loaded and args.verbose:
                if not core_ok:
                    bad = [k for k in _ES_BENCH_REQUIRED_KEYS if meta.get(k) != expected[k]]
                    print(f"Ignoring ES cache: core metadata mismatch ({', '.join(bad)})")
                else:
                    print(f"Ignoring ES cache: {len(missing)} missing sizes")
            elif cache_loaded and args.verbose and meta.get("ite") != args.ite:
                print(f"Using ES cache built with ite={meta.get('ite')} "
                      f"(requested {args.ite}); sufficient for timing")

        if not cache_loaded:
            if not es_warmed:
                warm_started = time.perf_counter()
                warmup_numba_es()
                timer.record("numba_es_warmup", warm_started)
                es_warmed = True
                t = time.perf_counter()
            null_built = True
            print(f"Building ES null cache: {len(sizes)} sizes, "
                  f"ite={args.ite}, workers={args.workers}")
            if args.workers > 1:
                cache.precompute_parallel(
                    E_unit,
                    pop,
                    sizes,
                    ranked_emb,
                    ite=args.ite,
                    seed=args.seed,
                    verbose=args.verbose,
                    n_workers=args.workers,
                    show_progress=True,
                    blas_threads_per_worker=args.worker_blas_threads,
                )
            else:
                with threadpool_limits(
                    limits=args.worker_blas_threads, user_api="blas"
                ):
                    cache.precompute(
                        E_unit,
                        pop,
                        sizes,
                        ranked_emb,
                        ite=args.ite,
                        seed=args.seed,
                        verbose=args.verbose,
                    )
            if args.cache:
                Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
                cache.save_artifact(args.cache, overwrite=os.path.exists(args.cache))
    timer.record("cache_build", t)

    t = time.perf_counter()
    ranked_emb_T = np.ascontiguousarray(ranked_emb.T, dtype=np.float32)
    score_workspace_mb = 0.0
    query_blas_limit = resolve_query_blas_limit(
        "gsea", args.query_blas_threads
    )
    query_blas_context = (
        nullcontext()
        if query_blas_limit is None
        else threadpool_limits(limits=query_blas_limit, user_api="blas")
    )
    with query_blas_context:
        query_blas_pools = blas_runtime_info()
        print(
            "GSEA query execution: Python workers requested=1, effective=1; "
            f"BLAS requested={args.query_blas_threads}, "
            f"limit={query_blas_limit}, "
            f"effective={effective_blas_threads(query_blas_pools)}"
        )
        if args.score_mode == "bestmatch":
            true_scores, z_scores, score_stats = score_terms_bestmatch(
                E_unit,
                idx,
                terms,
                ranked_emb,
                cache,
                max_workspace_mb=args.query_memory_mb,
            )
            score_workspace_mb = float(score_stats.get("workspace_mb", 0.0))
        else:
            true_scores, z_scores = score_terms_batched(
                E_unit, idx, terms, ranked_emb_T, cache
            )
    out = np.zeros((len(terms), 2), dtype=np.float32)
    for i, term in enumerate(terms):
        out[i, 0] = true_scores[term]
        out[i, 1] = z_scores[term]
    timer.record("query_scoring", t)

    t = time.perf_counter()
    if args.out:
        pd.DataFrame(out, index=terms, columns=["true_score", "z_score"]).to_csv(args.out)
    timer.record("save", t)

    print(f"GSEA terms: {len(terms)}")
    print(f"Unique sizes: {len(sizes)}")
    timer.print()
    report.update(
        {
            "n_terms": len(terms),
            "n_sizes": len(sizes),
            "ranked_list_len": int(len(ranked_idx)),
            "score_mode": args.score_mode,
            "cache_loaded": cache_loaded,
            "cache_entries": len(cache.cache),
            "query_workers": 1,
            "query_workers_requested": 1,
            "query_workers_effective": 1,
            "query_blas_threads_requested": args.query_blas_threads,
            "query_blas_threads_limit": query_blas_limit,
            "query_blas_threads_effective": effective_blas_threads(
                query_blas_pools
            ),
            "query_blas_pools": query_blas_pools,
            "worker_blas_threads": args.worker_blas_threads,
            "worker_blas_threads_requested": args.worker_blas_threads,
            "worker_blas_threads_effective": (
                args.worker_blas_threads if null_built else None
            ),
            "query_workspace_mb": score_workspace_mb,
            "embedding_shape": list(E_unit.shape),
            "timing": timer.as_dict(),
        }
    )
    return report


def run_andes_old(args):
    """Original ANDES: full N×N cosine-similarity matrix + per-pair MC (no null cache)."""
    from functools import partial
    from multiprocessing import Pool
    from sklearn import metrics

    timer = Timer()
    report = {"mode": "andes_old", "config": vars(args).copy()}

    t = time.perf_counter()
    raw, node_list = load_embedding(args.emb, args.genelist)
    timer.record("load", t)

    t = time.perf_counter()
    S = metrics.pairwise.cosine_similarity(raw, raw)
    timer.record("S_matrix", t)

    t = time.perf_counter()
    node_set = set(node_list)
    g_node2index = {g: i for i, g in enumerate(node_list)}
    geneset1 = ld.load_gmt(args.geneset1)
    geneset2 = ld.load_gmt(args.geneset2)
    g1_idx = ld.term2indexes(geneset1, g_node2index, upper=args.max_size, lower=args.min_size)
    g2_idx = ld.term2indexes(geneset2, g_node2index, upper=args.max_size, lower=args.min_size)
    terms1 = maybe_limit(list(g1_idx.keys()), args.limit_terms1)
    terms2 = maybe_limit(list(g2_idx.keys()), args.limit_terms2)
    if not terms1 or not terms2:
        raise ValueError("no terms passed size filters")
    bg1 = list(func.get_background_indices(geneset1, node_set, g_node2index))
    bg2 = list(func.get_background_indices(geneset2, node_set, g_node2index))
    timer.record("gene_sets", t)

    t = time.perf_counter()
    all_pairs = [(t1, t2) for t1 in terms1 for t2 in terms2]
    f = partial(func_old.andes, matrix=S,
                g1_term2index=g1_idx, g2_term2index=g2_idx,
                g1_population=bg1, g2_population=bg2,
                ite=args.ite)
    with Pool(args.workers) as p:
        results = list(p.imap(f, all_pairs, chunksize=max(1, len(all_pairs) // (args.workers * 8))))
    timer.record("scoring", t)

    t = time.perf_counter()
    if args.out:
        t1_idx = {t: i for i, t in enumerate(terms1)}
        t2_idx = {t: j for j, t in enumerate(terms2)}
        zmat = np.full((len(terms1), len(terms2)), np.nan, dtype=np.float32)
        for (pair_t1, pair_t2), ret in zip(all_pairs, results):
            zmat[t1_idx[pair_t1], t2_idx[pair_t2]] = ret[1]
        pd.DataFrame(zmat, index=terms1, columns=terms2).to_csv(args.out)
    timer.record("save", t)

    print(f"ANDES_old terms: {len(terms1)} x {len(terms2)}")
    print(f"Total pairs scored: {len(all_pairs)}")
    timer.print()
    report.update({
        "n_terms1": len(terms1),
        "n_terms2": len(terms2),
        "n_pairs": len(all_pairs),
        "embedding_shape": list(raw.shape),
        "timing": timer.as_dict(),
    })
    return report


def run_gsea_old(args):
    """Original GSEA-ANDES: full N×N cosine-similarity matrix + per-term MC (no null cache)."""
    from functools import partial
    from multiprocessing import Pool
    from sklearn import metrics

    timer = Timer()
    report = {"mode": "gsea_old", "config": vars(args).copy()}

    t = time.perf_counter()
    raw, node_list = load_embedding(args.emb, args.genelist)
    timer.record("load", t)

    t = time.perf_counter()
    S = metrics.pairwise.cosine_similarity(raw, raw)
    timer.record("S_matrix", t)

    t = time.perf_counter()
    node_set = set(node_list)
    g_node2index = {g: i for i, g in enumerate(node_list)}
    geneset = ld.load_gmt(args.geneset)
    g_idx = ld.term2indexes(geneset, g_node2index, upper=args.max_size, lower=args.min_size)
    terms = maybe_limit(sorted(g_idx.keys()), args.limit_terms)
    if not terms:
        raise ValueError("no terms passed size filters")
    bg = list(func.get_background_indices(geneset, node_set, g_node2index))
    ranked_idx = load_ranked(args.rankedlist, g_node2index, node_set)
    ranked_list = list(map(int, ranked_idx))
    timer.record("gene_sets", t)

    t = time.perf_counter()
    f = partial(func_old.gsea_andes, ranked_list=ranked_list, matrix=S,
                term2indices=g_idx, annotated_indices=bg, ite=args.ite)
    with Pool(args.workers) as p:
        results = list(p.imap(f, terms, chunksize=max(1, len(terms) // (args.workers * 4))))
    timer.record("scoring", t)

    t = time.perf_counter()
    if args.out:
        pd.DataFrame(
            {"true_score": [r[0] for r in results], "z_score": [r[1] for r in results]},
            index=terms,
        ).to_csv(args.out)
    timer.record("save", t)

    print(f"GSEA_old terms: {len(terms)}")
    timer.print()
    report.update({
        "n_terms": len(terms),
        "ranked_list_len": int(len(ranked_list)),
        "embedding_shape": list(raw.shape),
        "timing": timer.as_dict(),
    })
    return report


def write_report(report, path):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = dict(report)
    clean["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in clean["config"].items()
        if key != "func"
    }
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2, sort_keys=True)
    print(f"\nWrote JSON report to {path}")



def run_with_optional_profile(args):
    if not args.profile_out:
        return args.func(args)

    Path(args.profile_out).parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return args.func(args)
    finally:
        profiler.disable()
        profiler.dump_stats(args.profile_out)
        text_path = args.profile_out + ".txt"
        with open(text_path, "w") as fh:
            stats = pstats.Stats(profiler, stream=fh).sort_stats("cumtime")
            stats.print_stats(args.profile_top)
        print(f"\nWrote cProfile stats to {args.profile_out}")
        print(f"Wrote profile summary to {text_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage-timed ANDES/ANDES-GSEA benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--emb", required=True)
        s.add_argument("--genelist", required=True)
        s.add_argument("--out", default="")
        s.add_argument("--min", dest="min_size", type=int, default=10)
        s.add_argument("--max", dest="max_size", type=int, default=300)
        s.add_argument("--ite", type=int, default=100)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--query-workers", type=int, default=0)
        s.add_argument(
            "--query-blas-threads",
            "--blas-threads",
            dest="query_blas_threads",
            type=int,
            default=0,
            help=(
                "BLAS threads for main query scoring; 0 keeps the runtime "
                "default for bestmatch/GSEA and uses 1 for outer-threaded ANDES"
            ),
        )
        s.add_argument(
            "--worker-blas-threads",
            type=int,
            default=1,
            help="BLAS threads per null-cache worker or serial null builder",
        )
        s.add_argument("--chunk-size", type=int, default=0)
        s.add_argument(
            "--seed",
            type=int,
            default=12345,
            help="random seed for null cache; use -1 for OS entropy",
        )
        s.add_argument(
            "--numba-threshold",
            type=int,
            default=0,
            help="ANDES only: use Numba BMA scorer below this set-size product",
        )
        s.add_argument(
            "--query-mode",
            choices=["batched", "pairwise", "bestmatch"],
            default="bestmatch",
            help="ANDES only: true-score mode",
        )
        s.add_argument(
            "--null-mode",
            choices=["pairwise", "prefix"],
            default="prefix",
            help="ANDES only: null-cache builder mode",
        )
        s.add_argument(
            "--score-mode",
            choices=["batched", "bestmatch"],
            default="batched",
            help="GSEA only: true-score mode",
        )
        s.add_argument(
            "--query-memory-mb",
            type=float,
            default=1024.0,
            help=(
                "Target cap for temporary bestmatch/batched chunks; excludes "
                "persistent directed and output matrices"
            ),
        )
        s.add_argument("--cache", default="")
        s.add_argument("--skip-cache-build", action="store_true")
        s.add_argument("--verbose", action="store_true")
        s.add_argument("--json-out", default="")
        s.add_argument("--profile-out", default="")
        s.add_argument("--profile-top", type=int, default=40)

    a = sub.add_parser("andes")
    common(a)
    a.add_argument("--geneset1", required=True)
    a.add_argument("--geneset2", required=True)
    a.add_argument("--limit-terms1", type=int, default=0)
    a.add_argument("--limit-terms2", type=int, default=0)
    a.set_defaults(func=run_andes)

    g = sub.add_parser("gsea")
    common(g)
    g.add_argument("--geneset", required=True)
    g.add_argument("--rankedlist", required=True)
    g.add_argument("--limit-terms", type=int, default=0)
    g.set_defaults(func=run_gsea)

    ao = sub.add_parser("andes_old")
    common(ao)
    ao.add_argument("--geneset1", required=True)
    ao.add_argument("--geneset2", required=True)
    ao.add_argument("--limit-terms1", type=int, default=0)
    ao.add_argument("--limit-terms2", type=int, default=0)
    ao.set_defaults(func=run_andes_old)

    go = sub.add_parser("gsea_old")
    common(go)
    go.add_argument("--geneset", required=True)
    go.add_argument("--rankedlist", required=True)
    go.add_argument("--limit-terms", type=int, default=0)
    go.set_defaults(func=run_gsea_old)

    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.query_workers < 0:
        p.error("--query-workers must be non-negative")
    if args.query_blas_threads < 0:
        p.error("--query-blas-threads must be non-negative")
    if args.worker_blas_threads < 1:
        p.error("--worker-blas-threads must be at least 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    report = run_with_optional_profile(args)
    write_report(report, args.json_out)
