#!/usr/bin/env python3
"""
Run the PI-requested ANDES enrichment benchmarks.

This benchmarks the query GMT against the slow shared background GMTs and writes
one JSON report per case plus a combined summary CSV. It uses the same staged
timing code as bench_end_to_end.py, so the output separates load, normalize,
cache build, term blocks, query scoring, and save time.
"""

import argparse
import csv
import json
import os
import time
import traceback
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bench_end_to_end import Timer, load_embedding, maybe_limit, run_andes, write_report
from andes import data as ld
from andes import bma as func


DEFAULT_CASES = {
    "go_bp": "/grain/resources/gene_ontology/output/2025-03-16/hsa_ALL_BP_direct.gmt",
    "drugbank": "/grain/ll84/setmatch/data/gmt/drugbank.2301.gmt",
    "kegg_cpdb": "/grain/ll84/setmatch/data/gmt/KEGG_CPDB.gmt",
    "omim": "/grain/ll84/setmatch/data/gmt/omim.20231030.prop.gmt",
}

STAGE_NAMES = [
    "load",
    "normalize",
    "numba_warmup",
    "gene_sets",
    "cache_build",
    "term_blocks",
    "query_scoring",
    "save",
]


def parse_case_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("custom cases must be name=/path/to/file.gmt")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("custom cases must be name=/path/to/file.gmt")
    return name, path


def selected_cases(args) -> Dict[str, str]:
    cases = dict(DEFAULT_CASES)
    for name, path in args.extra_case:
        cases[name] = path
    if not args.case or args.case == ["all"]:
        return cases
    selected = {}
    for name in args.case:
        if name not in cases:
            raise SystemExit(f"Unknown case {name!r}. Available: {', '.join(sorted(cases))}")
        selected[name] = cases[name]
    return selected


def make_andes_args(args, case_name: str, background_gmt: str) -> Namespace:
    out_dir = Path(args.out_dir)
    score_out = "" if args.no_scores else str(out_dir / "scores" / f"{case_name}.csv")
    cache = "" if args.no_cache else str(out_dir / "cache" / f"{case_name}.null")
    return Namespace(
        cmd="andes",
        emb=args.emb,
        genelist=args.genelist,
        geneset1=args.query_gmt,
        geneset2=background_gmt,
        out=score_out,
        min_size=args.min_size,
        max_size=args.max_size,
        ite=args.ite,
        workers=args.workers,
        query_workers=args.query_workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        numba_threshold=args.numba_threshold,
        query_mode=args.query_mode,
        query_memory_mb=args.query_memory_mb,
        cache=cache,
        skip_cache_build=args.skip_cache_build,
        verbose=args.verbose,
        json_out=str(out_dir / "json" / f"{case_name}.json"),
        profile_out="",
        profile_top=40,
        limit_terms1=args.limit_queries,
        limit_terms2=args.limit_background,
        func=run_andes,
    )


def row_from_report(case_name: str, background_gmt: str, report: dict, elapsed_s: float) -> dict:
    timing = report.get("timing", {})
    stages = timing.get("stages", {})
    shared = report.get("shared_timing", {}).get("stages", {})
    n_pairs = int(report.get("n_scored_pairs") or report.get("n_pairs") or 0)
    query_s = float(stages.get("query_scoring", 0.0))
    total_s = float(timing.get("total_s", elapsed_s))
    row = {
        "case": case_name,
        "status": "ok",
        "background_gmt": background_gmt,
        "n_terms_query": report.get("n_terms1", ""),
        "n_terms_background": report.get("n_terms2", ""),
        "n_pairs": report.get("n_pairs", ""),
        "n_scored_pairs": report.get("n_scored_pairs", ""),
        "n_size_pairs": report.get("n_size_pairs", ""),
        "symmetric": report.get("symmetric", ""),
        "query_mode": report.get("query_mode", ""),
        "query_workers": report.get("query_workers", ""),
        "query_workspace_mb": report.get("query_workspace_mb", ""),
        "total_s": total_s,
        "pairs_per_s_query": (n_pairs / query_s) if query_s > 0 else "",
        "pairs_per_s_total": (n_pairs / total_s) if total_s > 0 else "",
        "error": "",
    }
    for stage in ("load", "normalize", "numba_warmup", "query_gene_sets"):
        row[f"shared_{stage}_s"] = shared.get(stage, "")
    for stage in STAGE_NAMES:
        row[f"{stage}_s"] = stages.get(stage, "")
    return row


def error_row(case_name: str, background_gmt: str, exc: BaseException, elapsed_s: float) -> dict:
    row = {
        "case": case_name,
        "status": "error",
        "background_gmt": background_gmt,
        "n_terms_query": "",
        "n_terms_background": "",
        "n_pairs": "",
        "n_scored_pairs": "",
        "n_size_pairs": "",
        "symmetric": "",
        "query_mode": "",
        "query_workers": "",
        "query_workspace_mb": "",
        "total_s": elapsed_s,
        "pairs_per_s_query": "",
        "pairs_per_s_total": "",
        "error": repr(exc),
    }
    for stage in ("load", "normalize", "numba_warmup", "query_gene_sets"):
        row[f"shared_{stage}_s"] = ""
    for stage in STAGE_NAMES:
        row[f"{stage}_s"] = ""
    return row


def write_summary(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "status",
        "background_gmt",
        "n_terms_query",
        "n_terms_background",
        "n_pairs",
        "n_scored_pairs",
        "n_size_pairs",
        "symmetric",
        "query_mode",
        "query_workers",
        "query_workspace_mb",
        "total_s",
        "pairs_per_s_query",
        "pairs_per_s_total",
        "shared_load_s",
        "shared_normalize_s",
        "shared_numba_warmup_s",
        "shared_query_gene_sets_s",
        *[f"{stage}_s" for stage in STAGE_NAMES],
        "error",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary to {path}")


def build_shared_context(args):
    timer = Timer()
    seed = func.BmaNullBuilder.resolve_seed(args.seed)

    t = time.perf_counter()
    raw, node_list = load_embedding(args.emb, args.genelist)
    timer.record("load", t)

    t = time.perf_counter()
    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    del raw
    timer.record("normalize", t)

    t = time.perf_counter()
    func.warmup_numba()
    timer.record("numba_warmup", t)

    t = time.perf_counter()
    node_set = set(node_list)
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(node_list)})
    query_gmt = ld.load_gmt(args.query_gmt)
    idx1 = func.preconvert_indices_to_arrays(
        ld.term2indexes(query_gmt, g_node2index, upper=args.max_size, lower=args.min_size)
    )
    terms1 = maybe_limit(list(idx1.keys()), args.limit_queries)
    if not terms1:
        raise ValueError("no query terms passed size filters")
    bg1 = np.asarray(func.get_background_indices(query_gmt, node_set, g_node2index), dtype=np.int32)
    query_blocks = func.precompute_term_embedding_blocks(E_unit, {term: idx1[term] for term in terms1})
    timer.record("query_gene_sets", t)

    print("\nShared benchmark setup")
    print(f"  embedding: {E_unit.shape[0]} genes x {E_unit.shape[1]} dims")
    print(f"  query terms: {len(terms1)}")
    timer.print()

    return {
        "seed": seed,
        "E_unit": E_unit,
        "node_set": node_set,
        "g_node2index": g_node2index,
        "query_gmt": query_gmt,
        "idx1": idx1,
        "terms1": terms1,
        "bg1": bg1,
        "query_blocks": query_blocks,
        "timing": timer.as_dict(),
    }


def load_or_build_bma_cache(args, cache_path, E_unit, bg1, bg2, size_pairs, seed):
    cache = func.BmaNullBuilder()
    if args.skip_cache_build:
        for pair in size_pairs:
            cache.cache[pair] = (0.0, 1.0)
        return cache

    expected = func.BmaNullBuilder.build_metadata(E_unit, bg1, bg2, args.ite, seed)
    cache_loaded = False
    if cache_path and os.path.exists(cache_path):
        cache.load_artifact(cache_path)
        metadata_ok, reason = cache.metadata_matches(expected)
        missing = [pair for pair in size_pairs if pair not in cache.cache]
        cache_loaded = metadata_ok and not missing
        if not cache_loaded and args.verbose:
            detail = reason if not metadata_ok else f"{len(missing)} missing pairs"
            print(f"Ignoring BMA cache: {detail}")

    if cache_loaded:
        return cache

    if args.workers > 1:
        cache.precompute_parallel(
            E_unit,
            bg1,
            size_pairs,
            ite=args.ite,
            seed=seed,
            verbose=args.verbose,
            n_workers=args.workers,
            chunk_size=None if args.chunk_size <= 0 else args.chunk_size,
            population_idx2=bg2,
        )
    else:
        cache.precompute(
            E_unit,
            bg1,
            size_pairs,
            ite=args.ite,
            seed=seed,
            verbose=args.verbose,
            population_idx2=bg2,
        )
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache.save_artifact(cache_path, overwrite=Path(cache_path).exists())
    return cache


def run_shared_case(args, context, case_name, background_gmt):
    timer = Timer()
    out_dir = Path(args.out_dir)
    E_unit = context["E_unit"]
    node_set = context["node_set"]
    g_node2index = context["g_node2index"]
    terms1 = context["terms1"]
    idx1 = context["idx1"]
    bg1 = context["bg1"]
    blocks1 = context["query_blocks"]
    seed = context["seed"]

    t = time.perf_counter()
    geneset2 = ld.load_gmt(background_gmt)
    idx2 = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset2, g_node2index, upper=args.max_size, lower=args.min_size)
    )
    terms2 = maybe_limit(list(idx2.keys()), args.limit_background)
    if not terms2:
        raise ValueError("no background terms passed size filters")
    bg2 = np.asarray(func.get_background_indices(geneset2, node_set, g_node2index), dtype=np.int32)
    timer.record("gene_sets", t)

    t = time.perf_counter()
    sizes1 = {len(idx1[t]) for t in terms1}
    sizes2 = {len(idx2[t]) for t in terms2}
    size_pairs = {(m, k) for m in sizes1 for k in sizes2}
    cache_path = "" if args.no_cache else str(out_dir / "cache" / f"{case_name}.null")
    cache = load_or_build_bma_cache(args, cache_path, E_unit, bg1, bg2, size_pairs, seed)
    timer.record("cache_build", t)

    t = time.perf_counter()
    blocks2 = func.precompute_term_embedding_blocks(E_unit, {term: idx2[term] for term in terms2})
    timer.record("term_blocks", t)

    t = time.perf_counter()
    symmetric = np.array_equal(bg1, bg2) and func.same_index_arrays_by_term(terms1, idx1, terms2, idx2)
    query_workers = args.workers if args.query_workers <= 0 else args.query_workers
    effective_query_workers = query_workers
    workspace_mb = 0.0
    if args.query_mode == "batched":
        zscores, effective_query_workers, workspace_mb = func.score_bma_zscore_matrix_batched(
            terms1,
            terms2,
            cache,
            blocks1,
            blocks2,
            symmetric=symmetric,
            n_workers=query_workers,
            max_workspace_mb=args.query_memory_mb,
            show_progress=args.verbose,
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
            n_workers=query_workers,
            numba_threshold=args.numba_threshold,
            show_progress=args.verbose,
        )
    timer.record("query_scoring", t)

    t = time.perf_counter()
    score_out = "" if args.no_scores else str(out_dir / "scores" / f"{case_name}.csv")
    if score_out:
        Path(score_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(zscores, index=terms1, columns=terms2).to_csv(score_out)
    timer.record("save", t)

    print(f"ANDES terms: {len(terms1)} x {len(terms2)}")
    print(f"Unique size pairs: {len(size_pairs)}")
    timer.print()
    return {
        "mode": "andes",
        "case": case_name,
        "config": vars(args).copy(),
        "shared_timing": context["timing"],
        "n_terms1": len(terms1),
        "n_terms2": len(terms2),
        "n_pairs": len(terms1) * len(terms2),
        "n_scored_pairs": len(terms1) * (len(terms1) + 1) // 2 if symmetric else len(terms1) * len(terms2),
        "n_size_pairs": len(size_pairs),
        "symmetric": symmetric,
        "query_mode": args.query_mode,
        "query_workers": effective_query_workers,
        "query_workspace_mb": workspace_mb,
        "embedding_shape": list(E_unit.shape),
        "timing": timer.as_dict(),
    }


def run_suite_shared(args):
    out_dir = Path(args.out_dir)
    for subdir in ("json", "scores", "cache", "logs"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    rows = []
    cases = selected_cases(args)
    summary_path = out_dir / "summary.csv"
    context = build_shared_context(args)

    for case_name, background_gmt in cases.items():
        print("\n" + "=" * 80)
        print(f"PI benchmark case: {case_name}")
        print(f"Background: {background_gmt}")
        print("=" * 80)
        started = time.perf_counter()
        try:
            report = run_shared_case(args, context, case_name, background_gmt)
            json_out = out_dir / "json" / f"{case_name}.json"
            write_report(report, str(json_out))
            rows.append(row_from_report(case_name, background_gmt, report, time.perf_counter() - started))
        except Exception as exc:
            traceback.print_exc()
            error_payload = {
                "case": case_name,
                "background_gmt": background_gmt,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            error_path = out_dir / "json" / f"{case_name}.error.json"
            with open(error_path, "w") as fh:
                json.dump(error_payload, fh, indent=2)
            rows.append(error_row(case_name, background_gmt, exc, time.perf_counter() - started))
            if args.fail_fast:
                write_summary(rows, summary_path)
                raise
        write_summary(rows, summary_path)

    suite_path = out_dir / "json" / "suite_context.json"
    with open(suite_path, "w") as fh:
        json.dump(
            {
                "shared_timing": context["timing"],
                "n_query_terms": len(context["terms1"]),
                "embedding_shape": list(context["E_unit"].shape),
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    return rows


def run_suite(args):
    out_dir = Path(args.out_dir)
    for subdir in ("json", "scores", "cache", "logs"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    rows = []
    cases = selected_cases(args)
    summary_path = out_dir / "summary.csv"

    for case_name, background_gmt in cases.items():
        print("\n" + "=" * 80)
        print(f"PI benchmark case: {case_name}")
        print(f"Background: {background_gmt}")
        print("=" * 80)
        started = time.perf_counter()
        bench_args = make_andes_args(args, case_name, background_gmt)
        try:
            report = run_andes(bench_args)
            write_report(report, bench_args.json_out)
            rows.append(row_from_report(case_name, background_gmt, report, time.perf_counter() - started))
        except Exception as exc:
            traceback.print_exc()
            error_payload = {
                "case": case_name,
                "background_gmt": background_gmt,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            error_path = out_dir / "json" / f"{case_name}.error.json"
            with open(error_path, "w") as fh:
                json.dump(error_payload, fh, indent=2)
            rows.append(error_row(case_name, background_gmt, exc, time.perf_counter() - started))
            if args.fail_fast:
                write_summary(rows, summary_path)
                raise
        write_summary(rows, summary_path)

    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark PI-requested ANDES test cases")
    p.add_argument("--emb", default="data/embedding/node2vec_consensus.csv")
    p.add_argument("--genelist", default="data/embedding/consensus_node.txt")
    p.add_argument("--query-gmt", default="/grain/rad4/github/ANDES/rd_test_data/test_cases.gmt")
    p.add_argument("--out-dir", default="remote_benchmarks/pi_cases")
    p.add_argument(
        "--case",
        action="append",
        help="case to run; can be repeated. Default: all. Built-ins: all, go_bp, drugbank, kegg_cpdb, omim",
    )
    p.add_argument("--extra-case", action="append", type=parse_case_arg, default=[])
    p.add_argument("--min", dest="min_size", type=int, default=10)
    p.add_argument("--max", dest="max_size", type=int, default=300)
    p.add_argument("--ite", type=int, default=1000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--query-workers", type=int, default=0)
    p.add_argument("--query-memory-mb", type=float, default=8192.0)
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--query-mode", choices=["batched", "pairwise"], default="batched")
    p.add_argument("--numba-threshold", type=int, default=0)
    p.add_argument("--limit-queries", type=int, default=0)
    p.add_argument("--limit-background", type=int, default=0)
    p.add_argument("--skip-cache-build", action="store_true")
    p.add_argument("--no-scores", action="store_true", help="benchmark without writing score matrices")
    p.add_argument("--no-cache", action="store_true", help="do not load/save cache PKLs")
    p.add_argument(
        "--no-shared-load",
        action="store_true",
        help="run each case in full isolation instead of reusing embedding/query setup",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = run_suite(args) if args.no_shared_load else run_suite_shared(args)
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"\nCompleted {ok}/{len(rows)} benchmark cases successfully")


if __name__ == "__main__":
    main()
