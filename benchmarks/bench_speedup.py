"""
bench_speedup.py — Wall-time comparison: original vs optimized ANDES and GSEA-ANDES

Two experiments on synthetic data (no real files needed):

  1. ANDES (set-vs-set BMA)
     Old: build full n×n cosine_similarity matrix, sample --n-pairs pairs, score with Pool
     New: NullCacheBMA precomputes null per size-pair, score ALL pairs with batched BLAS
     Speedup is reported two ways:
       measured:      old (n_pairs) vs new (all pairs)  — new did more work
       extrapolated:  old time scaled to all_pairs vs new (all pairs)  — apples-to-apples

  2. GSEA-ANDES (ranked-list enrichment)
     Old: build full n×n cosine_similarity matrix, Pool per-term MC
     New: NullCacheESBetter, batched GEMM null build, BLAS-pinned per-term scoring
     Both score all terms — comparison is fair.

Outputs
-------
  JSON summary with wall times, speedup, Spearman ρ, Pearson r, mean |Δz|
  (table printed to stdout)

Usage
-----
  # Quick smoke (< 1 min)
  python benchmarks/bench_speedup.py --n-genes 500 --n-terms 20 --ite 50 --workers 2

  # Realistic
  python benchmarks/bench_speedup.py --n-genes 5000 --n-terms 100 --ite 200 --workers 8
"""

import argparse
import json
import os
import random
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn import metrics as sk_metrics

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import func_optimized as func_new
import set_analysis_func as func_old
from func_gsea import (
    NullCacheESBetter,
    compute_ranked_emb,
    compute_es_score,
    warmup_numba_es,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic(n_genes, dim, n_terms, min_size, max_size, seed):
    rng = np.random.default_rng(seed)
    E_raw  = rng.standard_normal((n_genes, dim)).astype(np.float32)
    E_unit = np.ascontiguousarray(func_new.l2_normalize_rows(E_raw), dtype=np.float32)

    term_indices = {}
    for i in range(n_terms):
        sz = int(rng.integers(min_size, max_size + 1))
        idx = rng.choice(n_genes, sz, replace=False).astype(np.int32)
        term_indices[f"TERM_{i:04d}"] = np.sort(idx)

    pop = np.arange(n_genes, dtype=np.int32)
    ranked_idx = rng.permutation(n_genes).astype(np.int32)
    return E_unit, term_indices, pop, ranked_idx


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: ANDES (BMA set-vs-set)
# ─────────────────────────────────────────────────────────────────────────────

def bench_andes_old(E_unit, term_indices, pop, ite, workers, sampled_pairs):
    """Old path: build full cosine_similarity matrix, Pool BMA sampling on sampled_pairs."""
    t0 = time.perf_counter()
    S = sk_metrics.pairwise.cosine_similarity(E_unit, E_unit)
    matrix_time = time.perf_counter() - t0

    pop_list = pop.tolist()
    f = partial(func_old.andes,
                matrix=S,
                g1_term2index=term_indices,
                g2_term2index=term_indices,
                g1_population=pop_list,
                g2_population=pop_list,
                ite=ite)

    t1 = time.perf_counter()
    with Pool(workers) as p:
        rets = p.map(f, sampled_pairs)
    score_time = time.perf_counter() - t1

    scores = {pair: ret[1] for pair, ret in zip(sampled_pairs, rets)}
    return scores, matrix_time, score_time, matrix_time + score_time


def bench_andes_new(E_unit, term_indices, pop, ite, workers, seed):
    """New path: NullCacheBMA + score_bma_zscore_matrix_batched (no full matrix)."""
    terms = sorted(term_indices.keys())
    sizes = {len(term_indices[t]) for t in terms}
    size_pairs = {(m, k) for m in sizes for k in sizes}

    t0 = time.perf_counter()
    cache = func_new.NullCacheBMA()
    cache.precompute_parallel(
        E_unit, pop, size_pairs,
        ite=ite, seed=seed, n_workers=workers, verbose=False,
    )
    null_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    blocks = func_new.precompute_term_embedding_blocks(E_unit, term_indices)
    zscores_arr, _, _ = func_new.score_bma_zscore_matrix_batched(
        terms, terms, cache, blocks, blocks,
        symmetric=True, n_workers=workers,
    )
    score_time = time.perf_counter() - t1

    term_to_i = {t: i for i, t in enumerate(terms)}
    scores = {
        (t1, t2): float(zscores_arr[term_to_i[t1], term_to_i[t2]])
        for t1 in terms for t2 in terms if t1 <= t2
    }
    return scores, null_time, score_time, null_time + score_time


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: GSEA-ANDES (ranked-list enrichment)
# ─────────────────────────────────────────────────────────────────────────────

def bench_gsea_old(E_unit, term_indices, pop, ranked_idx, ite, workers):
    """Old path: full S matrix, Pool per-term MC sampling."""
    t0 = time.perf_counter()
    S = sk_metrics.pairwise.cosine_similarity(E_unit, E_unit)
    matrix_time = time.perf_counter() - t0

    terms    = sorted(term_indices.keys())
    pop_list = pop.tolist()
    ranked_list = ranked_idx.tolist()

    f = partial(func_old.gsea_andes,
                ranked_list=ranked_list,
                matrix=S,
                term2indices=term_indices,
                annotated_indices=pop_list,
                ite=ite)

    t1 = time.perf_counter()
    with Pool(workers) as p:
        rets = p.map(f, terms)
    score_time = time.perf_counter() - t1

    scores = {t: ret[1] for t, ret in zip(terms, rets)}
    return scores, matrix_time, score_time, matrix_time + score_time


def bench_gsea_new(E_unit, term_indices, pop, ranked_idx, ite, workers, seed):
    """New path: NullCacheESBetter, batched GEMM null, BLAS-pinned workers."""
    terms = sorted(term_indices.keys())
    sizes = {len(term_indices[t]) for t in terms}

    t0 = time.perf_counter()
    warmup_numba_es()
    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    cache = NullCacheESBetter()
    if workers > 1:
        cache.precompute_parallel(E_unit, pop, sizes, ranked_emb,
                                  ite=ite, seed=seed, verbose=False,
                                  n_workers=workers)
    else:
        cache.precompute(E_unit, pop, sizes, ranked_emb,
                         ite=ite, seed=seed, verbose=False)
    null_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    indices_np = func_new.preconvert_indices_to_arrays(term_indices)
    scores = {
        t: cache.get_zscore(compute_es_score(E_unit, indices_np[t], ranked_emb),
                            len(indices_np[t]))
        for t in terms
    }
    score_time = time.perf_counter() - t1

    return scores, null_time, score_time, null_time + score_time


# ─────────────────────────────────────────────────────────────────────────────
# Comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_agreement(old_scores, new_scores):
    """Spearman ρ, Pearson r, mean |Δz|, sign-agreement on the common key set."""
    keys = sorted(set(old_scores) & set(new_scores))
    n = len(keys)
    if n < 3:
        return dict(n=n, rho=float("nan"), pearson_r=float("nan"),
                    mean_abs_dz=float("nan"), sign_agree=float("nan"))
    old_v = np.array([old_scores[k] for k in keys], dtype=np.float64)
    new_v = np.array([new_scores[k] for k in keys], dtype=np.float64)
    rho, _   = spearmanr(old_v, new_v)
    r,   _   = pearsonr(old_v, new_v)
    mad      = float(np.abs(new_v - old_v).mean())
    sign_agr = float((np.sign(old_v) == np.sign(new_v)).mean())
    return dict(n=n, rho=float(rho), pearson_r=float(r),
                mean_abs_dz=mad, sign_agree=sign_agr)


def _fmt(t):
    return f"{t/60:.1f} min" if t >= 60 else f"{t:.1f} s"


def print_summary(rows):
    hdr = (f"{'Experiment':<28} {'Old':>9} {'New':>9} "
           f"{'Speedup':>8} {'(extrap)':>9} {'Spearman ρ':>11}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        extrap = f"{r['speedup_extrap']:.1f}x" if r.get("speedup_extrap") else "  n/a  "
        print(f"  {r['name']:<26} {_fmt(r['old_total']):>9} {_fmt(r['new_total']):>9}"
              f" {r['speedup']:>7.1f}x {extrap:>9} {r['rho']:>11.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Speedup benchmark: original vs optimized ANDES and GSEA-ANDES"
    )
    p.add_argument("--n-genes",   type=int, default=2000)
    p.add_argument("--dim",       type=int, default=128)
    p.add_argument("--n-terms",   type=int, default=50)
    p.add_argument("--min-size",  type=int, default=15)
    p.add_argument("--max-size",  type=int, default=150)
    p.add_argument("--n-pairs",   type=int, default=200,
                   help="ANDES pairs scored by old path (new always scores all pairs); "
                        "set 0 to use all pairs in both")
    p.add_argument("--ite",       type=int, default=100)
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--skip-andes", action="store_true")
    p.add_argument("--skip-gsea",  action="store_true")
    p.add_argument("--json-out",   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    matrix_gb = args.n_genes ** 2 * 4 / 1e9
    print(f"\nSynthetic: {args.n_genes} genes × dim={args.dim}  "
          f"|  {args.n_terms} terms  |  ite={args.ite}  |  workers={args.workers}")
    print(f"Full S matrix would be: {matrix_gb:.2f} GB")

    E_unit, term_indices, pop, ranked_idx = make_synthetic(
        args.n_genes, args.dim, args.n_terms,
        args.min_size, args.max_size, args.seed,
    )

    rows = []

    # ── ANDES ─────────────────────────────────────────────────────────────────
    if not args.skip_andes:
        terms = sorted(term_indices.keys())
        all_pairs = [(t1, t2) for t1 in terms for t2 in terms if t1 <= t2]
        n_all = len(all_pairs)
        n_sample = n_all if args.n_pairs <= 0 else min(args.n_pairs, n_all)
        sampled = random.Random(args.seed).sample(all_pairs, n_sample)

        print(f"\n{'='*60}")
        print(f"ANDES (BMA) — {n_all} total pairs, old scores {n_sample}, new scores all")
        print("  [OLD] full cosine_similarity matrix + Pool BMA ...")
        try:
            old_scores, old_mat, old_sc, old_tot = bench_andes_old(
                E_unit, term_indices, pop, args.ite, args.workers, sampled)
            print(f"        matrix {_fmt(old_mat)}  +  scoring {_fmt(old_sc)}"
                  f"  =  {_fmt(old_tot)}  ({n_sample} pairs)")

            # Extrapolate old time to all pairs (scoring scales linearly; matrix cost is fixed)
            old_sc_per_pair = old_sc / n_sample if n_sample > 0 else 0.0
            old_extrap = old_mat + old_sc_per_pair * n_all

            print("  [NEW] NullCacheBMA + batched BLAS (no full matrix) ...")
            new_scores, new_null, new_sc, new_tot = bench_andes_new(
                E_unit, term_indices, pop, args.ite, args.workers, args.seed)
            print(f"        null {_fmt(new_null)}  +  scoring {_fmt(new_sc)}"
                  f"  =  {_fmt(new_tot)}  ({n_all} pairs)")

            speedup_measured = old_tot / new_tot if new_tot > 0 else float("inf")
            speedup_extrap   = old_extrap / new_tot if new_tot > 0 else float("inf")
            agr = score_agreement(old_scores, new_scores)
            print(f"        speedup {speedup_measured:.1f}x (measured, {n_sample}/{n_all} pairs old)"
                  f"  |  {speedup_extrap:.1f}x (extrapolated to {n_all} pairs)")
            print(f"        Spearman ρ={agr['rho']:.3f}  Pearson r={agr['pearson_r']:.3f}"
                  f"  mean|Δz|={agr['mean_abs_dz']:.3f}  sign-agree={agr['sign_agree']:.1%}"
                  f"  (n={agr['n']})")
            rows.append(dict(
                name="ANDES (BMA)",
                old_total=old_tot, new_total=new_tot,
                speedup=speedup_measured, speedup_extrap=speedup_extrap,
                old_matrix_s=old_mat, old_score_s=old_sc,
                old_n_pairs=n_sample, new_n_pairs=n_all,
                old_extrap_s=old_extrap,
                new_null_s=new_null, new_score_s=new_sc,
                **{f"agr_{k}": v for k, v in agr.items()},
                rho=agr["rho"],
            ))
        except Exception:
            import traceback; traceback.print_exc()

    # ── GSEA-ANDES ────────────────────────────────────────────────────────────
    if not args.skip_gsea:
        n_terms = len(term_indices)
        print(f"\n{'='*60}")
        print(f"GSEA-ANDES — {n_terms} terms (both paths score all terms)")
        print("  [OLD] full cosine_similarity matrix + Pool MC ...")
        try:
            old_scores, old_mat, old_sc, old_tot = bench_gsea_old(
                E_unit, term_indices, pop, ranked_idx, args.ite, args.workers)
            print(f"        matrix {_fmt(old_mat)}  +  scoring {_fmt(old_sc)}  =  {_fmt(old_tot)}")

            print("  [NEW] NullCacheESBetter (batched GEMM, no full matrix) ...")
            new_scores, new_null, new_sc, new_tot = bench_gsea_new(
                E_unit, term_indices, pop, ranked_idx, args.ite, args.workers, args.seed)
            print(f"        null {_fmt(new_null)}  +  scoring {_fmt(new_sc)}  =  {_fmt(new_tot)}")

            speedup = old_tot / new_tot if new_tot > 0 else float("inf")
            agr = score_agreement(old_scores, new_scores)
            print(f"        speedup {speedup:.1f}x")
            print(f"        Spearman ρ={agr['rho']:.3f}  Pearson r={agr['pearson_r']:.3f}"
                  f"  mean|Δz|={agr['mean_abs_dz']:.3f}  sign-agree={agr['sign_agree']:.1%}"
                  f"  (n={agr['n']})")
            rows.append(dict(
                name="GSEA-ANDES",
                old_total=old_tot, new_total=new_tot,
                speedup=speedup, speedup_extrap=None,
                old_matrix_s=old_mat, old_score_s=old_sc,
                new_null_s=new_null, new_score_s=new_sc,
                **{f"agr_{k}": v for k, v in agr.items()},
                rho=agr["rho"],
            ))
        except Exception:
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    if rows:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print_summary(rows)

        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))
            print(f"JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
