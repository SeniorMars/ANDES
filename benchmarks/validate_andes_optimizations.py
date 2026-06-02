#!/usr/bin/env python3
"""
Validate optimized ANDES against old-ANDES statistical targets.

This script is intentionally diagnostic, not a speed benchmark. It checks:

1. Exact true-score equivalence:
   reference BMA from direct pair matrices vs optimized bestmatch all-vs-all.

2. Null-cache sanity:
   per-size-pair null estimates vs prefix-coupled null estimates for selected
   size pairs. These should be statistically close, not bit-for-bit identical.

3. End-to-end z-score agreement:
   old-style per-term-pair Monte Carlo z-scores vs optimized prefix-cache
   z-scores on a small subset. This should show high correlation/rank stability,
   but exact equality is not expected.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import load_data as ld
import func_optimized as func


def bma_from_indices(E_unit, x_idx, y_idx):
    A = E_unit[np.asarray(x_idx, dtype=np.int32)] @ E_unit[
        np.asarray(y_idx, dtype=np.int32)
    ].T
    return float((A.max(axis=1).sum() + A.max(axis=0).sum()) / (A.shape[0] + A.shape[1]))


def load_inputs(args):
    raw = np.loadtxt(args.emb, delimiter=",", dtype=np.float32)
    with open(args.genelist) as fh:
        genes = [line.strip() for line in fh]
    if raw.shape[0] != len(genes):
        raise ValueError("embedding rows do not match genelist length")

    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    node_set = set(genes)
    node2idx = defaultdict(lambda: -1, {g: i for i, g in enumerate(genes)})

    geneset1 = ld.load_gmt(args.geneset1)
    geneset2 = ld.load_gmt(args.geneset2)
    idx1 = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset1, node2idx, upper=args.max_size, lower=args.min_size)
    )
    idx2 = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset2, node2idx, upper=args.max_size, lower=args.min_size)
    )
    terms1_all = list(idx1.keys())
    terms2_all = list(idx2.keys())
    terms1 = terms1_all if args.limit_terms1 <= 0 else terms1_all[: args.limit_terms1]
    terms2 = terms2_all if args.limit_terms2 <= 0 else terms2_all[: args.limit_terms2]
    if not terms1 or not terms2:
        raise ValueError("no terms passed size filters")

    pop1 = np.asarray(func.get_background_indices(geneset1, node_set, node2idx), dtype=np.int32)
    pop2 = np.asarray(func.get_background_indices(geneset2, node_set, node2idx), dtype=np.int32)
    return E_unit, terms1, terms2, idx1, idx2, pop1, pop2


def finite_metrics(reference, observed, label, top_ks=(50, 100, 500)):
    reference = np.asarray(reference, dtype=np.float64).ravel()
    observed = np.asarray(observed, dtype=np.float64).ravel()
    mask = np.isfinite(reference) & np.isfinite(observed)
    ref = reference[mask]
    obs = observed[mask]
    out = {"label": label, "n": int(mask.sum())}
    if ref.size < 3:
        out.update({"pearson": float("nan"), "spearman": float("nan")})
        return out

    out["pearson"] = float(pearsonr(ref, obs).statistic)
    out["spearman"] = float(spearmanr(ref, obs).statistic)
    diff = obs - ref
    out["max_abs_diff"] = float(np.max(np.abs(diff)))
    out["mean_abs_diff"] = float(np.mean(np.abs(diff)))
    out["median_abs_diff"] = float(np.median(np.abs(diff)))

    overlaps = {}
    for k in top_ks:
        if ref.size >= k:
            ref_top = set(np.argpartition(ref, -k)[-k:])
            obs_top = set(np.argpartition(obs, -k)[-k:])
            overlaps[str(k)] = float(len(ref_top & obs_top) / k)
    out["top_k_overlap"] = overlaps
    return out


def print_metrics(metrics):
    print(f"\n[{metrics['label']}]")
    print(f"  n               {metrics['n']}")
    print(f"  Pearson r       {metrics.get('pearson', float('nan')):.6f}")
    print(f"  Spearman rho    {metrics.get('spearman', float('nan')):.6f}")
    if "max_abs_diff" in metrics:
        print(f"  max |delta|     {metrics['max_abs_diff']:.6g}")
        print(f"  mean |delta|    {metrics['mean_abs_diff']:.6g}")
        print(f"  median |delta|  {metrics['median_abs_diff']:.6g}")
    for k, overlap in metrics.get("top_k_overlap", {}).items():
        print(f"  top-{k} overlap  {overlap:.1%}")


def validate_true_scores(E_unit, terms1, terms2, idx1, idx2, args):
    print("\n== True-score equivalence ==")
    reference = np.empty((len(terms1), len(terms2)), dtype=np.float32)
    for i, t1 in enumerate(terms1):
        for j, t2 in enumerate(terms2):
            reference[i, j] = bma_from_indices(E_unit, idx1[t1], idx2[t2])

    cache = func.NullCacheBMA()
    sizes1 = {len(idx1[t]) for t in terms1}
    sizes2 = {len(idx2[t]) for t in terms2}
    for m in sizes1:
        for k in sizes2:
            cache.cache[(int(m), int(k))] = (0.0, 1.0)

    blocks1 = func.precompute_term_embedding_blocks(E_unit, {t: idx1[t] for t in terms1})
    blocks2 = func.precompute_term_embedding_blocks(E_unit, {t: idx2[t] for t in terms2})
    symmetric = func.same_index_arrays_by_term(terms1, idx1, terms2, idx2)
    observed, stats = func.score_bma_zscore_matrix_bestmatch(
        E_unit,
        terms1,
        terms2,
        idx1,
        idx2,
        cache,
        blocks1=blocks1,
        blocks2=blocks2,
        symmetric=symmetric,
        max_workspace_mb=args.query_memory_mb,
        show_progress=args.verbose,
    )
    metrics = finite_metrics(reference, observed, "true scores: direct BMA vs bestmatch")
    metrics["bestmatch_stats"] = stats
    print_metrics(metrics)
    print(f"  symmetric reuse {stats.get('symmetric_reuse', False)}")
    return metrics, reference, observed


def parse_size_pairs(text):
    pairs = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        for sep in ("x", ":", "-"):
            if sep in token:
                left, right = token.split(sep, 1)
                pairs.append((int(left), int(right)))
                break
        else:
            raise ValueError(f"cannot parse size pair {token!r}; use e.g. 10x10,25x50")
    return pairs


def choose_size_pairs(terms1, terms2, idx1, idx2, args):
    if args.null_pairs:
        return sorted(set(parse_size_pairs(args.null_pairs)))

    all_pairs = sorted(
        {
            (int(len(idx1[t1])), int(len(idx2[t2])))
            for t1 in terms1
            for t2 in terms2
        },
        key=lambda p: (p[0] * p[1], p[0], p[1]),
    )
    if len(all_pairs) <= args.null_pair_count:
        return all_pairs
    positions = np.linspace(0, len(all_pairs) - 1, args.null_pair_count, dtype=int)
    return sorted({all_pairs[int(pos)] for pos in positions})


def validate_nulls(E_unit, terms1, terms2, idx1, idx2, pop1, pop2, args):
    print("\n== Null-cache sanity ==")
    pairs = choose_size_pairs(terms1, terms2, idx1, idx2, args)
    print(f"Selected size pairs: {pairs}")

    pairwise = func.NullCacheBMA()
    pairwise.precompute(
        E_unit,
        pop1,
        set(pairs),
        ite=args.null_ite,
        seed=args.seed,
        population_idx2=pop2,
        verbose=args.verbose,
    )
    prefix = func.NullCacheBMA()
    prefix.precompute_prefix(
        E_unit,
        pop1,
        set(pairs),
        ite=args.null_ite,
        seed=args.seed,
        population_idx2=pop2,
        verbose=args.verbose,
    )

    rows = []
    print("\n  pair      mu_pairwise   mu_prefix    delta_mu     sd_pairwise   sd_prefix    delta_sd")
    for pair in pairs:
        mu_a, sd_a = pairwise.cache[pair]
        mu_b, sd_b = prefix.cache[pair]
        row = {
            "pair": list(pair),
            "pairwise_mu": float(mu_a),
            "prefix_mu": float(mu_b),
            "delta_mu": float(mu_b - mu_a),
            "pairwise_sd": float(sd_a),
            "prefix_sd": float(sd_b),
            "delta_sd": float(sd_b - sd_a),
        }
        rows.append(row)
        print(
            f"  {pair!s:10s} {mu_a:11.6f} {mu_b:11.6f} {mu_b - mu_a:11.6f}"
            f" {sd_a:12.6f} {sd_b:11.6f} {sd_b - sd_a:11.6f}"
        )
    return {"ite": args.null_ite, "pairs": rows}


def old_style_zscores(E_unit, terms1, terms2, idx1, idx2, pop1, pop2, ite, seed):
    rng = random.Random(seed)
    pop1_list = list(map(int, pop1))
    pop2_list = list(map(int, pop2))
    zscores = np.empty((len(terms1), len(terms2)), dtype=np.float32)
    true_scores = np.empty_like(zscores)
    for i, t1 in enumerate(terms1):
        x_idx = np.asarray(idx1[t1], dtype=np.int32)
        m = len(x_idx)
        for j, t2 in enumerate(terms2):
            y_idx = np.asarray(idx2[t2], dtype=np.int32)
            k = len(y_idx)
            true_score = bma_from_indices(E_unit, x_idx, y_idx)
            back = np.empty(ite, dtype=np.float64)
            for n in range(ite):
                rx = rng.sample(pop1_list, m)
                ry = rng.sample(pop2_list, k)
                back[n] = bma_from_indices(E_unit, rx, ry)
            mu = back.mean()
            sd = back.std(ddof=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (true_score - mu) / sd
            true_scores[i, j] = true_score
            zscores[i, j] = z
    return true_scores, zscores


def optimized_zscores(E_unit, terms1, terms2, idx1, idx2, pop1, pop2, args):
    size_pairs = {
        (int(len(idx1[t1])), int(len(idx2[t2])))
        for t1 in terms1
        for t2 in terms2
    }
    cache = func.NullCacheBMA()
    cache.precompute_prefix(
        E_unit,
        pop1,
        size_pairs,
        ite=args.z_ite,
        seed=args.seed,
        population_idx2=pop2,
        verbose=args.verbose,
    )
    blocks1 = func.precompute_term_embedding_blocks(E_unit, {t: idx1[t] for t in terms1})
    blocks2 = func.precompute_term_embedding_blocks(E_unit, {t: idx2[t] for t in terms2})
    symmetric = func.same_index_arrays_by_term(terms1, idx1, terms2, idx2)
    zscores, stats = func.score_bma_zscore_matrix_bestmatch(
        E_unit,
        terms1,
        terms2,
        idx1,
        idx2,
        cache,
        blocks1=blocks1,
        blocks2=blocks2,
        symmetric=symmetric,
        max_workspace_mb=args.query_memory_mb,
        show_progress=args.verbose,
    )
    return zscores, stats


def validate_zscores(E_unit, terms1, terms2, idx1, idx2, pop1, pop2, args):
    print("\n== End-to-end z-score agreement ==")
    print(
        f"Subset: {len(terms1)} x {len(terms2)} pairs; "
        f"old-style ite={args.z_ite}, optimized prefix ite={args.z_ite}"
    )
    _, old_z = old_style_zscores(
        E_unit,
        terms1,
        terms2,
        idx1,
        idx2,
        pop1,
        pop2,
        ite=args.z_ite,
        seed=args.seed,
    )
    new_z, stats = optimized_zscores(E_unit, terms1, terms2, idx1, idx2, pop1, pop2, args)
    metrics = finite_metrics(old_z, new_z, "z-scores: old-style MC vs optimized prefix")
    metrics["bestmatch_stats"] = stats
    print_metrics(metrics)
    print("  note: exact equality is not expected; old nulls are independent per term pair.")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate optimized ANDES true scores, prefix nulls, and z-scores"
    )
    p.add_argument("--emb", required=True)
    p.add_argument("--genelist", required=True)
    p.add_argument("--geneset1", required=True)
    p.add_argument("--geneset2", required=True)
    p.add_argument("--min", dest="min_size", type=int, default=10)
    p.add_argument("--max", dest="max_size", type=int, default=300)
    p.add_argument("--limit-terms1", type=int, default=50)
    p.add_argument("--limit-terms2", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--null-ite", type=int, default=2000)
    p.add_argument("--z-ite", type=int, default=300)
    p.add_argument("--null-pairs", default="", help="comma list like 10x10,25x50")
    p.add_argument("--null-pair-count", type=int, default=5)
    p.add_argument("--query-memory-mb", type=float, default=1024.0)
    p.add_argument("--skip-null", action="store_true")
    p.add_argument("--skip-zscore", action="store_true")
    p.add_argument("--json-out", default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    E_unit, terms1, terms2, idx1, idx2, pop1, pop2 = load_inputs(args)
    print(f"Embedding: {E_unit.shape}")
    print(f"Terms: {len(terms1)} x {len(terms2)}")
    print(f"Backgrounds: {len(pop1)} x {len(pop2)}")

    report = {
        "config": vars(args).copy(),
        "embedding_shape": list(E_unit.shape),
        "n_terms1": len(terms1),
        "n_terms2": len(terms2),
        "background1": int(len(pop1)),
        "background2": int(len(pop2)),
    }

    true_metrics, _, _ = validate_true_scores(E_unit, terms1, terms2, idx1, idx2, args)
    report["true_score"] = true_metrics

    if not args.skip_null:
        report["null_sanity"] = validate_nulls(
            E_unit, terms1, terms2, idx1, idx2, pop1, pop2, args
        )

    if not args.skip_zscore:
        report["zscore_agreement"] = validate_zscores(
            E_unit, terms1, terms2, idx1, idx2, pop1, pop2, args
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"\nWrote validation report to {out}")


if __name__ == "__main__":
    main()
