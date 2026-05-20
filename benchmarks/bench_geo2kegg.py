"""
bench_geo2kegg.py — GEO2KEGG benchmark: GSEA-ANDES vs KEGG disease ground truth

Ground truth: geo2kegg.txt maps each GEO dataset to a disease. For that disease,
malakegg/{DISEASE}.txt lists KEGG pathways with a relevance score > 0 = positive.

Metric: AUPRC (Average Precision) per GEO dataset, then mean across all datasets.

Ranked list format (TSV, no header):
    gene_id<TAB>ranking_statistic
Genes appear in descending order of the statistic (most up-regulated first).

Usage
-----
  # Quick test on one dataset
  python bench_geo2kegg.py --geos GSE3467 --out reports/quick_bench.csv

  # Full benchmark (requires all 42 ranked list files)
  python bench_geo2kegg.py --out reports/geo2kegg.csv --workers 8 --ite 500

Outputs
-------
  {out}          — per-GEO AUPRC (CSV)
  {out}_summary  — mean ± sd AUPRC (printed + CSV)
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import load_data as ld
import func_optimized as func
from func_gsea import (
    NullCacheESBetter,
    compute_ranked_emb,
    compute_es_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────

def load_geo2kegg(path):
    """Returns {geo_id: disease_abbrev}."""
    mapping = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                mapping[tokens[0]] = tokens[-1]
    return mapping


def load_malakegg(malakegg_dir, disease):
    """Returns set of KEGG term IDs with score > 0 for this disease."""
    path = Path(malakegg_dir) / f"{disease}.txt"
    if not path.exists():
        return set()
    positives = set()
    with open(path) as f:
        f.readline()  # header
        for line in f:
            tokens = line.strip().split(",")
            if len(tokens) < 3:
                continue
            term  = tokens[0].strip('"')
            try:
                score = float(tokens[2])
            except ValueError:
                continue
            if score > 0:
                positives.add(term)
    return positives


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ranked_list(path, g_node2index, node_set):
    """Returns ranked_idx (int32 array) from a two-column TSV."""
    df = pd.read_csv(path, sep="\t", index_col=0, header=None)
    genes = [str(g) for g in df.index if str(g) in node_set]
    return np.array([g_node2index[g] for g in genes], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_gsea_andes(E_unit, pop, ranked_idx, geneset_indices_np, terms,
                     ite, seed, workers, cache_path, verbose):
    """Returns {term: z_score} using GSEA-ANDES."""
    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    sizes      = {len(geneset_indices_np[t]) for t in terms}

    expected = NullCacheESBetter.build_metadata(E_unit, pop, ranked_emb, ite, seed)
    cache    = NullCacheESBetter()

    if cache_path and Path(cache_path).exists():
        cache = NullCacheESBetter.load(cache_path)
        ok, _ = cache.metadata_matches(expected)
        if not ok:
            cache = NullCacheESBetter()

    missing = cache.missing_sizes(sizes)
    if missing:
        if workers > 1:
            cache.precompute_parallel(E_unit, pop, missing, ranked_emb,
                                      ite=ite, seed=seed, verbose=verbose,
                                      n_workers=workers)
        else:
            cache.precompute(E_unit, pop, missing, ranked_emb,
                             ite=ite, seed=seed, verbose=verbose)
        if cache_path:
            cache.save(cache_path)

    return {
        t: cache.get_zscore(compute_es_score(E_unit, geneset_indices_np[t], ranked_emb),
                            len(geneset_indices_np[t]))
        for t in terms
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUPRC
# ─────────────────────────────────────────────────────────────────────────────

def compute_auprc(scores, positive_terms, all_terms):
    labels = np.array([1 if t in positive_terms else 0 for t in all_terms])
    if labels.sum() == 0:
        return float("nan")
    preds = np.array([scores.get(t, 0.0) for t in all_terms])
    return float(average_precision_score(labels, preds))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GEO2KEGG benchmark: GSEA-ANDES AUPRC on disease-enrichment datasets"
    )
    p.add_argument("--emb",        default="data/embedding/node2vec_consensus.csv")
    p.add_argument("--genelist",   default="data/embedding/consensus_node.txt")
    p.add_argument("--geneset",    default="paper/results/enrichment_analysis/GSEA/gene_sets.gmt")
    p.add_argument("--ranked-dir", default="data/expression", dest="ranked_dir")
    p.add_argument("--geo2kegg",   default="paper/data/geo2kegg.txt")
    p.add_argument("--malakegg",   default="paper/data/malakegg")
    p.add_argument("--cache-dir",  default="reports/bench_cache", dest="cache_dir")
    p.add_argument("--out",        required=True)
    p.add_argument("--ite",        type=int, default=1000)
    p.add_argument("--workers",    type=int, default=1)
    p.add_argument("--min",        dest="min_size", type=int, default=10)
    p.add_argument("--max",        dest="max_size", type=int, default=300)
    p.add_argument("--seed",       type=int, default=12345)
    p.add_argument("--geos",       nargs="+", default=None)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    print("Loading embedding...")
    raw = np.loadtxt(args.emb, delimiter=",", dtype=np.float32)
    with open(args.genelist) as fh:
        node_list = [line.strip() for line in fh]
    assert len(node_list) == raw.shape[0]
    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    del raw
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(node_list)})
    node_set     = set(node_list)
    print(f"  {len(node_list)} genes  |  dim={E_unit.shape[1]}")

    print("\nLoading gene sets...")
    _raw = ld.load_gmt(args.geneset)
    geneset = {
        term: [g[3:] if g.startswith("hsa") else g for g in genes]
        for term, genes in _raw.items()
    }
    geneset_indices    = ld.term2indexes(geneset, g_node2index,
                                         upper=args.max_size, lower=args.min_size)
    geneset_indices_np = func.preconvert_indices_to_arrays(geneset_indices)
    all_terms          = sorted(geneset_indices_np.keys())
    all_bg_genes       = set().union(*geneset.values()) & node_set
    pop = np.array(sorted(g_node2index[g] for g in all_bg_genes), dtype=np.int32)
    print(f"  {len(all_terms)} terms  |  {len(pop)} background genes")

    geo2kegg  = load_geo2kegg(args.geo2kegg)
    malaterms = {d: load_malakegg(args.malakegg, d) for d in set(geo2kegg.values())}
    print(f"\nGEO2KEGG: {len(geo2kegg)} datasets")

    ranked_dir = Path(args.ranked_dir)
    geos_to_run = args.geos or [
        geo for geo in geo2kegg if (ranked_dir / f"{geo}_rank.txt").exists()
    ]
    missing = [g for g in geos_to_run if not (ranked_dir / f"{geo}_rank.txt").exists()]
    if missing:
        print(f"Warning: ranked list files not found for: {missing}")
        geos_to_run = [g for g in geos_to_run if g not in missing]
    print(f"Running {len(geos_to_run)}/{len(geo2kegg)} GEO datasets\n")

    rows = []
    for geo in geos_to_run:
        disease   = geo2kegg[geo]
        positives = malaterms.get(disease, set()) & set(all_terms)
        if not positives:
            print(f"  {geo} ({disease}): no positive terms — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"{geo}  ({disease})  positives={len(positives)}")

        rank_path = ranked_dir / f"{geo}_rank.txt"
        try:
            ranked_idx = load_ranked_list(rank_path, g_node2index, node_set)
        except Exception as e:
            print(f"  Error loading {rank_path}: {e} — skipping")
            continue
        if len(ranked_idx) == 0:
            print(f"  Empty ranked list — skipping")
            continue
        print(f"  ranked list: {len(ranked_idx)} genes")

        cache_path = str(Path(args.cache_dir) / f"{geo}_gsea_null.pkl")
        t0 = time.perf_counter()
        scores = score_gsea_andes(
            E_unit, pop, ranked_idx, geneset_indices_np, all_terms,
            ite=args.ite, seed=args.seed, workers=args.workers,
            cache_path=cache_path, verbose=args.verbose,
        )
        auprc = compute_auprc(scores, positives, all_terms)
        elapsed = time.perf_counter() - t0
        print(f"  GSEA-ANDES  AUPRC={auprc:.4f}  ({elapsed:.1f}s)")
        rows.append({"geo": geo, "disease": disease,
                     "n_positives": len(positives), "AUPRC": auprc})

    if not rows:
        print("\nNo datasets produced results.")
        sys.exit(1)

    df = pd.DataFrame(rows).set_index("geo")
    out_path = args.out if args.out.endswith(".csv") else args.out + ".csv"
    df.to_csv(out_path)
    print(f"\nSaved per-GEO AUPRC to {out_path}")

    vals = df["AUPRC"].dropna()
    print(f"\n{'Method':<18}  {'Mean AUPRC':>10}  {'Median':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 55)
    print(f"  {'GSEA-ANDES':<16}  {vals.mean():>10.4f}  {vals.median():>8.4f}"
          f"  {vals.std():>8.4f}  {len(vals):>4}")

    summary_path = out_path.replace(".csv", "_summary.csv")
    df[["AUPRC"]].agg(["mean", "median", "std"]).T.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
