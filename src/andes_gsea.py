"""
gsea_andes.py

  - No full S matrix: uses E_unit (L2-normalized embeddings) + BLAS matmul
  - NullCacheES: precomputes (μ, σ) per gene-set size once, saves to disk
  - compute_ranked_emb: extracts ranked-list embedding block once per run
  - compute_es_score: per-term scoring = matmul + argmax, ~100 μs each
  - Empirical p-value uses the same fast compute_es_score (no MC per call)
  - Fixed: geneset_terms was hardcoded to 3 terms in original
  - Fixed: empirical p-value block was outside if args.empr guard
  - Added: --cache argument to save/load null distribution across runs
  - Added: --n_permutations argument (was hardcoded 100)
  - Added: BLAS thread control, Numba warmup, tqdm progress

Usage
-----
  # Basic (null built fresh each run)
  python gsea_andes_main.py \
      --emb embedding.csv --genelist genes.txt \
      --geneset db.gmt --rankedlist ranked.tsv \
      --out results/scores.csv

  # With saved null cache (skips precompute on second run)
  python gsea_andes_main.py ... --cache null_cache_es.pkl

  # With empirical p-values (requires expression file)
  python gsea_andes_main.py ... --expressionfile expr.tsv --empr
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import load_data as ld
import func_optimized as func
from func_gsea import (
    NullCacheESBetter,
    compute_ranked_emb,
    compute_es_score,
    warmup_numba_es,
)

# expression_analysis_func is only needed when --expressionfile is passed
try:
    import expression_analysis_func as expr_func
except ImportError:
    expr_func = None


def load_ranked_list(rankedlist_f, g_node2index, node_set):
    """Load a pre-ranked gene list and convert to embedding indices."""
    df = pd.read_csv(rankedlist_f, sep="\t", index_col=0, header=None)
    genes = [str(g) for g in df.index]
    return np.array([g_node2index[g] for g in genes if g in node_set], dtype=np.int32)


def ranked_list_from_expression(expression_f, g_node2index, node_set):
    """Derive ranked gene list from expression file (by differential expression)."""
    if expr_func is None:
        raise ImportError("expression_analysis_func not found in PYTHONPATH")
    data = pd.read_csv(expression_f, skiprows=1, sep="\t")
    condition = open(expression_f).readline().strip().split("\t")
    genes = expr_func.expression_data_to_ranked_list(data, condition)
    return (
        np.array([g_node2index[g] for g in genes if g in node_set], dtype=np.int32),
        data,
        condition,
    )


def load_or_build_cache(
    cache_path,
    E_unit,
    pop,
    sizes,
    ranked_emb,
    ite,
    seed,
    n_workers,
    verbose,
    rebuild=False,
):
    """Load null cache from disk if available; otherwise build and save it."""
    expected_metadata = NullCacheESBetter.build_metadata(
        E_unit, pop, ranked_emb, ite, seed
    )

    if cache_path and os.path.exists(cache_path) and not rebuild:
        cache = NullCacheESBetter.load(cache_path)
        metadata_ok, reason = cache.metadata_matches(expected_metadata)
        missing = cache.missing_sizes(sizes) if metadata_ok else list(sizes)
        if metadata_ok and not missing:
            print(f"Null cache loaded from '{cache_path}'  ({len(cache)} entries)")
            return cache
        if not metadata_ok:
            print(f"Cache metadata invalid ({reason}) — rebuilding.")
        else:
            print(f"Cache missing {len(missing)} sizes — rebuilding.")

    cache = NullCacheESBetter()

    print(
        f"\nBuilding ES null cache  ({len(sizes)} sizes, ite={ite}, workers={n_workers})..."
    )
    t0 = time.perf_counter()

    if n_workers > 1:
        cache.precompute_parallel(
            E_unit,
            pop,
            sizes,
            ranked_emb,
            ite=ite,
            seed=seed,
            verbose=verbose,
            n_workers=n_workers,
            chunk_size=max(4, len(sizes) // (n_workers * 4)),
        )
    else:
        cache.precompute(
            E_unit,
            pop,
            sizes,
            ranked_emb,
            ite=ite,
            seed=seed,
            verbose=verbose,
        )

    elapsed = time.perf_counter() - t0
    print(f"Cache built in {elapsed:.1f}s")

    if cache_path:
        cache.save(cache_path)

    return cache


def parse_args():
    p = argparse.ArgumentParser(
        description="Ranked ANDES gene-set enrichment (Approach 2: E_unit + null cache)"
    )
    p.add_argument("--emb", required=True, help="Embedding CSV (genes × dims)")
    p.add_argument("--genelist", required=True, help="Gene list (one per line)")
    p.add_argument("--geneset", required=True, help="Gene-set database (GMT)")
    p.add_argument(
        "--rankedlist", default="", help="Pre-ranked gene list (TSV, index=gene)"
    )
    p.add_argument(
        "--expressionfile",
        default="",
        help="Expression matrix (for ranked list + empirical p)",
    )
    p.add_argument("--out", required=True, help="Output CSV for z-scores")
    p.add_argument(
        "--cache",
        default="",
        help="Path to save/load null cache (auto-named from inputs if empty)",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="ignore any existing cache and rebuild it",
    )
    p.add_argument(
        "--empr",
        action="store_true",
        help="Also compute empirical p-values (requires --expressionfile)",
    )
    p.add_argument(
        "--n_permutations",
        type=int,
        default=100,
        help="Permutations for empirical p-value  (default: 100)",
    )
    p.add_argument(
        "--ite",
        type=int,
        default=1000,
        help="Monte Carlo iterations for null cache  (default: 1000)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for null-cache build  (default: 1)",
    )
    p.add_argument("--min", dest="min_size", type=int, default=10)
    p.add_argument("--max", dest="max_size", type=int, default=300)
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="random seed for null cache; use -1 for OS entropy",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.seed = NullCacheESBetter.resolve_seed(args.seed)
    print(f"Seed: {args.seed}")

    # ── Input validation ───────────────────────────────────────────────────
    if not args.rankedlist and not args.expressionfile:
        print("Error: one of --rankedlist or --expressionfile is required")
        sys.exit(1)
    if args.empr and not args.expressionfile:
        print("Error: --empr requires --expressionfile")
        sys.exit(1)
    if args.empr and expr_func is None:
        print(
            "Error: expression_analysis_func not found; cannot compute empirical p-values"
        )
        sys.exit(1)

    # ── Load & normalize embedding ─────────────────────────────────────────
    print("Loading embedding...")
    raw = np.loadtxt(args.emb, delimiter=",", dtype=np.float32)
    with open(args.genelist) as fh:
        node_list = [line.strip() for line in fh]

    if len(node_list) != raw.shape[0]:
        print("Error: embedding rows do not match gene-list length")
        sys.exit(1)

    print(f"  {len(node_list)} genes  |  dim={raw.shape[1]}")

    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    del raw
    print(
        f"  E_unit: {E_unit.nbytes / 1e6:.1f} MB  "
        f"(vs {len(node_list) ** 2 * 8 / 1e9:.1f} GB for full similarity matrix)"
    )

    # ── Numba warmup ───────────────────────────────────────────────────────
    print("\nWarming up Numba JIT...")
    func.warmup_numba()
    warmup_numba_es()

    # ── Gene-index mapping ─────────────────────────────────────────────────
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(node_list)})
    node_set = set(node_list)

    # ── Load gene sets ─────────────────────────────────────────────────────
    print("\nLoading gene sets...")
    geneset = ld.load_gmt(args.geneset)
    geneset_indices = ld.term2indexes(
        geneset, g_node2index, upper=args.max_size, lower=args.min_size
    )
    geneset_indices_np = func.preconvert_indices_to_arrays(geneset_indices)
    geneset_terms = sorted(geneset_indices_np.keys())

    # Background population
    all_bg_genes = set().union(*geneset.values()) & node_set
    pop = np.array(sorted(g_node2index[g] for g in all_bg_genes), dtype=np.int32)

    print(f"  {len(geneset_terms)} terms  |  {len(pop)} background genes")

    if not geneset_terms:
        print("Error: no gene sets passed size filters")
        sys.exit(1)

    # ── Load ranked list ───────────────────────────────────────────────────
    print("\nLoading ranked list...")
    data, condition = None, None

    if args.rankedlist:
        ranked_idx = load_ranked_list(args.rankedlist, g_node2index, node_set)
        print(f"  Loaded pre-ranked list: {len(ranked_idx)} genes")
    else:
        ranked_idx, data, condition = ranked_list_from_expression(
            args.expressionfile, g_node2index, node_set
        )
        print(f"  Derived ranked list from expression file: {len(ranked_idx)} genes")

    if len(ranked_idx) == 0:
        print("Error: ranked list is empty after filtering to embedding genes")
        sys.exit(1)

    # ── Precompute ranked embedding block ──────────────────────────────────
    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    print(f"  ranked_emb: {ranked_emb.shape}  ({ranked_emb.nbytes / 1e3:.1f} KB)")

    # ── Null cache ─────────────────────────────────────────────────────────
    sizes = {len(geneset_indices_np[t]) for t in geneset_terms}
    if not args.cache:
        args.cache = NullCacheESBetter.suggest_path("cache", E_unit, pop, ranked_emb)
        os.makedirs("cache", exist_ok=True)
        print(f"Auto cache: {args.cache}")
    cache = load_or_build_cache(
        cache_path=args.cache,
        E_unit=E_unit,
        pop=pop,
        sizes=sizes,
        ranked_emb=ranked_emb,
        ite=args.ite,
        seed=args.seed,
        n_workers=args.workers,
        verbose=args.verbose,
        rebuild=args.rebuild_cache,
    )

    # ── Score all terms ────────────────────────────────────────────────────
    print(f"\nScoring {len(geneset_terms)} terms...")
    t0 = time.perf_counter()

    true_scores = {}
    z_scores = {}

    for term in tqdm(geneset_terms, desc="Scoring", disable=not args.verbose):
        idx = geneset_indices_np[term]
        m = len(idx)
        score = compute_es_score(E_unit, idx, ranked_emb)
        true_scores[term] = score
        z_scores[term] = cache.get_zscore(score, m)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s  ({len(geneset_terms) / elapsed:.0f} terms/s)")

    # ── Save z-scores ──────────────────────────────────────────────────────
    out_df = pd.DataFrame(
        {
            "true_score": [true_scores[t] for t in geneset_terms],
            "z_score": [z_scores[t] for t in geneset_terms],
        },
        index=geneset_terms,
    )
    out_df.to_csv(args.out)
    print(f"\nSaved z-scores to {args.out}")

    # ── Empirical p-values (optional) ─────────────────────────────────────
    if args.empr:
        print(f"\nComputing empirical p-values ({args.n_permutations} permutations)...")

        if data is None or condition is None:
            # Need to reload expression data if ranked list was loaded from file
            data = pd.read_csv(args.expressionfile, skiprows=1, sep="\t")
            condition = open(args.expressionfile).readline().strip().split("\t")

        background = np.zeros(
            (args.n_permutations, len(geneset_terms)), dtype=np.float32
        )
        observed = np.array([true_scores[t] for t in geneset_terms], dtype=np.float32)

        for i in tqdm(range(args.n_permutations), desc="Permutations"):
            shuffled_genes = expr_func.expression_data_to_ranked_list_label_shuffled(
                data, condition, seed=i
            )
            shuf_idx = np.array(
                [g_node2index[g] for g in shuffled_genes if g in node_set],
                dtype=np.int32,
            )
            shuf_emb = compute_ranked_emb(E_unit, shuf_idx)

            for j, term in enumerate(geneset_terms):
                background[i, j] = compute_es_score(
                    E_unit, geneset_indices_np[term], shuf_emb
                )

        # Empirical p-value: fraction of background scores >= observed
        # (two-sided: use absolute values for symmetric null)
        empirical_p = (np.abs(background) >= np.abs(observed)).mean(axis=0)

        p_df = pd.DataFrame(
            {
                "true_score": observed,
                "z_score": [z_scores[t] for t in geneset_terms],
                "empirical_pval": empirical_p,
            },
            index=geneset_terms,
        )

        p_out = args.out.replace(".csv", "") + "_empirical_pval.csv"
        p_df.to_csv(p_out)
        print(f"Saved empirical p-values to {p_out}")


if __name__ == "__main__":
    main()
