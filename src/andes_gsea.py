"""
andes_gsea.py: ranked gene-set enrichment scoring with ES null cache

Scores gene sets against a ranked gene list using an embedding-based
enrichment score (ES).  For each gene set, the ES is the maximum signed
cumulative sum of (col_max - mean), where col_max[i] is the maximum cosine
similarity between the i-th ranked gene and any gene in the set.

Null distribution
-----------------
For each gene-set size m, (mu, sigma) are estimated from Monte Carlo
permutations of the background gene pool (NullCacheESBetter in func_gsea.py).
Permutations use prefix coupling: one matmul at max_m per iteration, with ES
extracted for every requested size by streaming through the prefix col-max.
The cache is keyed on (embedding hash, population hash, ranked-emb hash, ite,
seed) so stale entries are detected automatically.

Empirical p-values
------------------
When --empr is given together with --expressionfile, label-shuffled expression
profiles generate background ES distributions.  The empirical p-value for each
term is fraction(|background ES| >= |observed ES|).

Usage
-----
  # Null built fresh; cache auto-named from input hashes
  python andes_gsea.py \
      --emb embedding.csv --genelist genes.txt \
      --geneset db.gmt --rankedlist ranked.tsv \
      --out results/scores.csv

  # Use an explicit cache directory
  python andes_gsea.py ... --cache-dir my_cache/

  # Point to a specific cache file
  python andes_gsea.py ... --cache path/to/cache.pkl

  # Skip cache entirely (no load, no save)
  python andes_gsea.py ... --no-cache

  # Optional exact ranked best-match matrix query prototype
  python andes_gsea.py ... --score-mode bestmatch

  # The ES null cache is prefix-coupled by default:
  # one max-size random prefix updates every requested gene-set size.

  # Empirical p-values from expression permutations
  python andes_gsea.py ... --expressionfile expr.tsv --empr
"""

import os

# Respect the user's environment; only set 1 if not already set.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import load_data as ld
import func_optimized as func
from func_gsea import (
    NullCacheESBetter,
    compute_ranked_emb,
    score_terms_bestmatch,
    score_terms_batched,
    warmup_numba_es,
)

try:
    import expression_analysis_func as expr_func
except ImportError:
    expr_func = None


# I/O helpers


def load_ranked_list(rankedlist_f, g_node2index, node_set):
    """Read a pre-ranked gene list from a TSV and return embedding indices.

    Genes absent from node_set (not in the embedding) are silently dropped.

    Parameters
    ----------
    rankedlist_f : str
        Tab-separated file with gene IDs in the index column (no header).
    g_node2index : dict
        Maps gene ID strings to row indices in the embedding matrix.
    node_set : set
        Set of gene IDs present in the embedding; used to filter unknown genes.

    Returns
    -------
    numpy.ndarray of int32
        Embedding row indices in ranked order.
    """
    df = pd.read_csv(rankedlist_f, sep="\t", index_col=0, header=None)
    return np.array(
        [g_node2index[str(g)] for g in df.index if str(g) in node_set],
        dtype=np.int32,
    )


def ranked_list_from_expression(expression_f, g_node2index, node_set):
    """Derive a ranked gene list from an expression matrix.

    Delegates ranking to expression_analysis_func.expression_data_to_ranked_list.
    Returns the ranked indices plus the raw data and condition labels needed
    for label-shuffle permutations (--empr).

    Parameters
    ----------
    expression_f : str
        Expression matrix file.  First line is condition labels; remaining
        rows are tab-separated gene expression values.
    g_node2index : dict
        Maps gene ID strings to embedding row indices.
    node_set : set
        Genes present in the embedding; others are dropped.

    Returns
    -------
    idx : numpy.ndarray of int32
        Embedding indices in ranked order.
    data : pandas.DataFrame
        Raw expression table (used for permutations).
    condition : list of str
        Condition labels parsed from the first line of expression_f.
    """
    if expr_func is None:
        raise ImportError("expression_analysis_func not found in PYTHONPATH")
    data = pd.read_csv(expression_f, skiprows=1, sep="\t")
    condition = open(expression_f).readline().strip().split("\t")
    genes = expr_func.expression_data_to_ranked_list(data, condition)
    idx = np.array([g_node2index[g] for g in genes if g in node_set], dtype=np.int32)
    return idx, data, condition


# Cache


def load_or_build_cache(
    cache_path,
    E_unit,
    pop,
    sizes,
    ranked_emb,
    ite,
    seed,
    n_workers,
    blas_threads,
    es_batch_bytes,
    verbose,
    rebuild=False,
):
    """Load an existing null cache or build one from scratch.

    If cache_path exists and is not stale (metadata matches and all required
    sizes are present), the cache is returned immediately.  Otherwise the cache
    is rebuilt and, if cache_path is not None, saved to disk.

    Parameters
    ----------
    cache_path : str or None
        Path to the .pkl cache file.  None means no load and no save.
    E_unit : numpy.ndarray, float32 (N, d)
        L2-normalized embedding matrix.
    pop : numpy.ndarray, int32 (P,)
        Embedding row indices of the background gene pool.
    sizes : set of int
        Gene-set sizes that need null (mu, sigma) entries.
    ranked_emb : numpy.ndarray, float32 (L, d)
        Embedding rows for the ranked gene list, in ranked order.
    ite : int
        Monte Carlo iterations for the null distribution.
    seed : int
        Master random seed.
    n_workers : int
        Number of parallel worker processes for cache build.
    blas_threads : int
        BLAS threads per worker.
    es_batch_bytes : int
        Memory cap per worker for the ES workspace (bytes).
    verbose : bool
        Print per-size progress during build.
    rebuild : bool
        If True, ignore any existing cache file and rebuild unconditionally.

    Returns
    -------
    NullCacheESBetter
        Populated null cache.
    """
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
            blas_threads_per_worker=blas_threads,
            es_batch_bytes=es_batch_bytes,
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
            es_batch_bytes=es_batch_bytes,
        )

    print(f"Cache built in {time.perf_counter() - t0:.1f}s")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache.save(cache_path)

    return cache


# CLI


def parse_args():
    """Parse and validate CLI arguments.

    Exits with an error message for invalid combinations (e.g., --empr without
    --expressionfile, or --min > --max).
    """
    p = argparse.ArgumentParser(
        description="Ranked ANDES gene-set enrichment (E_unit + null cache)"
    )
    p.add_argument("--emb", required=True, help="Embedding CSV (genes × dims)")
    p.add_argument("--genelist", required=True, help="Gene list (one per line)")
    p.add_argument("--geneset", required=True, help="Gene-set database (GMT)")

    ranked_grp = p.add_mutually_exclusive_group(required=True)
    ranked_grp.add_argument(
        "--rankedlist", default="", help="Pre-ranked gene list (TSV, index=gene)"
    )
    ranked_grp.add_argument(
        "--expressionfile",
        default="",
        help="Expression matrix (derives ranked list + enables --empr)",
    )

    p.add_argument("--out", required=True, help="Output CSV for z-scores")

    cache_grp = p.add_mutually_exclusive_group()
    cache_grp.add_argument("--cache", default="", help="Explicit cache path")
    cache_grp.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory for auto-named cache (default: cache/)",
    )
    cache_grp.add_argument(
        "--no-cache", action="store_true", help="Do not load or save null cache"
    )

    p.add_argument(
        "--rebuild-cache", action="store_true", help="Ignore existing cache and rebuild"
    )
    p.add_argument(
        "--empr",
        action="store_true",
        help="Compute empirical p-values (requires --expressionfile)",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Permutations for empirical p-value (default: 100)",
    )
    p.add_argument(
        "--ite",
        type=int,
        default=1000,
        help="Monte Carlo iterations for null cache (default: 1000)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Workers for null-cache build (default: 1)",
    )
    p.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per worker (default: 1)",
    )
    p.add_argument(
        "--es-batch-mb",
        type=int,
        default=128,
        help="ES workspace cap per worker in MB (default: 128)",
    )
    p.add_argument(
        "--score-mode",
        choices=["batched", "bestmatch"],
        default="batched",
        help="True-score mode; bestmatch uses exact ranked gene-to-term matrices",
    )
    p.add_argument("--min", dest="min_size", type=int, default=10)
    p.add_argument("--max", dest="max_size", type=int, default=300)
    p.add_argument(
        "--seed", type=int, default=12345, help="Random seed; -1 for OS entropy"
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--warmup-numba",
        action="store_true",
        help="Run Numba JIT warmup before timed computation; useful for benchmarking",
    )

    args = p.parse_args()

    if args.min_size < 1:
        p.error("--min must be >= 1")
    if args.max_size < args.min_size:
        p.error("--max must be >= --min")
    if args.ite < 1:
        p.error("--ite must be >= 1")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.n_permutations < 1:
        p.error("--n-permutations must be >= 1")
    if args.empr and not args.expressionfile:
        p.error("--empr requires --expressionfile")

    return args


# Main


def main():
    """End-to-end GSEA pipeline.

    Steps:
      1. Load and L2-normalize the embedding matrix.
      2. Build gene-to-index mapping and load GMT gene sets.
      3. Derive or load the ranked gene list.
      4. Load or build the ES null cache (NullCacheESBetter).
      5. Score all gene-set terms via batched GEMM (score_terms_batched).
      6. Write z-scores to CSV.
      7. Optionally compute empirical p-values via label-shuffle permutations.
    """
    args = parse_args()
    args.seed = NullCacheESBetter.resolve_seed(args.seed)
    print(f"Seed: {args.seed}")

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

    # Numba warmup
    if args.warmup_numba:
        print("\nWarming up Numba JIT...")
        func.warmup_numba()
        warmup_numba_es()

    # Gene-index mapping
    # Use a plain dict — defaultdict(lambda: -1) silently maps missing genes
    # to the last embedding row, which corrupts results without any error.
    node_set = set(node_list)
    g_node2index = {g: i for i, g in enumerate(node_list)}

    # Load gene sets
    print("\nLoading gene sets...")
    geneset = ld.load_gmt(args.geneset)
    geneset_indices = ld.term2indexes(
        geneset, g_node2index, upper=args.max_size, lower=args.min_size
    )
    geneset_indices_np = func.preconvert_indices_to_arrays(geneset_indices)
    geneset_terms = sorted(geneset_indices_np.keys())

    all_bg_genes = set().union(*geneset.values()) & node_set
    pop = np.array(sorted(g_node2index[g] for g in all_bg_genes), dtype=np.int32)

    print(f"  {len(geneset_terms)} terms  |  {len(pop)} background genes")

    if not geneset_terms:
        print("Error: no gene sets passed size filters")
        sys.exit(1)

    # Load ranked list
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

    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    ranked_emb_T = np.ascontiguousarray(ranked_emb.T, dtype=np.float32)
    es_batch_bytes = args.es_batch_mb * 1024 * 1024
    print(f"  ranked_emb: {ranked_emb.shape}  ({ranked_emb.nbytes / 1e3:.1f} KB)")

    # Null cache
    sizes = {len(geneset_indices_np[t]) for t in geneset_terms}

    if args.no_cache:
        cache_path = None
    elif args.cache:
        cache_path = args.cache
    else:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = NullCacheESBetter.suggest_path(
            args.cache_dir, E_unit, pop, ranked_emb
        )
        print(f"Auto cache: {cache_path}")

    cache = load_or_build_cache(
        cache_path=cache_path,
        E_unit=E_unit,
        pop=pop,
        sizes=sizes,
        ranked_emb=ranked_emb,
        ite=args.ite,
        seed=args.seed,
        n_workers=args.workers,
        blas_threads=args.blas_threads,
        es_batch_bytes=args.es_batch_mb * 1024 * 1024,
        verbose=args.verbose,
        rebuild=args.rebuild_cache,
    )

    # Score all terms (batched by size)
    print(f"\nScoring {len(geneset_terms)} terms...")
    t0 = time.perf_counter()

    if args.score_mode == "bestmatch":
        true_scores, z_scores, score_stats = score_terms_bestmatch(
            E_unit,
            geneset_indices_np,
            geneset_terms,
            ranked_emb,
            cache,
            max_workspace_mb=args.es_batch_mb,
        )
        print(f"Best-match score workspace: ~{score_stats['workspace_mb']:.1f} MB")
    else:
        true_scores, z_scores = score_terms_batched(
            E_unit,
            geneset_indices_np,
            geneset_terms,
            ranked_emb_T,
            cache,
            batch_bytes=es_batch_bytes,
        )

    rows = []
    for term in geneset_terms:
        m = len(geneset_indices_np[term])
        mu, sigma = cache.cache[m]
        rows.append(
            {
                "term": term,
                "size": m,
                "true_score": true_scores[term],
                "null_mu": float(mu),
                "null_sigma": float(sigma),
                "z_score": z_scores[term],
            }
        )

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s  ({len(geneset_terms) / elapsed:.0f} terms/s)")

    # Save z-scores
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).set_index("term")
    out_df.to_csv(out_path)
    print(f"\nSaved z-scores to {out_path}")

    # Empirical p-values (optional)
    if args.empr:
        print(f"\nComputing empirical p-values ({args.n_permutations} permutations)...")

        if data is None or condition is None:
            data = pd.read_csv(args.expressionfile, skiprows=1, sep="\t")
            condition = open(args.expressionfile).readline().strip().split("\t")

        observed = np.array([r["true_score"] for r in rows], dtype=np.float32)
        background = np.zeros(
            (args.n_permutations, len(geneset_terms)), dtype=np.float32
        )

        for i in tqdm(range(args.n_permutations), desc="Permutations"):
            shuffled_genes = expr_func.expression_data_to_ranked_list_label_shuffled(
                data, condition, seed=args.seed + i
            )
            shuf_idx = np.array(
                [g_node2index[g] for g in shuffled_genes if g in node_set],
                dtype=np.int32,
            )
            shuf_emb = compute_ranked_emb(E_unit, shuf_idx)
            shuf_emb_T = np.ascontiguousarray(shuf_emb.T, dtype=np.float32)

            if args.score_mode == "bestmatch":
                shuf_true, _, _ = score_terms_bestmatch(
                    E_unit,
                    geneset_indices_np,
                    geneset_terms,
                    shuf_emb,
                    cache,
                    max_workspace_mb=args.es_batch_mb,
                )
            else:
                shuf_true, _ = score_terms_batched(
                    E_unit,
                    geneset_indices_np,
                    geneset_terms,
                    shuf_emb_T,
                    cache,
                    batch_bytes=es_batch_bytes,
                )
            for j, term in enumerate(geneset_terms):
                background[i, j] = shuf_true[term]

        empirical_p = (np.abs(background) >= np.abs(observed)).mean(axis=0)

        p_df = pd.DataFrame(
            {
                "true_score": observed,
                "z_score": out_df["z_score"].values,
                "empirical_pval": empirical_p,
            },
            index=geneset_terms,
        )
        p_out = out_path.with_name(out_path.stem + "_empirical_pval.csv")
        p_df.to_csv(p_out)
        print(f"Saved empirical p-values to {p_out}")


if __name__ == "__main__":
    main()
