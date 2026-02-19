import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
import load_data as ld
import func_optimized as func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized ANDES")
    parser.add_argument("--emb", required=True, help="embedding file")
    parser.add_argument("--genelist", required=True, help="gene list file")
    parser.add_argument("--geneset1", required=True, help="first gene set (GMT)")
    parser.add_argument("--geneset2", required=True, help="second gene set (GMT)")
    parser.add_argument("--out", required=True, help="output file")
    parser.add_argument("--cache", default="null_cache_bma.pkl", help="cache file")
    parser.add_argument(
        "--no-precompute", action="store_true", help="load cache instead"
    )
    parser.add_argument("--min", dest="min_size", type=int, default=10)
    parser.add_argument("--max", dest="max_size", type=int, default=300)
    parser.add_argument("--ite", type=int, default=1000, help="null iterations")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="verbose output"
    )

    args = parser.parse_args()

    # Load embeddings
    print("Loading embeddings...")
    node_vectors = np.loadtxt(args.emb, delimiter=",")
    with open(args.genelist, "r") as f:
        node_list = [line.strip() for line in f]

    print(f"Loaded: {len(node_list)} genes, dim={node_vectors.shape[1]}")

    # Normalize
    print("Normalizing embeddings...")
    E_unit = np.ascontiguousarray(
        func.l2_normalize_rows(node_vectors), dtype=np.float32
    )
    del node_vectors

    # Warmup numba
    print("Warming up numba...")
    func.warmup_numba()

    # Mappings
    node_set = set(node_list)
    g_node2index = defaultdict(lambda: -1, {j: i for i, j in enumerate(node_list)})

    # Load gene sets
    print("Loading gene sets...")
    geneset1 = ld.load_gmt(args.geneset1)
    geneset1_indices = ld.term2indexes(
        geneset1, g_node2index, upper=args.max_size, lower=args.min_size
    )

    geneset2 = ld.load_gmt(args.geneset2)
    geneset2_indices = ld.term2indexes(
        geneset2, g_node2index, upper=args.max_size, lower=args.min_size
    )

    # Preconvert to arrays
    print("Converting indices...")
    geneset1_indices_np = func.preconvert_indices_to_arrays(geneset1_indices)
    geneset2_indices_np = func.preconvert_indices_to_arrays(geneset2_indices)

    geneset1_terms = list(geneset1_indices_np.keys())
    geneset2_terms = list(geneset2_indices_np.keys())

    # Background
    bg1 = func.get_background_indices(geneset1, node_set, g_node2index)
    bg2 = func.get_background_indices(geneset2, node_set, g_node2index)
    all_background = sorted(set(bg1) | set(bg2))

    print(f"Database 1: {len(geneset1_terms)} terms")
    print(f"Database 2: {len(geneset2_terms)} terms")
    print(f"Background: {len(all_background)} genes")

    # Null cache
    null_cache = func.NullCacheBMA()

    if not args.no_precompute:
        print("\nComputing unique size pairs...")
        sizes1 = {len(geneset1_indices_np[t]) for t in geneset1_terms}
        sizes2 = {len(geneset2_indices_np[t]) for t in geneset2_terms}
        size_pairs = {(m, k) for m in sizes1 for k in sizes2}

        print(f"Unique pairs: {len(size_pairs)}")
        print(f"Range: {min(sizes1)}-{max(sizes1)} x {min(sizes2)}-{max(sizes2)}")

        null_cache.precompute_parallel(
            E_unit,
            all_background,
            size_pairs,
            ite=args.ite,
            seed=args.seed,
            verbose=args.verbose,
            n_workers=8,
            chunk_size=64,
        )
        null_cache.save(args.cache)
    else:
        null_cache.load(args.cache)

    # Main computation: row-wise streaming
    print(f"\nComputing {len(geneset1_terms)} x {len(geneset2_terms)} pairs...")
    print("Row-wise streaming (no futures overhead)")

    zscores = np.zeros((len(geneset1_terms), len(geneset2_terms)), dtype=np.float32)
    max_m = max(len(geneset1_indices_np[t]) for t in geneset1_terms)
    max_k = max(len(geneset2_indices_np[t]) for t in geneset2_terms)
    ws_max = func.BMAWorkspaceMax(max_m, max_k, E_unit.shape[1])

    for i, t1 in enumerate(
        tqdm(geneset1_terms, desc="Computing", disable=not args.verbose)
    ):
        X_idx = geneset1_indices_np[t1]
        m = len(X_idx)

        for j, t2 in enumerate(geneset2_terms):
            Y_idx = geneset2_indices_np[t2]
            k = len(Y_idx)

            # optional tiny-set dispatch
            if m * k < 400:
                true_score = func.compute_bma_numba(E_unit, X_idx, Y_idx)
            else:
                true_score = func.compute_bma_fast_ws_view(
                    E_unit, X_idx, Y_idx, ws_max.views(m, k)
                )

            zscores[i, j] = null_cache.get_zscore(true_score, m, k)

    # Save
    print("\nSaving results...")
    zscores_df = pd.DataFrame(zscores, index=geneset1_terms, columns=geneset2_terms)
    zscores_df.to_csv(args.out)

    print(f"Saved to {args.out}")
    print("Done!")
