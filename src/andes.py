import os
import sys

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
    parser.add_argument("--cache", default="", help="cache file; auto-named from inputs if empty")
    parser.add_argument(
        "--no-precompute",
        action="store_true",
        help="require an existing valid cache instead of rebuilding",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="ignore any existing cache and rebuild it",
    )
    parser.add_argument("--min", dest="min_size", type=int, default=10)
    parser.add_argument("--max", dest="max_size", type=int, default=300)
    parser.add_argument("--ite", type=int, default=1000, help="null iterations")
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel workers for null cache"
    )
    parser.add_argument(
        "--query-workers",
        type=int,
        default=0,
        help="thread workers for query scoring; 0 reuses --workers",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="null-cache chunk size; 0 enables cost-based auto tuning",
    )
    parser.add_argument(
        "--no-term-block-cache",
        action="store_true",
        help="avoid precomputing per-term embedding blocks before scoring",
    )
    parser.add_argument(
        "--numba-threshold",
        type=int,
        default=0,
        help="use Numba BMA scorer when set_size_product is below this value",
    )
    parser.add_argument(
        "--query-mode",
        choices=["batched", "pairwise"],
        default="batched",
        help="BMA query scorer; batched scores whole rows with larger GEMMs",
    )
    parser.add_argument(
        "--query-memory-mb",
        type=float,
        default=1024.0,
        help="approximate memory cap for batched query workspaces",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="random seed for null cache; use -1 for OS entropy",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="verbose output"
    )

    args = parser.parse_args()
    args.seed = func.NullCacheBMA.resolve_seed(args.seed)
    print(f"Seed: {args.seed}")

    # Load embeddings
    print("Loading embeddings...")
    node_vectors = np.loadtxt(args.emb, delimiter=",", dtype=np.float32)
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

    if not geneset1_terms:
        print("Error: no terms from geneset1 passed size filters")
        sys.exit(1)
    if not geneset2_terms:
        print("Error: no terms from geneset2 passed size filters")
        sys.exit(1)

    # Background
    bg1 = func.get_background_indices(geneset1, node_set, g_node2index)
    bg2 = func.get_background_indices(geneset2, node_set, g_node2index)
    all_background = sorted(set(bg1) | set(bg2))
    bg1_np = np.asarray(bg1, dtype=np.int32)
    bg2_np = np.asarray(bg2, dtype=np.int32)

    print(f"Database 1: {len(geneset1_terms)} terms")
    print(f"Database 2: {len(geneset2_terms)} terms")
    print(f"Background: {len(all_background)} genes")

    if not args.cache:
        args.cache = func.NullCacheBMA.suggest_path("cache", E_unit, bg1_np, bg2_np)
        os.makedirs("cache", exist_ok=True)
        print(f"Auto cache: {args.cache}")

    # Null cache
    null_cache = func.NullCacheBMA()
    expected_metadata = func.NullCacheBMA.build_metadata(
        E_unit, bg1_np, bg2_np, args.ite, args.seed
    )

    print("\nComputing unique size pairs...")
    sizes1 = {len(geneset1_indices_np[t]) for t in geneset1_terms}
    sizes2 = {len(geneset2_indices_np[t]) for t in geneset2_terms}
    size_pairs = {(m, k) for m in sizes1 for k in sizes2}

    print(f"Unique pairs: {len(size_pairs)}")
    print(f"Range: {min(sizes1)}-{max(sizes1)} x {min(sizes2)}-{max(sizes2)}")

    cache_ready = False
    if args.cache and os.path.exists(args.cache) and not args.rebuild_cache:
        null_cache.load(args.cache)
        metadata_ok, reason = null_cache.metadata_matches(expected_metadata)
        missing = sorted(pair for pair in size_pairs if pair not in null_cache.cache)
        cache_ready = metadata_ok and not missing
        if cache_ready:
            print(f"Loaded valid null cache from {args.cache}")
        elif args.no_precompute:
            detail = reason if not metadata_ok else f"{len(missing)} missing size pairs"
            print(f"Error: cache invalid ({detail}); rebuild the cache")
            sys.exit(1)
        else:
            detail = reason if not metadata_ok else f"{len(missing)} missing size pairs"
            print(f"Cache invalid/incomplete ({detail}); rebuilding")
    elif args.no_precompute:
        print(f"Error: cache file not found: {args.cache}")
        sys.exit(1)

    if not cache_ready:
        null_cache.precompute_parallel(
            E_unit,
            bg1_np,
            size_pairs,
            ite=args.ite,
            seed=args.seed,
            verbose=args.verbose,
            n_workers=args.workers,
            chunk_size=None if args.chunk_size <= 0 else args.chunk_size,
            population_idx2=bg2_np,
        )
        null_cache.save(args.cache)

    # Main computation: row-wise streaming
    print(f"\nComputing {len(geneset1_terms)} x {len(geneset2_terms)} pairs...")
    query_workers = args.workers if args.query_workers <= 0 else args.query_workers
    print(f"Row-wise scoring with {query_workers} query worker(s)")

    use_blocks = not args.no_term_block_cache
    if use_blocks:
        print("Precomputing per-term embedding blocks...")
        geneset1_blocks = func.precompute_term_embedding_blocks(E_unit, geneset1_indices_np)
        geneset2_blocks = func.precompute_term_embedding_blocks(E_unit, geneset2_indices_np)
        block_mb = (
            sum(block.nbytes for block in geneset1_blocks.values())
            + sum(block.nbytes for block in geneset2_blocks.values())
        ) / 1e6
        print(f"Term block cache: {block_mb:.1f} MB")
    else:
        geneset1_blocks = {}
        geneset2_blocks = {}

    symmetric = (
        np.array_equal(bg1_np, bg2_np)
        and func.same_index_arrays_by_term(
            geneset1_terms, geneset1_indices_np, geneset2_terms, geneset2_indices_np
        )
    )
    if symmetric:
        print("Detected symmetric gene-set comparison; scoring upper triangle only")

    if args.query_mode == "batched" and use_blocks:
        zscores, effective_workers, workspace_mb = func.score_bma_zscore_matrix_batched(
            geneset1_terms,
            geneset2_terms,
            null_cache,
            geneset1_blocks,
            geneset2_blocks,
            symmetric=symmetric,
            n_workers=query_workers,
            max_workspace_mb=args.query_memory_mb,
            show_progress=args.verbose,
        )
        print(
            f"Batched query workspace: ~{workspace_mb:.1f} MB/worker; "
            f"using {effective_workers} worker(s)"
        )
    else:
        zscores = func.score_bma_zscore_matrix(
            E_unit,
            geneset1_terms,
            geneset2_terms,
            geneset1_indices_np,
            geneset2_indices_np,
            null_cache,
            blocks1=geneset1_blocks if use_blocks else None,
            blocks2=geneset2_blocks if use_blocks else None,
            symmetric=symmetric,
            n_workers=query_workers,
            numba_threshold=args.numba_threshold,
            show_progress=args.verbose,
        )

    # Save
    print("\nSaving results...")
    zscores_df = pd.DataFrame(zscores, index=geneset1_terms, columns=geneset2_terms)
    zscores_df.to_csv(args.out)

    print(f"Saved to {args.out}")
    print("Done!")
