"""
Ranked gene-set enrichment application and command-line entrypoint.

Scores gene sets against a ranked gene list using an embedding-based
enrichment score (ES).  For each gene set, the ES is the maximum signed
cumulative sum of (col_max - mean), where col_max[i] is the maximum cosine
similarity between the i-th ranked gene and any gene in the set.

Null distribution
-----------------
For each gene-set size m, (mu, sigma) are estimated from Monte Carlo
permutations of the background gene pool (``RankedNullBuilder``).
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
  andes enrich \
      --emb embedding.csv --genelist genes.txt \
      --geneset db.gmt --rankedlist ranked.tsv \
      --out results/scores.csv

  # Use an explicit cache directory
  andes enrich ... --cache-dir my_cache/

  # Point to a specific typed cache artifact
  andes enrich ... --cache path/to/cache.null

  # Skip cache entirely (no load, no save)
  andes enrich ... --no-cache

  # Optional exact ranked best-match matrix query prototype
  andes enrich ... --score-mode bestmatch

  # Reuse a persistent index; raw embedding/GMT inputs are not required
  andes enrich \
      --index path/to/index --score-mode indexed \
      --rankedlist ranked.tsv --out results/scores.csv

  # The ES null cache is prefix-coupled by default:
  # one max-size random prefix updates every requested gene-set size.

  # Empirical p-values: batched OLS + indexed ranked traversal
  andes enrich \
      --index path/to/index --score-mode indexed \
      --expressionfile expr.tsv --empr --out results/scores.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

from . import data as ld
from . import bma as func
from . import index as index_api
from . import application
from .nulls import RankedNullModel
from .provenance import RunProvenance, write_result_sidecar
from .runtime import (
    format_blas_runtime,
    query_blas_context,
)
from .ranked import (
    RankedNullBuilder,
    compute_ranked_emb,
    score_terms_indexed,
    warmup_numba_es,
)

try:
    from . import expression as expr_func
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


def ranked_list_from_expression(
    expression_f,
    g_node2index,
    node_set,
    *,
    return_context=False,
):
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
    data = pd.read_csv(expression_f, skiprows=1, sep="\t", index_col=0)
    with open(expression_f) as fh:
        condition = fh.readline().strip().split("\t")

    sample_columns = tuple(data.columns[: len(condition)])
    values, genes, encoded = expr_func.prepare_expression_data(
        data,
        condition,
        sample_columns=sample_columns,
    )
    row_to_embedding = np.fromiter(
        (g_node2index.get(str(gene), -1) for gene in genes),
        dtype=np.int32,
        count=len(genes),
    )
    known_rows = row_to_embedding >= 0
    if not np.any(known_rows):
        raise ValueError("expression data has no genes present in the embedding")

    # Filtering before sorting is order-equivalent to sorting all rows and then
    # removing unknown genes, including stable tie behavior.  It avoids doing
    # OLS work for rows that can never participate in ranked ANDES.
    values_known = np.ascontiguousarray(values[known_rows], dtype=np.float64)
    embedding_rows = np.ascontiguousarray(row_to_embedding[known_rows], dtype=np.int32)
    statistics = expr_func.binary_ols_t_statistics(values_known, encoded)
    order = expr_func.stable_rank_orders(statistics)
    idx = embedding_rows[order]

    if return_context:
        context = {
            "values": values_known,
            "condition": encoded,
            "row_to_embedding": embedding_rows,
        }
        return idx, data, condition, context
    return idx, data, condition


def validate_index_compatibility(
    index,
    *,
    E_unit=None,
    gene_list=None,
    terms=None,
    term_indices=None,
    background=None,
):
    """Validate an index against optional in-memory ranked-GSEA inputs.

    ``load_andes_index`` validates the artifact itself.  This function guards
    the more subtle failure mode where a valid index is paired with a different
    embedding row order or GMT database on the command line.
    """
    if gene_list is not None and list(gene_list) != list(index.gene_list):
        raise ValueError("index gene order does not match the supplied gene list")

    if E_unit is not None:
        E_unit = np.asarray(E_unit, dtype=np.float32)
        if E_unit.shape != index.E_unit.shape:
            raise ValueError(
                "index embedding shape does not match the supplied embedding"
            )
        expected_hash = index.metadata.get("embedding_hash")
        if expected_hash and index_api._hash_array(E_unit) != expected_hash:
            raise ValueError(
                "index embedding does not match the supplied normalized embedding"
            )

    if terms is not None:
        if list(terms) != list(index.terms):
            raise ValueError("index terms do not match the supplied gene-set database")
        if term_indices is None:
            raise ValueError("term_indices are required when validating terms")
        for term in terms:
            supplied = np.unique(np.asarray(term_indices[term], dtype=np.int32))
            indexed = np.asarray(index.term_indices[term], dtype=np.int32)
            if not np.array_equal(supplied, indexed):
                raise ValueError(f"index membership differs for term {term!r}")

    if background is not None:
        supplied_background = np.unique(np.asarray(background, dtype=np.int32))
        if not np.array_equal(supplied_background, index.background):
            raise ValueError(
                "index background does not match the supplied gene-set database"
            )


def build_empirical_bestmatch(
    E_unit,
    geneset_indices_np,
    geneset_terms,
    *,
    max_workspace_mb,
    show_progress=False,
):
    """Build one reusable in-memory B matrix for empirical permutations."""
    blocks = func.precompute_term_embedding_blocks(E_unit, geneset_indices_np)
    bestmatch, workspace_mb = func.gene_to_term_best_match_matrix(
        E_unit,
        geneset_terms,
        blocks,
        max_workspace_mb=max_workspace_mb,
        show_progress=show_progress,
    )
    return bestmatch, workspace_mb


def empirical_exceedance_counts(
    bestmatch,
    term_sizes,
    observed_scores,
    expression_values,
    condition,
    row_to_embedding,
    *,
    n_permutations,
    seed,
    permutation_batch_size=16,
    max_workspace_mb=128,
    progress_callback=None,
):
    """Count ``abs(permuted ES) >= abs(observed ES)`` incrementally.

    No permutation-by-term score matrix is retained.  OLS rankings are produced
    in phenotype batches, while each ranked order traverses the cached
    gene-to-term best-match matrix.
    """
    observed_scores = np.asarray(observed_scores, dtype=np.float32)
    term_sizes = np.asarray(term_sizes, dtype=np.int32)
    row_to_embedding = np.asarray(row_to_embedding, dtype=np.int32)
    expression_values = np.asarray(expression_values, dtype=np.float64)
    if observed_scores.ndim != 1 or observed_scores.size != term_sizes.size:
        raise ValueError("observed_scores and term_sizes must be same-length vectors")
    if (
        expression_values.ndim != 2
        or row_to_embedding.ndim != 1
        or expression_values.shape[0] != row_to_embedding.size
    ):
        raise ValueError("row_to_embedding must contain one index per expression row")

    counts = np.zeros(observed_scores.size, dtype=np.int64)
    observed_abs = np.abs(observed_scores)
    batches = expr_func.iter_label_shuffled_rank_orders(
        expression_values,
        condition,
        n_permutations,
        seed=seed,
        batch_size=permutation_batch_size,
    )
    for _, orders in batches:
        for column in range(orders.shape[1]):
            shuffled_idx = row_to_embedding[orders[:, column]]
            shuffled_scores, _, _ = score_terms_indexed(
                bestmatch,
                shuffled_idx,
                term_sizes,
                cache=None,
                max_workspace_mb=max_workspace_mb,
            )
            counts += np.abs(shuffled_scores) >= observed_abs
        if progress_callback is not None:
            progress_callback(int(orders.shape[1]))
    return counts


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
        Typed null-artifact directory. None disables persistence.
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
    RankedNullBuilder
        Populated null cache.
    """
    expected_metadata = RankedNullBuilder.build_metadata(
        E_unit, pop, ranked_emb, ite, seed
    )

    if cache_path and os.path.exists(cache_path) and not rebuild:
        cache = RankedNullBuilder.load_artifact(cache_path)
        metadata_ok, reason = cache.metadata_matches(expected_metadata)
        missing = cache.missing_sizes(sizes) if metadata_ok else list(sizes)
        if metadata_ok and not missing:
            print(f"Null cache loaded from '{cache_path}'  ({len(cache)} entries)")
            return cache
        if not metadata_ok:
            print(f"Cache metadata invalid ({reason}) — rebuilding.")
        else:
            print(f"Cache missing {len(missing)} sizes — rebuilding.")

    cache = RankedNullBuilder()
    print(
        "\nBuilding ES null cache  "
        f"({len(sizes)} sizes, ite={ite}, workers={n_workers}, "
        f"worker_blas={blas_threads})..."
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
        # With no child process, the main process is the null worker. Scope the
        # same worker-BLAS policy here that the parallel initializer applies.
        with threadpool_limits(limits=blas_threads, user_api="blas"):
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
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.save_artifact(cache_path, overwrite=cache_path.exists())

    return cache


# CLI


def parse_args(argv=None):
    """Parse and validate CLI arguments.

    Exits with an error message for invalid combinations (e.g., --empr without
    --expressionfile, or --min > --max).
    """
    p = argparse.ArgumentParser(
        description="Ranked ANDES gene-set enrichment (E_unit + null cache)"
    )
    p.add_argument(
        "--emb",
        default="",
        help="Embedding CSV (required unless --score-mode indexed)",
    )
    p.add_argument(
        "--genelist",
        default="",
        help="Gene list (required unless --score-mode indexed)",
    )
    p.add_argument(
        "--geneset",
        default="",
        help="Gene-set database (required unless --score-mode indexed)",
    )
    p.add_argument(
        "--index",
        default="",
        help="Persistent ANDES index directory for indexed scoring",
    )

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
        "--permutation-batch-size",
        type=int,
        default=16,
        help="Phenotype permutations per vectorized OLS GEMM (default: 16)",
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
        "--query-blas-threads",
        dest="query_blas_threads",
        type=int,
        default=0,
        help=(
            "BLAS threads for main scoring and expression OLS; "
            "0 keeps the runtime default"
        ),
    )
    p.add_argument(
        "--worker-blas-threads",
        "--blas-threads",
        dest="worker_blas_threads",
        type=int,
        default=1,
        help=(
            "BLAS threads per null-cache worker or serial builder (default: 1); "
            "--blas-threads is retained as a legacy alias"
        ),
    )
    p.add_argument(
        "--es-batch-mb",
        type=int,
        default=128,
        help="ES workspace cap per worker in MB (default: 128)",
    )
    p.add_argument(
        "--score-mode",
        choices=["batched", "bestmatch", "indexed"],
        default="batched",
        help=(
            "True-score mode; indexed slices a persistent gene-to-term "
            "best-match matrix"
        ),
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

    args = p.parse_args(argv)

    if args.min_size < 1:
        p.error("--min must be >= 1")
    if args.max_size < args.min_size:
        p.error("--max must be >= --min")
    if args.ite < 1:
        p.error("--ite must be >= 1")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.query_blas_threads < 0:
        p.error("--query-blas-threads must be non-negative")
    if args.worker_blas_threads < 1:
        p.error("--worker-blas-threads must be >= 1")
    if args.n_permutations < 1:
        p.error("--n-permutations must be >= 1")
    if args.permutation_batch_size < 1:
        p.error("--permutation-batch-size must be >= 1")
    if args.empr and not args.expressionfile:
        p.error("--empr requires --expressionfile")
    if bool(args.emb) != bool(args.genelist):
        p.error("--emb and --genelist must be supplied together")
    if args.score_mode == "indexed":
        if not args.index:
            p.error("--score-mode indexed requires --index")
    else:
        missing = [
            flag
            for flag, value in (
                ("--emb", args.emb),
                ("--genelist", args.genelist),
                ("--geneset", args.geneset),
            )
            if not value
        ]
        if missing:
            p.error(f"{args.score_mode} mode requires " + ", ".join(missing))

    return args


# Main


def main(argv=None):
    """End-to-end GSEA pipeline.

    Steps:
      1. Load and L2-normalize the embedding matrix.
      2. Build gene-to-index mapping and load GMT gene sets.
      3. Derive or load the ranked gene list.
      4. Load or build the typed ranked null calibration.
      5. Score all gene-set terms through the typed ranked application API.
      6. Write z-scores to CSV.
      7. Optionally compute empirical p-values via label-shuffle permutations.
    """
    args = parse_args(argv)
    args.seed = RankedNullBuilder.resolve_seed(args.seed)
    print(f"Seed: {args.seed}")

    # ── Load normalized embedding and term database ────────────────────────
    index = None
    if args.score_mode == "indexed":
        print(f"Loading ANDES index from {args.index}...")
        index = index_api.load_andes_index(args.index, mmap=True)
        E_unit = index.E_unit
        node_list = list(index.gene_list)
        geneset_terms = list(index.terms)
        geneset_indices_np = index.term_indices
        pop = np.asarray(index.background, dtype=np.int32)

        # Optional raw inputs are compatibility assertions in indexed mode.
        if args.emb:
            supplied_E, supplied_genes = index_api.load_embedding(
                args.emb, args.genelist
            )
            validate_index_compatibility(
                index, E_unit=supplied_E, gene_list=supplied_genes
            )
        if args.geneset:
            (
                _,
                supplied_terms,
                supplied_indices,
                supplied_background,
            ) = index_api.load_target_gmt(
                args.geneset,
                node_list,
                min_size=args.min_size,
                max_size=args.max_size,
            )
            validate_index_compatibility(
                index,
                terms=supplied_terms,
                term_indices=supplied_indices,
                background=supplied_background,
            )
        print(
            f"  {len(node_list)} genes  |  dim={E_unit.shape[1]}  |  "
            f"{len(geneset_terms)} indexed terms"
        )
    else:
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

        node_set = set(node_list)
        g_node2index = {g: i for i, g in enumerate(node_list)}

        print("\nLoading gene sets...")
        geneset = ld.load_gmt(args.geneset)
        geneset_indices = ld.term2indexes(
            geneset,
            g_node2index,
            upper=args.max_size,
            lower=args.min_size,
        )
        geneset_indices_np = func.preconvert_indices_to_arrays(geneset_indices)
        geneset_terms = sorted(geneset_indices_np.keys())
        all_bg_genes = set().union(*geneset.values()) & node_set
        pop = np.array(
            sorted(g_node2index[g] for g in all_bg_genes),
            dtype=np.int32,
        )

        if args.index:
            index = index_api.load_andes_index(args.index, mmap=True)
            validate_index_compatibility(
                index,
                E_unit=E_unit,
                gene_list=node_list,
                terms=geneset_terms,
                term_indices=geneset_indices_np,
                background=pop,
            )

    print(
        f"  E_unit: {E_unit.nbytes / 1e6:.1f} MB  "
        f"(vs {len(node_list) ** 2 * 8 / 1e9:.1f} GB for full similarity matrix)"
    )

    # Numba warmup
    if args.warmup_numba:
        print("\nWarming up Numba JIT...")
        warmup_numba_es()

    # Gene-index mapping
    # Use a plain dict — defaultdict(lambda: -1) silently maps missing genes
    # to the last embedding row, which corrupts results without any error.
    node_set = set(node_list)
    g_node2index = {g: i for i, g in enumerate(node_list)}

    print(f"  {len(geneset_terms)} terms  |  {len(pop)} background genes")

    if not geneset_terms:
        print("Error: no gene sets passed size filters")
        sys.exit(1)

    # Load ranked list
    print("\nLoading ranked list...")
    data, condition, expression_context = None, None, None

    if args.rankedlist:
        ranked_idx = load_ranked_list(args.rankedlist, g_node2index, node_set)
        print(f"  Loaded pre-ranked list: {len(ranked_idx)} genes")
    else:
        with query_blas_context(args.query_blas_threads):
            (
                ranked_idx,
                data,
                condition,
                expression_context,
            ) = ranked_list_from_expression(
                args.expressionfile,
                g_node2index,
                node_set,
                return_context=True,
            )
        print(f"  Derived ranked list from expression file: {len(ranked_idx)} genes")

    if len(ranked_idx) == 0:
        print("Error: ranked list is empty after filtering to embedding genes")
        sys.exit(1)

    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    print(f"  ranked_emb: {ranked_emb.shape}  ({ranked_emb.nbytes / 1e3:.1f} KB)")

    # Null cache
    sizes = {len(geneset_indices_np[t]) for t in geneset_terms}

    if args.no_cache:
        cache_path = None
    elif args.cache:
        cache_path = args.cache
    else:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = RankedNullBuilder.suggest_path(
            args.cache_dir,
            E_unit,
            pop,
            ranked_emb,
            ite=args.ite,
            seed=args.seed,
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
        blas_threads=args.worker_blas_threads,
        es_batch_bytes=args.es_batch_mb * 1024 * 1024,
        verbose=args.verbose,
        rebuild=args.rebuild_cache,
    )
    embedding_model = ld.EmbeddingSpace.from_arrays(E_unit, node_list, normalize=False)
    database_model = ld.GeneSetDatabase.from_index_mapping(
        geneset_indices_np,
        n_genes=len(node_list),
        embedding_fingerprint=embedding_model.fingerprint,
        terms=geneset_terms,
        background=pop,
        min_size=1,
    )
    typed_null = RankedNullModel.from_builder(cache)

    # Score all terms (batched by size)
    print(f"\nScoring {len(geneset_terms)} terms...")
    t0 = time.perf_counter()

    with query_blas_context(args.query_blas_threads):
        requested = (
            "runtime default"
            if args.query_blas_threads == 0
            else str(args.query_blas_threads)
        )
        print(f"Query BLAS: {requested} ({format_blas_runtime()})")
        runtime_metadata = {
            "query_blas_threads_requested": int(args.query_blas_threads),
            "worker_blas_threads": int(args.worker_blas_threads),
            "workers": int(args.workers),
            "blas_pools": [
                {
                    "internal_api": pool.get("internal_api", "unknown"),
                    "prefix": pool.get("prefix", "unknown"),
                    "num_threads": int(pool.get("num_threads", 0)),
                }
                for pool in threadpool_info()
                if pool.get("user_api") == "blas"
            ],
        }
        ranked_result = application.run_ranked(
            application.RankedRequest(
                embedding=embedding_model,
                database=database_model,
                ranked_indices=ranked_idx,
                engine=args.score_mode,
                bestmatch=(
                    index.ranked_bestmatch_artifact()
                    if args.score_mode == "indexed"
                    else None
                ),
                workspace_mb=args.es_batch_mb,
                null_model=typed_null,
                runtime=runtime_metadata,
            )
        )

    true_scores = {
        term: float(ranked_result.true_scores[i])
        for i, term in enumerate(geneset_terms)
    }
    z_scores = {
        term: float(ranked_result.scores.scores[i])
        for i, term in enumerate(geneset_terms)
    }
    if ranked_result.scores.stats.workspace_bytes:
        print(
            f"{args.score_mode.capitalize()} score workspace: "
            f"~{ranked_result.scores.stats.workspace_bytes / 1e6:.1f} MB"
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
    application.write_dataframe_atomic(out_df, out_path)
    provenance = RunProvenance(
        method="andes_ranked",
        score_engine=args.score_mode,
        score_kind="z_score",
        numeric_dtype="float32",
        accumulator_dtype="float64",
        embedding_fingerprint=embedding_model.fingerprint,
        left_database_fingerprint=database_model.fingerprint,
        null_spec=typed_null.spec.to_dict(),
        runtime=runtime_metadata,
        extra={
            "ranked_genes": int(len(ranked_idx)),
            "terms": len(geneset_terms),
            "expression_derived": bool(args.expressionfile),
        },
    )
    sidecar = write_result_sidecar(out_path, provenance)
    print(f"\nSaved z-scores to {out_path}")
    print(f"Provenance: {sidecar}")

    # Empirical p-values (optional)
    if args.empr:
        print(f"\nComputing empirical p-values ({args.n_permutations} permutations)...")

        if expression_context is None:
            (
                _,
                data,
                condition,
                expression_context,
            ) = ranked_list_from_expression(
                args.expressionfile,
                g_node2index,
                node_set,
                return_context=True,
            )

        observed = np.array([r["true_score"] for r in rows], dtype=np.float32)
        term_sizes = np.asarray(
            [len(geneset_indices_np[term]) for term in geneset_terms],
            dtype=np.int32,
        )

        if index is not None:
            empirical_bestmatch = index.bestmatch
            owns_empirical_bestmatch = False
            print("  Reusing persistent indexed best-match matrix")
        else:
            print("  Building reusable best-match matrix for permutations...")
            with query_blas_context(args.query_blas_threads):
                empirical_bestmatch, workspace_mb = build_empirical_bestmatch(
                    E_unit,
                    geneset_indices_np,
                    geneset_terms,
                    max_workspace_mb=args.es_batch_mb,
                    show_progress=args.verbose,
                )
            owns_empirical_bestmatch = True
            print(f"  Best-match build workspace: ~{workspace_mb:.1f} MB")

        with tqdm(
            total=args.n_permutations,
            desc="Permutations",
        ) as progress:
            with query_blas_context(args.query_blas_threads):
                exceedance_counts = empirical_exceedance_counts(
                    empirical_bestmatch,
                    term_sizes,
                    observed,
                    expression_context["values"],
                    expression_context["condition"],
                    expression_context["row_to_embedding"],
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    permutation_batch_size=args.permutation_batch_size,
                    max_workspace_mb=args.es_batch_mb,
                    progress_callback=progress.update,
                )

        empirical_p = exceedance_counts.astype(np.float64) / float(args.n_permutations)
        if owns_empirical_bestmatch:
            del empirical_bestmatch

        p_df = pd.DataFrame(
            {
                "true_score": observed,
                "z_score": out_df["z_score"].values,
                "empirical_pval": empirical_p,
            },
            index=geneset_terms,
        )
        p_out = out_path.with_name(out_path.stem + "_empirical_pval.csv")
        application.write_dataframe_atomic(p_df, p_out)
        empirical_provenance = RunProvenance(
            method="andes_ranked_empirical",
            score_engine="indexed",
            score_kind="empirical_p_value",
            numeric_dtype="float64",
            accumulator_dtype="int64",
            embedding_fingerprint=embedding_model.fingerprint,
            left_database_fingerprint=database_model.fingerprint,
            null_spec=typed_null.spec.to_dict(),
            runtime=runtime_metadata,
            extra={
                "ranked_genes": int(len(ranked_idx)),
                "terms": len(geneset_terms),
                "permutations": int(args.n_permutations),
                "permutation_batch_size": int(args.permutation_batch_size),
                "seed": int(args.seed),
            },
        )
        empirical_sidecar = write_result_sidecar(p_out, empirical_provenance)
        print(f"Saved empirical p-values to {p_out}")
        print(f"Provenance: {empirical_sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
