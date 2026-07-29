"""Ranked gene-set enrichment application and command-line entrypoint.

For each gene set, ranked ANDES returns the largest signed cumulative
deviation of its centered best-match affinities. Prefix-coupled Monte Carlo
samples estimate the null distribution for each gene-set size.

Expression runs shuffle condition labels and report the plus-one empirical
p-value ``(1 + exceedances) / (1 + permutations)``.
"""

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_info
from tqdm import tqdm

from . import application, artifacts, bma
from . import data as ld
from . import expression as expr_func
from . import index as index_api
from .nulls import (
    RankedNullModel,
    null_cache_dir,
    resolve_null_seed,
    resolve_ranked_null,
)
from .provenance import write_result_sidecar
from .ranked import (
    compute_ranked_emb,
    count_indexed_ranked_exceedances,
    plan_ranked_null_runtime,
)
from .runtime import (
    format_blas_runtime,
    query_blas_context,
)


def ranked_list_from_expression(
    expression_f,
    g_node2index,
):
    """Derive a ranked gene list from an expression matrix.

    Returns ranked indices plus the expression context needed for phenotype
    permutations.

    Parameters
    ----------
    expression_f : str
        Expression matrix file. First line is condition labels; remaining
        rows are tab-separated gene expression values.
    g_node2index : dict
        Maps gene ID strings to embedding row indices.
    Returns
    -------
    tuple
        Embedding indices in ranked order and the validated expression context.
    """
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
    # removing unknown genes, including stable tie behavior. It avoids doing
    # OLS work for rows that can never participate in ranked ANDES.
    values_known = np.ascontiguousarray(values[known_rows], dtype=np.float64)
    embedding_rows = np.ascontiguousarray(row_to_embedding[known_rows], dtype=np.int32)
    statistics = expr_func.binary_ols_t_statistics(values_known, encoded)
    order = expr_func.stable_rank_orders(statistics)
    idx = embedding_rows[order]

    context = {
        "values": values_known,
        "condition": encoded,
        "row_to_embedding": embedding_rows,
    }
    return idx, context


def validate_index_compatibility(
    index,
    *,
    E_unit=None,
    gene_list=None,
    database=None,
):
    """Check that optional raw inputs describe the loaded index."""
    if not isinstance(index, index_api.AndesIndex):
        raise TypeError("index must be an AndesIndex")
    if gene_list is not None and list(gene_list) != list(index.gene_list):
        raise ValueError("index gene order does not match the supplied gene list")

    if E_unit is not None:
        E_unit = np.asarray(E_unit, dtype=np.float32)
        if E_unit.shape != index.E_unit.shape:
            raise ValueError(
                "index embedding shape does not match the supplied embedding"
            )
        expected_hash = index.metadata.get("embedding_hash")
        if expected_hash and artifacts.hash_array(E_unit) != expected_hash:
            raise ValueError(
                "index embedding does not match the supplied normalized embedding"
            )

    if database is not None:
        if not isinstance(database, ld.GeneSetDatabase):
            raise TypeError("database must be a GeneSetDatabase")
        if database.embedding_fingerprint != index.embedding_fingerprint:
            raise ValueError(
                "index embedding identity does not match the supplied gene-set database"
            )
        if database.fingerprint != index.database_fingerprint:
            raise ValueError(
                "index terms, memberships, or background do not match the "
                "supplied gene-set database"
            )


def monte_carlo_pvalues(exceedance_counts, n_permutations):
    """Apply the plus-one estimate ``(count + 1) / (permutations + 1)``."""
    if isinstance(n_permutations, (bool, np.bool_)) or not isinstance(
        n_permutations,
        (int, np.integer),
    ):
        raise TypeError("n_permutations must be an integer")
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    counts = np.asarray(exceedance_counts)
    if counts.dtype.kind not in "iu":
        raise TypeError("exceedance_counts must contain integers")
    if np.any(counts < 0) or np.any(counts > n_permutations):
        raise ValueError("exceedance_counts must lie between zero and n_permutations")
    return (counts.astype(np.float64) + 1.0) / float(n_permutations + 1)


@dataclass(frozen=True, slots=True)
class EmpiricalCountStats:
    """Peak indexed-traversal memory across phenotype-permutation batches."""

    indexed_workspace_bytes: int
    requested_workspace_bytes: int
    term_chunk_size: int
    phenotype_batch_size: int
    batches: int


@dataclass(frozen=True, slots=True)
class EmpiricalCountResult:
    """Incremental empirical exceedance counts and execution metadata."""

    counts: NDArray[np.int64]
    stats: EmpiricalCountStats


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
    max_workspace_mb: float | None = 128,
    progress_callback=None,
):
    """Count ``abs(permuted ES) >= abs(observed ES)`` incrementally.

    OLS rankings are produced in phenotype batches. Each ranked order traverses
    the cached gene-to-term best-match matrix, and exceedance counts are updated
    without storing a permutation-by-term score matrix.
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
    peak_workspace_bytes = 0
    requested_workspace_bytes = 0
    term_chunk_size = 0
    batches_processed = 0
    batches = expr_func.iter_label_shuffled_rank_orders(
        expression_values,
        condition,
        n_permutations,
        seed=seed,
        batch_size=permutation_batch_size,
    )
    for _, orders in batches:
        counts, stats = count_indexed_ranked_exceedances(
            bestmatch,
            row_to_embedding,
            orders,
            observed_scores,
            counts=counts,
            max_workspace_mb=max_workspace_mb,
        )
        peak_workspace_bytes = max(
            peak_workspace_bytes,
            int(stats["workspace_bytes"]),
        )
        requested_workspace_bytes = max(
            requested_workspace_bytes,
            int(stats["requested_workspace_bytes"]),
        )
        term_chunk_size = max(term_chunk_size, int(stats["term_chunk_size"]))
        batches_processed += 1
        if progress_callback is not None:
            progress_callback(int(orders.shape[1]))
    return EmpiricalCountResult(
        counts=counts,
        stats=EmpiricalCountStats(
            indexed_workspace_bytes=peak_workspace_bytes,
            requested_workspace_bytes=requested_workspace_bytes,
            term_chunk_size=term_chunk_size,
            phenotype_batch_size=int(permutation_batch_size),
            batches=batches_processed,
        ),
    )


def parse_args(argv=None):
    """Parse and validate CLI arguments.

    Exits with an error message for invalid combinations (e.g., --empr without
    --expressionfile, or --min > --max).
    """
    p = argparse.ArgumentParser(
        prog="andes enrich",
        description="Ranked ANDES gene-set enrichment",
    )
    p.add_argument(
        "--emb",
        default="",
        help="embedding CSV or NPY; required when --index is absent",
    )
    p.add_argument(
        "--genelist",
        default="",
        help="embedding gene list; required when --index is absent",
    )
    p.add_argument(
        "--geneset",
        default="",
        help="GMT database; required when --index is absent",
    )
    p.add_argument(
        "--index",
        default="",
        help="persistent ANDES index directory",
    )

    ranked_grp = p.add_mutually_exclusive_group(required=True)
    ranked_grp.add_argument("--rankedlist", default="", help="pre-ranked gene list")
    ranked_grp.add_argument(
        "--expressionfile",
        default="",
        help="expression matrix; derives the ranking and enables --empr",
    )

    p.add_argument("--out", required=True, help="output CSV for z-scores")

    cache_grp = p.add_mutually_exclusive_group()
    cache_grp.add_argument("--cache", default="", help="null artifact directory")
    cache_grp.add_argument(
        "--cache-root",
        default="",
        help=(
            "content-addressed artifact root; defaults to ANDES_CACHE_ROOT or cache/"
        ),
    )
    cache_grp.add_argument(
        "--no-cache",
        action="store_true",
        help="build the null in memory and skip artifact storage",
    )

    p.add_argument(
        "--rebuild-cache", action="store_true", help="replace an existing null artifact"
    )
    p.add_argument(
        "--cache-policy",
        choices=("build", "require"),
        default=None,
        help=(
            "build missing null entries or require a complete artifact; "
            "defaults to require with --index and build otherwise"
        ),
    )
    p.add_argument(
        "--empr",
        action="store_true",
        help="compute empirical p-values; requires --expressionfile",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="empirical permutations (default: 100)",
    )
    p.add_argument(
        "--permutation-batch-size",
        type=int,
        default=16,
        help="phenotype permutations per OLS batch (default: 16)",
    )
    p.add_argument(
        "--ite",
        type=int,
        default=1000,
        help="Monte Carlo null iterations (default: 1000)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=("null-construction workers; 0 selects a memory-aware count (default: 0)"),
    )
    p.add_argument(
        "--query-blas-threads",
        dest="query_blas_threads",
        type=int,
        default=0,
        help=(
            "BLAS threads for scoring and expression OLS; 0 keeps the runtime default"
        ),
    )
    p.add_argument(
        "--worker-blas-threads",
        dest="worker_blas_threads",
        type=int,
        default=1,
        help="BLAS threads per null worker or serial builder (default: 1)",
    )
    p.add_argument(
        "--workspace-mb",
        type=float,
        default=128,
        help=(
            "ranked-scoring workspace target in decimal MB; one term "
            "may exceed it (default: 128)"
        ),
    )
    p.add_argument(
        "--null-memory-mb",
        type=float,
        default=128,
        help=(
            "total ranked-null workspace across workers in decimal MB (default: 128)"
        ),
    )
    p.add_argument(
        "--min",
        dest="min_size",
        type=int,
        default=10,
        help="minimum mapped term size (default: 10)",
    )
    p.add_argument(
        "--max",
        dest="max_size",
        type=int,
        default=300,
        help="maximum mapped term size (default: 300)",
    )
    p.add_argument(
        "--seed", type=int, default=12345, help="random seed; -1 uses OS entropy"
    )
    args = p.parse_args(argv)

    if args.min_size < 1:
        p.error("--min must be >= 1")
    if args.max_size < args.min_size:
        p.error("--max must be >= --min")
    if args.ite < 2:
        p.error("--ite must be >= 2")
    if args.workers < 0:
        p.error("--workers must be non-negative")
    if args.query_blas_threads < 0:
        p.error("--query-blas-threads must be non-negative")
    if args.worker_blas_threads < 1:
        p.error("--worker-blas-threads must be >= 1")
    if args.workspace_mb <= 0:
        p.error("--workspace-mb must be positive")
    if args.null_memory_mb <= 0:
        p.error("--null-memory-mb must be positive")
    if args.n_permutations < 1:
        p.error("--n-permutations must be >= 1")
    if args.permutation_batch_size < 1:
        p.error("--permutation-batch-size must be >= 1")
    if args.empr and not args.expressionfile:
        p.error("--empr requires --expressionfile")
    if bool(args.emb) != bool(args.genelist):
        p.error("--emb and --genelist must be supplied together")
    if not args.index:
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
            p.error("standalone scoring requires " + ", ".join(missing))
    if args.cache_policy is None:
        args.cache_policy = "require" if args.index else "build"
    if args.cache_policy == "require" and args.rebuild_cache:
        p.error("--rebuild-cache requires --cache-policy build")
    if args.cache_policy == "require" and args.no_cache:
        p.error("--no-cache cannot be used with --cache-policy require")

    return args


def main(argv=None):
    """Run ranked enrichment from command-line inputs."""
    args = parse_args(argv)
    args.seed = resolve_null_seed(args.seed)
    print(f"Seed: {args.seed}")

    index = None
    index_load_seconds = 0.0
    if args.index:
        print(f"Loading ANDES index from {args.index}...")
        index_load_started = time.perf_counter()
        index = index_api.load_andes_index(args.index, mmap=True)
        index_load_seconds = time.perf_counter() - index_load_started
        node_list = list(index.gene_list)
        embedding_model = index.embedding_space()
        database_model = index.gene_set_database()

        # Optional raw inputs assert that the caller selected the same data.
        if args.emb:
            supplied = ld.load_embedding_space(args.emb, args.genelist)
            validate_index_compatibility(
                index,
                E_unit=supplied.vectors,
                gene_list=supplied.genes,
            )
        if args.geneset:
            supplied = ld.load_gene_set_database(
                args.geneset,
                embedding_model,
                min_size=args.min_size,
                max_size=args.max_size,
                sort_terms=True,
            )
            validate_index_compatibility(
                index,
                database=supplied,
            )
    else:
        print("Loading embedding and gene sets...")
        embedding_model = ld.load_embedding_space(args.emb, args.genelist)
        database_model = ld.load_gene_set_database(
            args.geneset,
            embedding_model,
            min_size=args.min_size,
            max_size=args.max_size,
            sort_terms=True,
        )

    E_unit = embedding_model.vectors
    node_list = list(embedding_model.genes)
    geneset_terms = list(database_model.terms)
    pop = np.asarray(database_model.background, dtype=np.int32)

    print(
        f"  {len(node_list)} genes  |  dim={E_unit.shape[1]}  |  "
        f"{len(geneset_terms)} terms\n"
        f"  E_unit: {E_unit.nbytes / 1e6:.1f} MB  "
        f"(vs {len(node_list) ** 2 * 8 / 1e9:.1f} GB for full similarity matrix)"
    )

    # A defaultdict(lambda: -1) would map missing genes to the last embedding
    # row and corrupt the result.
    g_node2index = {g: i for i, g in enumerate(node_list)}

    print(f"  {len(geneset_terms)} terms  |  {len(pop)} background genes")

    print("\nLoading ranked list...")
    expression_context = None

    if args.rankedlist:
        ranked_idx = ld.load_ranked_indices(args.rankedlist, g_node2index)
        print(f"  Loaded pre-ranked list: {len(ranked_idx)} genes")
    else:
        with query_blas_context(args.query_blas_threads):
            ranked_idx, expression_context = ranked_list_from_expression(
                args.expressionfile,
                g_node2index,
            )
        print(f"  Derived ranked list from expression file: {len(ranked_idx)} genes")

    if len(ranked_idx) == 0:
        raise ValueError("ranked list is empty after filtering to embedding genes")

    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    print(f"  ranked_emb: {ranked_emb.shape}  ({ranked_emb.nbytes / 1e3:.1f} KB)")

    sizes = {int(size) for size in database_model.sizes}
    null_plan = plan_ranked_null_runtime(
        requested_workers=args.workers,
        total_workspace_bytes=int(args.null_memory_mb * 1e6),
        iterations=args.ite,
        max_size=max(sizes),
        ranked_length=len(ranked_idx),
        embedding_dimensions=E_unit.shape[1],
        population_size=len(pop),
        n_sizes=len(sizes),
    )

    null_started = time.perf_counter()
    resolution = resolve_ranked_null(
        embedding_model,
        pop,
        sizes=sizes,
        ranked_embeddings=ranked_emb,
        runtime_plan=null_plan,
        path=None if args.no_cache or not args.cache else args.cache,
        base_dir=(
            None
            if args.no_cache or args.cache
            else null_cache_dir("ranked", args.cache_root or None)
        ),
        iterations=args.ite,
        seed=args.seed,
        no_build=args.cache_policy == "require",
        worker_blas_threads=args.worker_blas_threads,
        rebuild=args.rebuild_cache,
    )
    null_resolution_seconds = time.perf_counter() - null_started
    null_model = resolution.model
    if not isinstance(null_model, RankedNullModel):
        raise TypeError("ranked null resolver returned the wrong model kind")
    if resolution.path is not None:
        print(f"Null artifact: {resolution.path}")

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
            "null_workers_requested": int(args.workers),
            "null_workers_effective": int(null_plan.workers),
            "null_strategy": null_plan.strategy,
            "null_workspace_bytes_total": int(null_plan.total_workspace_bytes),
            "null_workspace_bytes_shared": int(null_plan.shared_workspace_bytes),
            "null_workspace_bytes_per_worker": int(
                null_plan.workspace_bytes_per_worker
            ),
            "null_cache_policy": args.cache_policy,
            "null_cache_built": bool(resolution.built),
            "null_cache_added_entries": int(resolution.added_entries),
            "null_cache_resolution_seconds": null_resolution_seconds,
            "index_load_seconds": index_load_seconds,
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
            embedding=embedding_model,
            database=database_model,
            ranked_indices=ranked_idx,
            bestmatch=(None if index is None else index.ranked_bestmatch_artifact()),
            workspace_mb=args.workspace_mb,
            null_model=null_model,
            runtime=runtime_metadata,
        )
        runtime_metadata["query_seconds"] = time.perf_counter() - t0
        ranked_result = replace(
            ranked_result,
            provenance=replace(
                ranked_result.provenance,
                runtime=runtime_metadata,
            ),
        )

    if ranked_result.scores.stats.workspace_bytes:
        print(
            f"{ranked_result.scores.stats.engine.capitalize()} score workspace: "
            f"~{ranked_result.scores.stats.workspace_bytes / 1e6:.1f} MB"
        )

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s  ({len(geneset_terms) / elapsed:.0f} terms/s)")

    out_path = Path(args.out)
    sidecar = application.write_ranked_result(
        ranked_result,
        out_path,
        provenance_extra={
            "expression_derived": bool(args.expressionfile),
        },
    )
    print(f"\nSaved z-scores to {out_path}")
    print(f"Provenance: {sidecar}")

    if args.empr:
        print(f"\nComputing empirical p-values ({args.n_permutations} permutations)...")
        if expression_context is None:
            raise RuntimeError("empirical scoring requires expression data")

        observed = np.asarray(ranked_result.true_scores, dtype=np.float32)
        term_sizes = np.asarray(database_model.sizes, dtype=np.int32)

        if index is not None:
            empirical_bestmatch = index.bestmatch
            owns_empirical_bestmatch = False
            bestmatch_source = "persistent_index"
            print("  Reusing persistent indexed best-match matrix")
        else:
            print("  Building reusable best-match matrix for permutations...")
            with query_blas_context(args.query_blas_threads):
                empirical_bestmatch, workspace_mb = bma.gene_to_term_best_match_matrix(
                    E_unit,
                    database_model.packed_axis,
                    max_workspace_mb=args.workspace_mb,
                )
            owns_empirical_bestmatch = True
            bestmatch_source = "transient_build"
            print(f"  Best-match build workspace: ~{workspace_mb:.1f} MB")

        with (
            tqdm(
                total=args.n_permutations,
                desc="Permutations",
            ) as progress,
            query_blas_context(args.query_blas_threads),
        ):
            empirical_result = empirical_exceedance_counts(
                empirical_bestmatch,
                term_sizes,
                observed,
                expression_context["values"],
                expression_context["condition"],
                expression_context["row_to_embedding"],
                n_permutations=args.n_permutations,
                seed=args.seed,
                permutation_batch_size=args.permutation_batch_size,
                max_workspace_mb=args.workspace_mb,
                progress_callback=progress.update,
            )
        exceedance_counts = empirical_result.counts

        empirical_p = monte_carlo_pvalues(
            exceedance_counts,
            args.n_permutations,
        )
        if owns_empirical_bestmatch:
            del empirical_bestmatch

        p_df = pd.DataFrame(
            {
                "true_score": observed,
                "z_score": np.asarray(ranked_result.scores.scores, dtype=np.float32),
                "empirical_pval": empirical_p,
            },
            index=geneset_terms,
        )
        p_out = out_path.with_name(out_path.stem + "_empirical_pval.csv")
        application.write_dataframe_atomic(p_df, p_out)
        empirical_base = replace(
            ranked_result.provenance,
            method="andes_ranked_empirical",
            score_engine="bestmatch_matrix",
            score_kind="empirical_p_value",
            extra={
                **dict(ranked_result.provenance.extra),
                "bestmatch_source": bestmatch_source,
                "ranked_genes": len(ranked_idx),
                "terms": len(geneset_terms),
                "permutations": int(args.n_permutations),
                "permutation_batch_size": int(args.permutation_batch_size),
                "p_value_correction": "plus_one",
                "seed": int(args.seed),
                "empirical_exceedance_dtype": "int64",
                "indexed_workspace_bytes": int(
                    empirical_result.stats.indexed_workspace_bytes
                ),
                "requested_workspace_bytes": int(
                    empirical_result.stats.requested_workspace_bytes
                ),
                "term_chunk_size": int(empirical_result.stats.term_chunk_size),
                "phenotype_batches": int(empirical_result.stats.batches),
            },
        )
        empirical_provenance = empirical_base.for_tabular_output(
            {
                "term": "string",
                **{str(column): str(dtype) for column, dtype in p_df.dtypes.items()},
            }
        )
        empirical_sidecar = write_result_sidecar(p_out, empirical_provenance)
        print(f"Saved empirical p-values to {p_out}")
        print(f"Provenance: {empirical_sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
