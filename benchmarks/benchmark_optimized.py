"""Benchmark the current optimized ANDES scoring paths."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from andes import data, index, scoring

ROOT = Path(__file__).resolve().parents[1]
ResultT = TypeVar("ResultT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="use the bundled embedding, GO database, and GSE3467 ranking",
    )
    parser.add_argument("--emb", default="")
    parser.add_argument("--genelist", default="")
    parser.add_argument("--geneset", default="")
    parser.add_argument("--rankedlist", default="")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=300)
    parser.add_argument("--genes", type=int, default=2000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--terms", type=int, default=300)
    parser.add_argument("--term-size", type=int, default=40)
    parser.add_argument("--ranked-length", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workspace-mb", type=float, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    supplied_real_paths = any((args.emb, args.genelist, args.geneset, args.rankedlist))
    if args.real_data:
        args.emb = args.emb or str(
            ROOT / "data" / "embedding" / "node2vec_consensus.csv"
        )
        args.genelist = args.genelist or str(
            ROOT / "data" / "embedding" / "consensus_node.txt"
        )
        args.geneset = args.geneset or str(
            ROOT / "data" / "gene_sets" / "hsa_experimental_eval_BP_propagated.gmt"
        )
        args.rankedlist = args.rankedlist or str(
            ROOT / "data" / "expression" / "GSE3467_rank.txt"
        )
    elif supplied_real_paths and not all(
        (args.emb, args.genelist, args.geneset, args.rankedlist)
    ):
        parser.error(
            "--emb, --genelist, --geneset, and --rankedlist must be supplied together"
        )
    for name in ("genes", "dimensions", "terms", "term_size", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.term_size > args.genes:
        parser.error("--term-size cannot exceed --genes")
    if not 1 <= args.ranked_length <= args.genes:
        parser.error("--ranked-length must be between 1 and --genes")
    if args.workspace_mb <= 0:
        parser.error("--workspace-mb must be positive")
    if args.min_size < 1 or args.max_size < args.min_size:
        parser.error("invalid real-data size range")
    return args


def make_inputs(args):
    if args.emb:
        embedding = data.load_embedding_space(args.emb, args.genelist)
        database = data.load_gene_set_database(
            args.geneset,
            embedding,
            min_size=args.min_size,
            max_size=args.max_size,
            sort_terms=True,
        )
        ranked = data.load_ranked_indices(
            args.rankedlist,
            embedding.gene_to_index,
        )
        return embedding, database, ranked, "real"

    rng = np.random.default_rng(args.seed)
    embedding = data.EmbeddingSpace.from_arrays(
        rng.normal(size=(args.genes, args.dimensions)).astype(np.float32),
        [f"g{i}" for i in range(args.genes)],
    )
    memberships = {
        f"term_{i:05d}": np.sort(
            rng.choice(
                args.genes,
                size=args.term_size,
                replace=False,
            ).astype(np.int32)
        )
        for i in range(args.terms)
    }
    database = data.GeneSetDatabase.from_index_mapping(
        memberships,
        n_genes=args.genes,
        embedding_fingerprint=embedding.fingerprint,
        background_policy="retained_members",
    )
    ranked = rng.permutation(args.genes)[: args.ranked_length].astype(np.int32)
    return embedding, database, ranked, "synthetic"


def median_seconds(
    repeats: int,
    operation: Callable[[], ResultT],
) -> tuple[float, list[float], ResultT]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    samples = []
    result: ResultT | None = None
    for _ in range(repeats):
        if result is not None:
            del result
            result = None
            gc.collect()
        started = time.perf_counter()
        current = operation()
        samples.append(time.perf_counter() - started)
        result = current
    if result is None:
        raise RuntimeError("benchmark operation did not produce a result")
    return statistics.median(samples), samples, result


def peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main(argv=None):
    args = parse_args(argv)
    input_started = time.perf_counter()
    embedding, database, ranked, input_mode = make_inputs(args)
    input_seconds = time.perf_counter() - input_started
    total_memberships = int(database.members.size)
    unique_members = int(np.unique(database.members).size)
    canonical_sets = {
        tuple(database.members_at(position).tolist())
        for position in range(len(database.terms))
    }

    # Compile the ranked kernels before collecting timings.
    scoring.score_ranked(
        embedding,
        database,
        ranked,
        workspace_mb=args.workspace_mb,
    )

    bma_median, bma_samples, bma_result = median_seconds(
        args.repeats,
        lambda: scoring.score_bma_matrix(
            embedding,
            database,
            database,
            workspace_mb=args.workspace_mb,
        ),
    )
    bma_workspace_bytes = bma_result.stats.workspace_bytes
    del bma_result
    gc.collect()

    ranked_median, ranked_samples, standalone = median_seconds(
        args.repeats,
        lambda: scoring.score_ranked(
            embedding,
            database,
            ranked,
            workspace_mb=args.workspace_mb,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="andes-index-benchmark-") as root:
        index_dir = Path(root) / "pathways.index"
        build_started = time.perf_counter()
        build_metadata = index.build_andes_index(
            embedding,
            database,
            index_dir,
            max_workspace_mb=args.workspace_mb,
        )
        index_build_seconds = time.perf_counter() - build_started

        metadata_load_median, metadata_load_samples, loaded = median_seconds(
            args.repeats,
            lambda: index.load_andes_index(
                index_dir,
                mmap=True,
                verify="metadata",
            ),
        )
        del loaded
        gc.collect()
        full_audit_median, full_audit_samples, audited = median_seconds(
            args.repeats,
            lambda: index.load_andes_index(
                index_dir,
                mmap=True,
                verify="full",
            ),
        )
        del audited
        gc.collect()

        loaded = index.load_andes_index(
            index_dir,
            mmap=True,
            verify="metadata",
        )
        indexed_embedding = loaded.embedding_space()
        indexed_database = loaded.gene_set_database()
        artifact = loaded.ranked_bestmatch_artifact()

        def score_loaded_index(
            indexed_embedding=indexed_embedding,
            indexed_database=indexed_database,
            artifact=artifact,
        ):
            return scoring.score_ranked(
                indexed_embedding,
                indexed_database,
                ranked,
                bestmatch=artifact,
                workspace_mb=args.workspace_mb,
            )

        indexed_median, indexed_samples, indexed = median_seconds(
            args.repeats,
            score_loaded_index,
        )
        bestmatch_mb = loaded.bestmatch.nbytes / 1e6
        del (
            artifact,
            indexed_database,
            indexed_embedding,
            loaded,
            score_loaded_index,
        )
        gc.collect()

    np.testing.assert_allclose(
        indexed.scores,
        standalone.scores,
        rtol=2e-5,
        atol=2e-5,
    )
    ranked_signed_max_abs_diff = float(
        np.max(np.abs(indexed.scores - standalone.scores))
    )
    indexed_workspace_bytes = indexed.stats.workspace_bytes

    construction_seconds = build_metadata["bestmatch_construction_seconds"]
    payload_seconds = build_metadata["payload_build_seconds"]
    if not isinstance(construction_seconds, (int, float)):
        raise TypeError("index metadata has a non-numeric construction time")
    if not isinstance(payload_seconds, (int, float)):
        raise TypeError("index metadata has a non-numeric payload time")

    report = {
        "config": {
            "input_mode": input_mode,
            "genes": len(embedding.genes),
            "dimensions": int(embedding.vectors.shape[1]),
            "terms": len(database.terms),
            "ranked_length": int(ranked.size),
            "repeats": args.repeats,
            "workspace_mb": args.workspace_mb,
            "seed": args.seed,
        },
        "database": {
            "membership_occurrences": total_memberships,
            "unique_member_genes": unique_members,
            "redundancy_ratio": (
                total_memberships / unique_members if unique_members else 0.0
            ),
            "duplicate_term_sets": len(database.terms) - len(canonical_sets),
        },
        "seconds": {
            "input_loading": input_seconds,
            "bma_streamed_median": bma_median,
            "bma_streamed_samples": bma_samples,
            "ranked_bestmatch_median": ranked_median,
            "ranked_bestmatch_samples": ranked_samples,
            "index_build_and_publish": index_build_seconds,
            "index_numerical_construction": float(construction_seconds),
            "index_payload_build": float(payload_seconds),
            "index_publish_and_full_audit": max(
                0.0,
                index_build_seconds - float(payload_seconds),
            ),
            "index_metadata_load_median": metadata_load_median,
            "index_metadata_load_samples": metadata_load_samples,
            "index_full_audit_median": full_audit_median,
            "index_full_audit_samples": full_audit_samples,
            "ranked_indexed_median": indexed_median,
            "ranked_indexed_samples": indexed_samples,
        },
        "index": {
            "bestmatch_mb": bestmatch_mb,
            "build_workspace_mb": build_metadata["chunk_workspace_mb"],
            "ranked_signed_max_abs_diff": ranked_signed_max_abs_diff,
        },
        "memory": {
            "process_peak_rss_bytes": peak_rss_bytes(),
            "scope": "cumulative peak for this benchmark process",
        },
        "planner_estimates": {
            "bma_planned_workspace_bytes": bma_workspace_bytes,
            "ranked_standalone_planned_workspace_bytes": (
                standalone.stats.workspace_bytes
            ),
            "ranked_indexed_planned_workspace_bytes": indexed_workspace_bytes,
        },
        "system": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "blas_pools": [
                {
                    "internal_api": pool.get("internal_api", "unknown"),
                    "prefix": pool.get("prefix", "unknown"),
                    "num_threads": int(pool.get("num_threads", 0)),
                    "version": pool.get("version", "unknown"),
                }
                for pool in threadpool_info()
                if pool.get("user_api") == "blas"
            ],
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
