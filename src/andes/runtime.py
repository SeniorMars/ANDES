"""Explicit BLAS and worker execution policy for ANDES applications."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from threadpoolctl import threadpool_info, threadpool_limits


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    workers: int = 1
    query_blas_threads: int = 0
    worker_blas_threads: int = 1
    workspace_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self):
        if int(self.workers) < 1:
            raise ValueError("workers must be at least one")
        if int(self.query_blas_threads) < 0:
            raise ValueError("query_blas_threads must be non-negative")
        if int(self.worker_blas_threads) < 1:
            raise ValueError("worker_blas_threads must be at least one")
        if int(self.workspace_bytes) < 0:
            raise ValueError("workspace_bytes must be non-negative")


def numba_warmup_requirements(
    query_mode,
    null_mode,
    numba_threshold,
    size_pairs,
    needs_null_build,
):
    """Return the exact optional kernels that the requested run can execute."""
    warm_bma = (
        query_mode == "pairwise"
        and int(numba_threshold) > 0
        and any(int(m) * int(k) < int(numba_threshold) for m, k in size_pairs)
    )
    warm_fys = (
        bool(needs_null_build)
        and null_mode == "pairwise"
        and bool(size_pairs)
    )
    return warm_bma, warm_fys


def resolve_query_blas_limit(query_mode, requested_threads):
    """Resolve main-query BLAS policy without mutating the environment."""
    requested_threads = int(requested_threads)
    if requested_threads < 0:
        raise ValueError("query BLAS threads must be non-negative")
    if requested_threads > 0:
        return requested_threads
    return None if query_mode == "bestmatch" else 1


def query_blas_context(requested_threads):
    """Return a scoped main-process BLAS limit for a direct thread count."""
    threads = int(requested_threads)
    if threads < 0:
        raise ValueError("query BLAS threads must be non-negative")
    if threads == 0:
        return nullcontext()
    return threadpool_limits(limits=threads, user_api="blas")


def blas_runtime_info():
    """Return compact, JSON-safe metadata for active BLAS pools."""
    return [
        {
            "internal_api": pool.get("internal_api", "unknown"),
            "prefix": pool.get("prefix", "unknown"),
            "num_threads": int(pool.get("num_threads", 0)),
        }
        for pool in threadpool_info()
        if pool.get("user_api") == "blas"
    ]


def format_blas_runtime(pools=None):
    """Format explicit or currently active BLAS pool metadata."""
    pools = blas_runtime_info() if pools is None else pools
    if not pools:
        return "BLAS runtime not reported"
    return ", ".join(
        f"{pool.get('internal_api', 'unknown')}:{pool.get('num_threads', 0)}"
        for pool in pools
    )
