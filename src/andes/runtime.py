"""Explicit BLAS and worker execution policy for ANDES applications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import TypeAlias, cast

from threadpoolctl import threadpool_info, threadpool_limits

BlasPool: TypeAlias = dict[str, str | int]


def query_blas_context(requested_threads: int):
    """Return a scoped main-process BLAS limit for a direct thread count."""
    threads = int(requested_threads)
    if threads < 0:
        raise ValueError("query BLAS threads must be non-negative")
    if threads == 0:
        return nullcontext()
    return threadpool_limits(limits=threads, user_api="blas")


def blas_runtime_info() -> list[BlasPool]:
    """Return compact, JSON-safe metadata for active BLAS pools."""
    result: list[BlasPool] = []
    pools = cast(Sequence[Mapping[str, object]], threadpool_info())
    for pool in pools:
        if pool.get("user_api") != "blas":
            continue
        raw_threads = pool.get("num_threads", 0)
        num_threads = int(raw_threads) if isinstance(raw_threads, (int, str)) else 0
        result.append(
            {
                "internal_api": str(pool.get("internal_api", "unknown")),
                "prefix": str(pool.get("prefix", "unknown")),
                "num_threads": num_threads,
            }
        )
    return result


def format_blas_runtime(
    pools: Sequence[Mapping[str, object]] | None = None,
) -> str:
    """Format explicit or currently active BLAS pool metadata."""
    pools = blas_runtime_info() if pools is None else pools
    if not pools:
        return "BLAS runtime not reported"
    return ", ".join(
        f"{pool.get('internal_api', 'unknown')}:{pool.get('num_threads', 0)}"
        for pool in pools
    )
