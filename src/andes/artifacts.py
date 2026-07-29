"""Shared fingerprints and atomic artifact helpers for the ANDES package.

Numerical kernels, data constructors, and artifact loaders can import this
module without pulling in scoring or command-line policy.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse  # pyright: ignore[reportMissingTypeStubs]

FINGERPRINT_DIGEST_BYTES = 16
ARRAY_CHUNK_BYTES = 16_000_000
ArtifactPath: TypeAlias = str | Path
DEFAULT_LOCK_TIMEOUT_SECONDS = 3600.0


def iter_array_row_chunks(
    array: ArrayLike,
    *,
    chunk_bytes: int = ARRAY_CHUNK_BYTES,
) -> Iterator[NDArray[np.generic]]:
    """Yield row-aligned array views bounded by a target byte count."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    value = np.asarray(array)
    if value.ndim == 0:
        yield value.reshape(1)
        return
    if value.shape[0] == 0:
        return

    row_bytes = int(np.prod(value.shape[1:], dtype=np.int64)) * value.dtype.itemsize
    rows_per_chunk = max(1, int(chunk_bytes) // max(1, row_bytes))
    for start in range(0, value.shape[0], rows_per_chunk):
        yield value[start : start + rows_per_chunk]


def hash_array(
    array: ArrayLike,
    digest_size: int = FINGERPRINT_DIGEST_BYTES,
    *,
    chunk_bytes: int = ARRAY_CHUNK_BYTES,
) -> str:
    """Hash canonical C-order array bytes without a full-size copy."""
    value = np.asarray(array)
    if value.ndim == 0:
        value = value.reshape(1)
    digest = hashlib.blake2b(digest_size=int(digest_size))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    for block in iter_array_row_chunks(value, chunk_bytes=chunk_bytes):
        contiguous = np.ascontiguousarray(block)
        digest.update(contiguous.data.cast("B"))
    return digest.hexdigest()


def hash_strings(
    values: Sequence[str],
    digest_size: int = FINGERPRINT_DIGEST_BYTES,
) -> str:
    """Hash an ordered string sequence with unambiguous item boundaries."""
    digest = hashlib.blake2b(digest_size=int(digest_size))
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_sparse_csr(
    matrix,
    digest_size: int = FINGERPRINT_DIGEST_BYTES,
) -> str:
    """Hash a CSR matrix including its canonical sparse representation."""
    value = sparse.csr_matrix(matrix, copy=True)
    value.sum_duplicates()
    value.sort_indices()
    digest = hashlib.blake2b(digest_size=int(digest_size))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    for part in (value.indptr, value.indices, value.data):
        digest.update(hash_array(part, digest_size=int(digest_size)).encode("ascii"))
    return digest.hexdigest()


def hash_file(path, digest_size: int = FINGERPRINT_DIGEST_BYTES) -> str:
    """Hash a file payload without loading it completely into memory."""
    digest = hashlib.blake2b(digest_size=int(digest_size))
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine_fingerprints(
    kind: str,
    components: Mapping[str, str],
    digest_size: int = FINGERPRINT_DIGEST_BYTES,
) -> str:
    """Combine named component hashes into a version-stable fingerprint."""
    digest = hashlib.blake2b(digest_size=int(digest_size))
    digest.update(str(kind).encode("utf-8"))
    digest.update(b"\0")
    for name in sorted(components):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"=")
        digest.update(str(components[name]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: ArtifactPath) -> object:
    """Read a UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(
    path: ArtifactPath,
    payload: object,
    *,
    indent: int = 2,
) -> None:
    """Atomically replace one JSON file on the same filesystem."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def save_npy_atomic(path: ArtifactPath, array: ArrayLike) -> None:
    """Atomically replace one ``.npy`` payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".npy",
        dir=target.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, np.asarray(array), allow_pickle=False)
        os.replace(temporary, target)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


@contextmanager
def artifact_lock(
    path: ArtifactPath,
    *,
    shared: bool = False,
    create: bool = True,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = 0.1,
) -> Iterator[Path]:
    """Hold a shared reader or exclusive writer lock for one artifact target.

    Lock files are stable hidden siblings of their artifacts and remain after
    release. Removing a lock file while another process is waiting on its inode
    can split future writers across two independent locks. A shared lock with
    ``create=False`` is a read-only operation and rejects unpublished artifacts
    that lack the sibling lock.
    """
    timeout_seconds = float(timeout_seconds)
    poll_seconds = float(poll_seconds)
    if timeout_seconds < 0.0:
        raise ValueError("lock timeout must be non-negative")
    if poll_seconds <= 0.0:
        raise ValueError("lock poll interval must be positive")
    if not create and not shared:
        raise ValueError("a non-creating artifact lock must be shared")

    target = Path(path)
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    if shared and not create:
        try:
            handle = lock_path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"artifact lock does not exist: {lock_path}; "
                "republish or adopt the artifact before read-only deployment"
            ) from exc
    elif shared:
        handle = lock_path.open("a+", encoding="utf-8")
    else:
        handle = lock_path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for artifact lock: {lock_path}"
                    ) from exc
                time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

        if not shared:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield lock_path
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def atomic_artifact_directory(
    path,
    *,
    overwrite: bool = False,
) -> Iterator[Path]:
    """Publish a sibling temporary directory after the context succeeds.

    Existing targets are rejected by default. With ``overwrite=True``, the old
    target moves to a private sibling backup and is restored if publication
    fails.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_dir():
        raise ValueError(f"artifact target must be a directory: {target}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.building.", dir=target.parent)
    )
    backup = None
    try:
        yield temporary
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"artifact target already exists: {target}")
            backup = target.with_name(f".{target.name}.previous.{os.getpid()}")
            if backup.exists():
                raise FileExistsError(f"artifact backup already exists: {backup}")
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except BaseException:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
