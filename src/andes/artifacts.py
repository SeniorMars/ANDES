"""Shared fingerprints and atomic artifact helpers for the ANDES package.

This module deliberately contains no scoring or command-line policy.  It is
safe to import from numerical kernels, data-model constructors, and artifact
loaders without creating dependency cycles or process-wide side effects.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Mapping, Sequence

import numpy as np
from scipy import sparse


FINGERPRINT_DIGEST_BYTES = 16


def hash_array(array, digest_size: int = FINGERPRINT_DIGEST_BYTES) -> str:
    """Hash an array's shape, dtype, and canonical contiguous byte payload."""
    value = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=int(digest_size))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.view(np.uint8))
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


def read_json(path) -> object:
    """Read a UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, payload, *, indent: int = 2) -> None:
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def save_npy_atomic(path, array) -> None:
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def atomic_artifact_directory(
    path,
    *,
    overwrite: bool = False,
) -> Iterator[Path]:
    """Yield a sibling temporary directory and publish it on success.

    The final target is never observed partially built.  Existing targets are
    rejected by default.  ``overwrite=True`` publishes the new directory only
    after moving the old target to a private sibling backup; the backup is
    restored if publication fails.
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
