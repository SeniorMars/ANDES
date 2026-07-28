"""Scientific run provenance and result sidecars."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from . import artifacts


def software_version() -> str:
    try:
        return importlib_metadata.version("andes")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0+source"


def metadata_sidecar_path(output_path) -> Path:
    output = Path(output_path)
    if output.suffix:
        return output.with_suffix(".metadata.json")
    return output.with_name(output.name + ".metadata.json")


@dataclass(frozen=True, slots=True)
class RunProvenance:
    method: str
    score_engine: str
    score_kind: str
    numeric_dtype: str
    accumulator_dtype: str
    embedding_fingerprint: str
    left_database_fingerprint: str
    right_database_fingerprint: str | None = None
    symmetric_reuse: bool = False
    null_spec: Mapping[str, object] | None = None
    runtime: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    extra: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    software_version: str = field(default_factory=software_version)

    def to_dict(self) -> dict:
        payload = {
            "method": self.method,
            "score_engine": self.score_engine,
            "score_kind": self.score_kind,
            "numeric_dtype": self.numeric_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "embedding_fingerprint": self.embedding_fingerprint,
            "left_database_fingerprint": self.left_database_fingerprint,
            "right_database_fingerprint": self.right_database_fingerprint,
            "symmetric_reuse": self.symmetric_reuse,
            "null_spec": (
                None if self.null_spec is None else dict(self.null_spec)
            ),
            "runtime": dict(self.runtime),
            "extra": dict(self.extra),
            "software_version": self.software_version,
        }
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        return payload


def write_result_sidecar(output_path, provenance: RunProvenance) -> Path:
    """Write provenance next to an already-complete result payload."""
    output = Path(output_path)
    if not output.is_file():
        raise FileNotFoundError(f"result payload does not exist: {output}")
    payload = provenance.to_dict()
    payload["output_file"] = output.name
    payload["output_size_bytes"] = output.stat().st_size
    payload["output_hash"] = artifacts.hash_file(output)
    sidecar = metadata_sidecar_path(output)
    artifacts.write_json_atomic(sidecar, payload)
    return sidecar
