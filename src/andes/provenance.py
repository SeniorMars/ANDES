"""Scientific run provenance and result sidecars."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType

from . import artifacts


def software_version() -> str:
    try:
        return importlib_metadata.version("andes")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0+source"


def metadata_sidecar_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix:
        return output.with_suffix(".metadata.json")
    return output.with_name(output.name + ".metadata.json")


@dataclass(frozen=True, slots=True)
class RunProvenance:
    method: str
    score_engine: str
    score_kind: str
    similarity_dtype: str
    score_accumulator_dtype: str
    null_accumulator_dtype: str
    output_dtype: str
    tie_policy: str
    embedding_fingerprint: str
    left_database_fingerprint: str
    right_database_fingerprint: str | None = None
    symmetric_reuse: bool = False
    null_spec: Mapping[str, object] | None = None
    runtime: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    extra: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    software_version: str = field(default_factory=software_version)

    def __post_init__(self) -> None:
        if self.null_spec is not None:
            object.__setattr__(
                self,
                "null_spec",
                MappingProxyType(dict(self.null_spec)),
            )
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def for_tabular_output(
        self,
        column_dtypes: Mapping[str, str],
        *,
        extra: Mapping[str, object] | None = None,
    ) -> RunProvenance:
        """Derive provenance for a heterogeneous table without losing run data."""
        logical_dtypes = {
            str(column): str(dtype) for column, dtype in column_dtypes.items()
        }
        if not logical_dtypes:
            raise ValueError("tabular output must describe at least one column")
        distinct = set(logical_dtypes.values())
        output_dtype = next(iter(distinct)) if len(distinct) == 1 else "mixed"
        merged_extra = dict(self.extra)
        merged_extra.update(dict(extra or {}))
        merged_extra["output_column_dtypes"] = logical_dtypes
        return replace(
            self,
            output_dtype=output_dtype,
            extra=MappingProxyType(merged_extra),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "method": self.method,
            "score_engine": self.score_engine,
            "score_kind": self.score_kind,
            "similarity_dtype": self.similarity_dtype,
            "score_accumulator_dtype": self.score_accumulator_dtype,
            "null_accumulator_dtype": self.null_accumulator_dtype,
            "output_dtype": self.output_dtype,
            "tie_policy": self.tie_policy,
            "embedding_fingerprint": self.embedding_fingerprint,
            "left_database_fingerprint": self.left_database_fingerprint,
            "right_database_fingerprint": self.right_database_fingerprint,
            "symmetric_reuse": self.symmetric_reuse,
            "null_spec": (None if self.null_spec is None else dict(self.null_spec)),
            "runtime": dict(self.runtime),
            "extra": dict(self.extra),
            "software_version": self.software_version,
        }
        payload["created_at"] = datetime.now(UTC).isoformat()
        return payload


def write_result_sidecar(
    output_path: str | Path,
    provenance: RunProvenance,
    *,
    companion_paths: Iterable[str | Path] = (),
) -> Path:
    """Publish a manifest for one complete result and its companion files."""
    output = Path(output_path)
    if not output.is_file():
        raise FileNotFoundError(f"result payload does not exist: {output}")
    companions = tuple(Path(path) for path in companion_paths)
    for companion in companions:
        if companion.parent.resolve() != output.parent.resolve():
            raise ValueError("result companion files must share the output directory")
        if not companion.is_file():
            raise FileNotFoundError(f"result companion does not exist: {companion}")
    files = (output, *companions)
    if len({path.name for path in files}) != len(files):
        raise ValueError("result payload and companion filenames must be unique")

    payload = provenance.to_dict()
    payload["manifest_version"] = 1
    payload["output_file"] = output.name
    payload["output_size_bytes"] = output.stat().st_size
    payload["output_hash"] = artifacts.hash_file(output)
    payload["files"] = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "hash": artifacts.hash_file(path),
        }
        for path in files
    }
    sidecar = metadata_sidecar_path(output)
    artifacts.write_json_atomic(sidecar, payload)
    return sidecar
