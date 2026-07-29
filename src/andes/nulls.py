"""Null-calibration models and their on-disk artifacts.

BMA calibration uses a two-dimensional lookup keyed by
``(left_size, right_size)``. Ranked calibration uses a one-dimensional lookup
keyed by gene-set size. A boolean array marks available entries. Artifacts
contain JSON metadata and NumPy arrays.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from threadpoolctl import threadpool_limits

from . import artifacts
from .data import EmbeddingSpace

NULL_ARTIFACT_VERSION = 2
_ARRAY_FILES = ("means.npy", "stds.npy", "present.npy")
DEFAULT_CACHE_ROOT = Path("cache")

BmaNullValues = Mapping[tuple[int, int], tuple[float, float]]
RankedNullValues = Mapping[int, tuple[float, float]]
Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _json_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _json_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _json_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a list of strings")
    return tuple(
        _json_string(item, f"{name}[{position}]") for position, item in enumerate(value)
    )


def _json_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("null metadata shape must be a list of integers")
    return tuple(
        _json_int(item, f"null metadata shape[{position}]")
        for position, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class NullSpec:
    """Scientific identity of one null model."""

    kind: str
    iterations: int
    seed: int
    sampling: str
    ddof: int
    embedding_hash: str
    population_hashes: tuple[str, ...]
    ranked_hash: str | None = None
    tie_policy: str | None = None

    def __post_init__(self):
        if self.kind not in {"bma", "ranked"}:
            raise ValueError(f"unsupported null kind {self.kind!r}")
        if self.sampling != "prefix_coupled":
            raise ValueError(
                f"only prefix-coupled null sampling is supported; got {self.sampling!r}"
            )
        if int(self.ddof) < 0:
            raise ValueError("null ddof must be non-negative")
        if int(self.iterations) <= int(self.ddof):
            raise ValueError("null iterations must be greater than ddof")
        if int(self.seed) < 0:
            raise ValueError("null seed must be resolved before artifact creation")
        if not self.embedding_hash:
            raise ValueError("embedding_hash must not be empty")
        expected_populations = 2 if self.kind == "bma" else 1
        if len(self.population_hashes) != expected_populations:
            raise ValueError(
                f"{self.kind} null requires {expected_populations} population hash(es)"
            )
        if any(not value for value in self.population_hashes):
            raise ValueError("population hashes must not be empty")
        if self.kind == "ranked" and not self.ranked_hash:
            raise ValueError("ranked null requires ranked_hash")
        if self.kind == "bma" and self.ranked_hash is not None:
            raise ValueError("BMA null must not define ranked_hash")
        if self.kind == "ranked" and not self.tie_policy:
            raise ValueError("ranked null requires tie_policy")
        if self.kind == "bma" and self.tie_policy is not None:
            raise ValueError("BMA null must not define tie_policy")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["population_hashes"] = list(self.population_hashes)
        return payload

    @property
    def fingerprint(self) -> str:
        """Return a stable identity covering every null-model parameter."""
        components = {
            "kind": self.kind,
            "iterations": str(self.iterations),
            "seed": str(self.seed),
            "sampling": self.sampling,
            "ddof": str(self.ddof),
            "embedding": self.embedding_hash,
            **{
                f"population_{i}": value
                for i, value in enumerate(self.population_hashes)
            },
        }
        if self.ranked_hash is not None:
            components["ranking"] = self.ranked_hash
        if self.tie_policy is not None:
            components["tie_policy"] = self.tie_policy
        return artifacts.combine_fingerprints(
            "andes_null_spec_v1",
            components,
        )

    @classmethod
    def from_dict(cls, payload: object) -> NullSpec:
        if not isinstance(payload, Mapping):
            raise TypeError("null spec must be a JSON object")
        required = {
            "kind",
            "iterations",
            "seed",
            "sampling",
            "ddof",
            "embedding_hash",
            "population_hashes",
            "tie_policy",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError("null spec is missing: " + ", ".join(missing))
        ranked_hash_value = payload.get("ranked_hash")
        tie_policy_value = payload.get("tie_policy")
        return cls(
            kind=_json_string(payload["kind"], "kind"),
            iterations=_json_int(payload["iterations"], "iterations"),
            seed=_json_int(payload["seed"], "seed"),
            sampling=_json_string(payload["sampling"], "sampling"),
            ddof=_json_int(payload["ddof"], "ddof"),
            embedding_hash=_json_string(
                payload["embedding_hash"],
                "embedding_hash",
            ),
            population_hashes=_json_string_tuple(
                payload["population_hashes"],
                "population_hashes",
            ),
            ranked_hash=(
                None
                if ranked_hash_value is None
                else _json_string(ranked_hash_value, "ranked_hash")
            ),
            tie_policy=(
                None
                if tie_policy_value is None
                else _json_string(tie_policy_value, "tie_policy")
            ),
        )


def _readonly(values, dtype):
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _validate_arrays(means, stds, present, *, ndim):
    means = np.asarray(means)
    stds = np.asarray(stds)
    present = np.asarray(present)
    if means.ndim != ndim:
        raise ValueError(f"null means must be {ndim}-dimensional")
    if stds.shape != means.shape or present.shape != means.shape:
        raise ValueError("null means, stds, and present arrays must have one shape")
    if means.dtype != np.float64 or stds.dtype != np.float64:
        raise TypeError("null means and stds must use float64")
    if present.dtype != np.bool_:
        raise TypeError("null present array must use bool")
    if np.any(stds[present] < 0.0):
        raise ValueError("null standard deviations must be non-negative")
    if not np.isfinite(means[present]).all() or not np.isfinite(stds[present]).all():
        raise ValueError("present null entries must be finite")
    return (
        _readonly(means, np.float64),
        _readonly(stds, np.float64),
        _readonly(present, np.bool_),
    )


def _artifact_metadata(kind, spec, means, stds, present):
    return {
        "artifact": "andes_null",
        "artifact_version": NULL_ARTIFACT_VERSION,
        "kind": kind,
        "spec": spec.to_dict(),
        "shape": list(means.shape),
        "dtypes": {
            "means": str(means.dtype),
            "stds": str(stds.dtype),
            "present": str(present.dtype),
        },
        "hashes": {
            "means": artifacts.hash_array(means),
            "stds": artifacts.hash_array(stds),
            "present": artifacts.hash_array(present),
        },
    }


def _save_model(path, kind, spec, means, stds, present, *, overwrite):
    with artifacts.atomic_artifact_directory(path, overwrite=overwrite) as building:
        np.save(building / "means.npy", means, allow_pickle=False)
        np.save(building / "stds.npy", stds, allow_pickle=False)
        np.save(building / "present.npy", present, allow_pickle=False)
        artifacts.write_json_atomic(
            building / "metadata.json",
            _artifact_metadata(kind, spec, means, stds, present),
        )


def _load_payload(path, expected_kind):
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"null artifact is not a directory: {root}")
    required = ("metadata.json", *_ARRAY_FILES)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(
            f"invalid null artifact at {root}: missing {', '.join(missing)}"
        )
    metadata = artifacts.read_json(root / "metadata.json")
    if not isinstance(metadata, Mapping):
        raise TypeError("null metadata must be a JSON object")
    if metadata.get("artifact") != "andes_null":
        raise ValueError("not an ANDES null artifact")
    if metadata.get("artifact_version") != NULL_ARTIFACT_VERSION:
        raise ValueError(
            "unsupported null artifact version "
            f"{metadata.get('artifact_version')!r}; rebuild the artifact"
        )
    if metadata.get("kind") != expected_kind:
        raise ValueError(
            f"expected {expected_kind} null, found {metadata.get('kind')!r}"
        )

    means = np.load(root / "means.npy", allow_pickle=False)
    stds = np.load(root / "stds.npy", allow_pickle=False)
    present = np.load(root / "present.npy", allow_pickle=False)
    expected_shape = _json_shape(metadata.get("shape"))
    if means.shape != expected_shape:
        raise ValueError(
            f"null means shape {means.shape} does not match {expected_shape}"
        )
    expected_dtypes = metadata.get("dtypes", {})
    actual_dtypes = {
        "means": str(means.dtype),
        "stds": str(stds.dtype),
        "present": str(present.dtype),
    }
    if expected_dtypes != actual_dtypes:
        raise ValueError(f"null array dtype metadata mismatch: {actual_dtypes}")
    expected_hashes = metadata.get("hashes", {})
    actual_hashes = {
        "means": artifacts.hash_array(means),
        "stds": artifacts.hash_array(stds),
        "present": artifacts.hash_array(present),
    }
    if expected_hashes != actual_hashes:
        raise ValueError("null array hash mismatch; artifact is corrupt or stale")
    return NullSpec.from_dict(metadata.get("spec")), means, stds, present


@dataclass(frozen=True, slots=True)
class BmaNullModel:
    """Dense, explicitly masked BMA null calibration."""

    spec: NullSpec
    means: Float64Array
    stds: Float64Array
    present: BoolArray

    def __post_init__(self):
        if self.spec.kind != "bma":
            raise ValueError("BmaNullModel requires a BMA NullSpec")
        means, stds, present = _validate_arrays(
            self.means, self.stds, self.present, ndim=2
        )
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "stds", stds)
        object.__setattr__(self, "present", present)

    @classmethod
    def from_mapping(
        cls,
        values: BmaNullValues,
        spec: NullSpec,
    ) -> BmaNullModel:
        if not isinstance(values, Mapping):
            raise TypeError("BMA null values must be a mapping")
        if not values:
            raise ValueError("cannot create a BMA null model from empty values")
        pairs = [(int(m), int(k)) for m, k in values]
        if min(min(pair) for pair in pairs) < 1:
            raise ValueError("BMA null sizes must be positive")
        means = np.zeros(
            (max(m for m, _ in pairs) + 1, max(k for _, k in pairs) + 1),
            dtype=np.float64,
        )
        stds = np.zeros_like(means)
        present = np.zeros(means.shape, dtype=np.bool_)
        for key, value in values.items():
            m, k = (int(key[0]), int(key[1]))
            mean, standard_deviation = value
            means[m, k] = float(mean)
            stds[m, k] = float(standard_deviation)
            present[m, k] = True
        return cls(spec=spec, means=means, stds=stds, present=present)

    def to_mapping(self) -> dict[tuple[int, int], tuple[float, float]]:
        rows, columns = np.nonzero(self.present)
        return {
            (int(m), int(k)): (
                float(self.means[m, k]),
                float(self.stds[m, k]),
            )
            for m, k in zip(rows, columns, strict=True)
        }

    def standardize_matrix(
        self,
        true_scores,
        left_sizes,
        right_sizes,
        *,
        out=None,
    ):
        scores = np.asarray(true_scores)
        left = _validated_sizes(left_sizes, "left_sizes", allow_empty=True)
        right = _validated_sizes(right_sizes, "right_sizes", allow_empty=True)
        expected = (left.size, right.size)
        if scores.shape != expected:
            raise ValueError(
                f"true_scores shape {scores.shape} does not match {expected}"
            )
        if (
            left.size
            and (int(left.min()) < 0 or int(left.max()) >= self.means.shape[0])
        ) or (
            right.size
            and (int(right.min()) < 0 or int(right.max()) >= self.means.shape[1])
        ):
            raise KeyError("requested BMA size lies outside the null artifact")
        requested_present = self.present[left[:, None], right[None, :]]
        if not requested_present.all():
            row, column = np.argwhere(~requested_present)[0]
            raise KeyError((int(left[row]), int(right[column])))

        if out is None:
            out = np.empty(expected, dtype=np.float32)
        else:
            out = np.asarray(out)
            if out.shape != expected:
                raise ValueError("out shape does not match true_scores")
        for row, size in enumerate(left):
            means = self.means[int(size), right]
            stds = self.stds[int(size), right]
            out[row] = scores[row]
            out[row] -= means
            nonzero = stds != 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(out[row], stds, out=out[row], where=nonzero)
            out[row, ~nonzero] = 0.0
        return np.asarray(out, dtype=np.float32)

    def save(self, path, *, overwrite: bool = False):
        _save_model(
            path,
            "bma",
            self.spec,
            self.means,
            self.stds,
            self.present,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, path):
        spec, means, stds, present = _load_payload(path, "bma")
        return cls(spec=spec, means=means, stds=stds, present=present)


@dataclass(frozen=True, slots=True)
class RankedNullModel:
    """Dense, explicitly masked ranked-ES null calibration."""

    spec: NullSpec
    means: Float64Array
    stds: Float64Array
    present: BoolArray

    def __post_init__(self):
        if self.spec.kind != "ranked":
            raise ValueError("RankedNullModel requires a ranked NullSpec")
        means, stds, present = _validate_arrays(
            self.means, self.stds, self.present, ndim=1
        )
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "stds", stds)
        object.__setattr__(self, "present", present)

    @classmethod
    def from_mapping(
        cls,
        values: RankedNullValues,
        spec: NullSpec,
    ) -> RankedNullModel:
        if not isinstance(values, Mapping):
            raise TypeError("ranked null values must be a mapping")
        if not values:
            raise ValueError("cannot create a ranked null model from empty values")
        sizes = [int(size) for size in values]
        if min(sizes) < 1:
            raise ValueError("ranked null sizes must be positive")
        means = np.zeros(max(sizes) + 1, dtype=np.float64)
        stds = np.zeros_like(means)
        present = np.zeros(means.shape, dtype=np.bool_)
        for size, value in values.items():
            mean, standard_deviation = value
            means[int(size)] = float(mean)
            stds[int(size)] = float(standard_deviation)
            present[int(size)] = True
        return cls(spec=spec, means=means, stds=stds, present=present)

    def to_mapping(self) -> dict[int, tuple[float, float]]:
        return {
            int(size): (float(self.means[size]), float(self.stds[size]))
            for size in np.flatnonzero(self.present)
        }

    def standardize(self, true_scores, sizes, *, out=None):
        scores = np.asarray(true_scores)
        sizes = _validated_sizes(sizes, "sizes", allow_empty=True)
        if scores.ndim != 1 or sizes.ndim != 1 or scores.size != sizes.size:
            raise ValueError("true_scores and sizes must be same-length vectors")
        if sizes.size and (int(sizes.min()) < 0 or int(sizes.max()) >= self.means.size):
            raise KeyError("requested ranked size lies outside the null artifact")
        if not self.present[sizes].all():
            missing_position = int(np.flatnonzero(~self.present[sizes])[0])
            raise KeyError(int(sizes[missing_position]))
        if out is None:
            out = np.empty(scores.shape, dtype=np.float32)
        else:
            out = np.asarray(out)
            if out.shape != scores.shape:
                raise ValueError("out shape does not match true_scores")
        out[:] = scores
        out -= self.means[sizes]
        stds = self.stds[sizes]
        nonzero = stds != 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(out, stds, out=out, where=nonzero)
        out[~nonzero] = 0.0
        return np.asarray(out, dtype=np.float32)

    def save(self, path, *, overwrite: bool = False):
        _save_model(
            path,
            "ranked",
            self.spec,
            self.means,
            self.stds,
            self.present,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, path):
        spec, means, stds, present = _load_payload(path, "ranked")
        return cls(spec=spec, means=means, stds=stds, present=present)


def load_null_model(path):
    """Load either concrete null artifact after inspecting only its metadata."""
    metadata = artifacts.read_json(Path(path) / "metadata.json")
    kind = metadata.get("kind") if isinstance(metadata, Mapping) else None
    if kind == "bma":
        return BmaNullModel.load(path)
    if kind == "ranked":
        return RankedNullModel.load(path)
    raise ValueError(f"unknown ANDES null artifact kind {kind!r}")


@dataclass(frozen=True, slots=True)
class NullResolution:
    """One null model plus artifact lifecycle metadata."""

    model: BmaNullModel | RankedNullModel
    path: Path | None
    built: bool
    added_entries: int


@dataclass(frozen=True, slots=True)
class NullInspection:
    """Read-only lifecycle state for one content-addressed null artifact."""

    spec: NullSpec
    path: Path | None
    status: str
    exists: bool
    compatible: bool
    requested_entries: int
    missing_entries: int


def null_artifact_path(base_dir, spec: NullSpec) -> Path:
    """Return the canonical content-addressed path for one null specification."""
    return Path(base_dir) / f"{spec.kind}_{spec.fingerprint}.null"


def resolve_cache_root(path: str | Path | None = None) -> Path:
    """Resolve the shared cache root from an explicit path or the environment."""
    if path is not None and str(path):
        return Path(path).expanduser()
    configured = os.environ.get("ANDES_CACHE_ROOT", "")
    return Path(configured).expanduser() if configured else DEFAULT_CACHE_ROOT


def null_cache_dir(
    kind: str,
    cache_root: str | Path | None = None,
) -> Path:
    """Return the canonical directory for one null-artifact kind."""
    if kind not in {"bma", "ranked"}:
        raise ValueError(f"unsupported null cache kind {kind!r}")
    return resolve_cache_root(cache_root) / kind


def resolve_null_seed(seed: int | None) -> int:
    """Return a concrete seed without mutating NumPy's global RNG state."""
    if seed is None or seed < 0:
        state = np.random.SeedSequence().generate_state(1, dtype=np.uint64)
        return int(state[0])
    return int(seed)


def _validated_sizes(values, name, *, allow_empty=False):
    sizes = np.asarray(values)
    if sizes.ndim != 1 or sizes.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer sizes")
    if not allow_empty and sizes.size == 0:
        raise ValueError(f"{name} must contain at least one positive size")
    if sizes.size and int(sizes.min()) < 1:
        raise ValueError(f"{name} must contain positive sizes")
    if sizes.size and int(sizes.max()) > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds the supported int32 size range")
    return sizes.astype(np.int32, copy=False)


def _canonical_sizes(values, name):
    return np.unique(_validated_sizes(list(values), name))


def _bma_null_spec(
    embedding: EmbeddingSpace,
    population1: ArrayLike,
    population2: ArrayLike,
    *,
    iterations: int,
    seed: int,
) -> NullSpec:
    return NullSpec(
        kind="bma",
        iterations=int(iterations),
        seed=int(seed),
        sampling="prefix_coupled",
        ddof=1,
        embedding_hash=embedding.vector_hash,
        population_hashes=(
            artifacts.hash_array(population1),
            artifacts.hash_array(population2),
        ),
    )


def _ranked_null_spec(
    embedding: EmbeddingSpace,
    population: ArrayLike,
    ranked_embeddings: ArrayLike,
    *,
    iterations: int,
    seed: int,
) -> NullSpec:
    from .ranked import RANKED_ES_TIE_POLICY

    return NullSpec(
        kind="ranked",
        iterations=int(iterations),
        seed=int(seed),
        sampling="prefix_coupled",
        ddof=1,
        embedding_hash=embedding.vector_hash,
        population_hashes=(artifacts.hash_array(population),),
        ranked_hash=artifacts.hash_array(ranked_embeddings),
        tie_policy=RANKED_ES_TIE_POLICY,
    )


def _requested_artifact_path(path, base_dir, spec: NullSpec) -> Path | None:
    if path is not None and base_dir is not None:
        raise ValueError("supply either path or base_dir, not both")
    if path is not None:
        return Path(path)
    if base_dir is not None:
        return null_artifact_path(base_dir, spec)
    return None


def inspect_bma_null(
    embedding: EmbeddingSpace,
    population1: ArrayLike,
    population2: ArrayLike,
    row_sizes: Iterable[int],
    column_sizes: Iterable[int],
    *,
    path: str | Path | None = None,
    base_dir: str | Path | None = None,
    iterations: int = 1000,
    seed: int = 12345,
) -> NullInspection:
    """Inspect BMA null coverage without constructing or mutating an artifact."""
    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    population1 = np.asarray(population1, dtype=np.int32)
    population2 = np.asarray(population2, dtype=np.int32)
    rows = _canonical_sizes(row_sizes, "row_sizes")
    columns = _canonical_sizes(column_sizes, "column_sizes")
    seed = resolve_null_seed(seed)
    spec = _bma_null_spec(
        embedding,
        population1,
        population2,
        iterations=int(iterations),
        seed=seed,
    )
    artifact_path = _requested_artifact_path(path, base_dir, spec)
    requested = int(rows.size * columns.size)
    if artifact_path is None:
        return NullInspection(
            spec,
            None,
            "uncached",
            False,
            False,
            requested,
            requested,
        )
    if not artifact_path.exists():
        return NullInspection(
            spec,
            artifact_path,
            "build",
            False,
            False,
            requested,
            requested,
        )
    with artifacts.artifact_lock(artifact_path, shared=True, create=False):
        model = BmaNullModel.load(artifact_path)
    compatible = model.spec == spec
    if not compatible:
        return NullInspection(
            spec,
            artifact_path,
            "incompatible",
            True,
            False,
            requested,
            requested,
        )
    missing = sum(
        1
        for row in rows
        for column in columns
        if row >= model.present.shape[0]
        or column >= model.present.shape[1]
        or not model.present[row, column]
    )
    return NullInspection(
        spec,
        artifact_path,
        "reuse" if missing == 0 else "extend",
        True,
        True,
        requested,
        missing,
    )


def inspect_ranked_null(
    embedding: EmbeddingSpace,
    population: ArrayLike,
    sizes: Iterable[int],
    ranked_embeddings: ArrayLike,
    *,
    path: str | Path | None = None,
    base_dir: str | Path | None = None,
    iterations: int = 1000,
    seed: int = 12345,
) -> NullInspection:
    """Inspect ranked null coverage without constructing or mutating an artifact."""
    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    population = np.asarray(population, dtype=np.int32)
    requested_sizes = _canonical_sizes(sizes, "sizes")
    ranked_embeddings = np.asarray(ranked_embeddings, dtype=np.float32)
    seed = resolve_null_seed(seed)
    spec = _ranked_null_spec(
        embedding,
        population,
        ranked_embeddings,
        iterations=int(iterations),
        seed=seed,
    )
    artifact_path = _requested_artifact_path(path, base_dir, spec)
    requested = int(requested_sizes.size)
    if artifact_path is None:
        return NullInspection(
            spec,
            None,
            "uncached",
            False,
            False,
            requested,
            requested,
        )
    if not artifact_path.exists():
        return NullInspection(
            spec,
            artifact_path,
            "build",
            False,
            False,
            requested,
            requested,
        )
    with artifacts.artifact_lock(artifact_path, shared=True, create=False):
        model = RankedNullModel.load(artifact_path)
    compatible = model.spec == spec
    if not compatible:
        return NullInspection(
            spec,
            artifact_path,
            "incompatible",
            True,
            False,
            requested,
            requested,
        )
    missing = sum(
        1
        for size in requested_sizes
        if size >= model.present.size or not model.present[size]
    )
    return NullInspection(
        spec,
        artifact_path,
        "reuse" if missing == 0 else "extend",
        True,
        True,
        requested,
        missing,
    )


def resolve_bma_null(
    embedding: EmbeddingSpace,
    population1: ArrayLike,
    population2: ArrayLike,
    row_sizes: Iterable[int],
    column_sizes: Iterable[int],
    *,
    path: str | Path | None = None,
    base_dir: str | Path | None = None,
    iterations: int = 1000,
    seed: int = 12345,
    no_build: bool = False,
    rebuild: bool = False,
    blas_threads: int = 1,
) -> NullResolution:
    """Load, extend, or build one axis-aware BMA null artifact."""
    from .bma import build_prefix_null

    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    iterations = int(iterations)
    population1 = np.asarray(population1, dtype=np.int32)
    population2 = np.asarray(population2, dtype=np.int32)
    rows = _canonical_sizes(row_sizes, "row_sizes")
    columns = _canonical_sizes(column_sizes, "column_sizes")
    seed = resolve_null_seed(seed)
    spec = _bma_null_spec(
        embedding,
        population1,
        population2,
        iterations=iterations,
        seed=seed,
    )
    artifact_path = _requested_artifact_path(path, base_dir, spec)
    if no_build and (artifact_path is None or not artifact_path.exists()):
        raise FileNotFoundError(f"BMA null artifact does not exist: {artifact_path}")

    def resolve_at_path() -> NullResolution:
        existing = None
        if artifact_path is not None and artifact_path.exists() and not rebuild:
            existing = BmaNullModel.load(artifact_path)
            if existing.spec != spec:
                raise ValueError(
                    "BMA null artifact has incompatible scientific identity; "
                    "choose another path or pass --rebuild-cache"
                )

        pairs = {(int(m), int(k)) for m in rows for k in columns}
        missing = (
            sorted(pairs)
            if existing is None
            else sorted(
                (m, k)
                for m, k in pairs
                if m >= existing.present.shape[0]
                or k >= existing.present.shape[1]
                or not existing.present[m, k]
            )
        )
        if not missing:
            if existing is None:
                raise RuntimeError("BMA null resolution lost its loaded model")
            return NullResolution(existing, artifact_path, False, 0)
        if no_build:
            raise ValueError(f"BMA null artifact is missing {len(missing)} size pairs")

        with threadpool_limits(limits=int(blas_threads), user_api="blas"):
            values = build_prefix_null(
                embedding.vectors,
                population1,
                pairs,
                iterations=iterations,
                seed=seed,
                population2=population2,
                existing=None if existing is None else existing.to_mapping(),
            )
        model = BmaNullModel.from_mapping(values, spec)
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(artifact_path, overwrite=artifact_path.exists())
        return NullResolution(model, artifact_path, True, len(missing))

    lock = (
        artifacts.artifact_lock(
            artifact_path,
            shared=no_build,
            create=not no_build,
        )
        if artifact_path is not None
        else nullcontext()
    )
    with lock:
        return resolve_at_path()


def resolve_ranked_null(
    embedding,
    population,
    sizes,
    ranked_embeddings,
    *,
    runtime_plan,
    path=None,
    base_dir=None,
    iterations=1000,
    seed=12345,
    no_build=False,
    rebuild=False,
    worker_blas_threads=1,
) -> NullResolution:
    """Load, extend, or build one ranked prefix-coupled null artifact."""
    from .ranked import build_ranked_null, build_ranked_null_parallel

    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    iterations = int(iterations)
    population = np.asarray(population, dtype=np.int32)
    requested_sizes = _canonical_sizes(sizes, "sizes")
    ranked_embeddings = np.asarray(ranked_embeddings, dtype=np.float32)
    seed = resolve_null_seed(seed)
    spec = _ranked_null_spec(
        embedding,
        population,
        ranked_embeddings,
        iterations=iterations,
        seed=seed,
    )
    artifact_path = _requested_artifact_path(path, base_dir, spec)
    if no_build and (artifact_path is None or not artifact_path.exists()):
        raise FileNotFoundError(f"ranked null artifact does not exist: {artifact_path}")

    def resolve_at_path() -> NullResolution:
        existing = None
        if artifact_path is not None and artifact_path.exists() and not rebuild:
            existing = RankedNullModel.load(artifact_path)
            if existing.spec != spec:
                raise ValueError(
                    "ranked null artifact has incompatible scientific identity; "
                    "choose another path or pass --rebuild-cache"
                )

        missing = (
            requested_sizes.tolist()
            if existing is None
            else [
                int(size)
                for size in requested_sizes
                if size >= existing.present.size or not existing.present[size]
            ]
        )
        if not missing:
            if existing is None:
                raise RuntimeError("ranked null resolution lost its loaded model")
            return NullResolution(existing, artifact_path, False, 0)
        if no_build:
            raise ValueError(f"ranked null artifact is missing {len(missing)} sizes")

        existing_values = None if existing is None else existing.to_mapping()
        if (
            runtime_plan.workers > 1
            or runtime_plan.strategy == "precomputed_similarity"
        ):
            values = build_ranked_null_parallel(
                embedding.vectors,
                population,
                requested_sizes,
                ranked_embeddings,
                iterations=iterations,
                seed=seed,
                workers=runtime_plan.workers,
                blas_threads_per_worker=worker_blas_threads,
                worker_workspace_bytes=runtime_plan.workspace_bytes_per_worker,
                precompute_similarities=(
                    runtime_plan.strategy == "precomputed_similarity"
                ),
                existing=existing_values,
            )
        else:
            with threadpool_limits(
                limits=int(worker_blas_threads),
                user_api="blas",
            ):
                values = build_ranked_null(
                    embedding.vectors,
                    population,
                    requested_sizes,
                    ranked_embeddings,
                    iterations=iterations,
                    seed=seed,
                    worker_workspace_bytes=runtime_plan.workspace_bytes_per_worker,
                    existing=existing_values,
                )
        model = RankedNullModel.from_mapping(values, spec)
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(artifact_path, overwrite=artifact_path.exists())
        return NullResolution(model, artifact_path, True, len(missing))

    lock = (
        artifacts.artifact_lock(
            artifact_path,
            shared=no_build,
            create=not no_build,
        )
        if artifact_path is not None
        else nullcontext()
    )
    with lock:
        return resolve_at_path()
