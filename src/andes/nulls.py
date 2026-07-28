"""Typed null-calibration models and safe on-disk artifacts.

The numerical builders still live beside their optimized kernels during the
migration.  These models own the stable query and persistence contracts:

* BMA calibration is a two-dimensional lookup keyed by ``(left_size, right_size)``.
* Ranked calibration is a one-dimensional lookup keyed by gene-set size.
* Missing entries are explicit through a boolean presence array.
* Artifacts are JSON metadata plus non-pickled NumPy arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from . import artifacts


NULL_ARTIFACT_VERSION = 1
_ARRAY_FILES = ("means.npy", "stds.npy", "present.npy")


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

    def __post_init__(self):
        if self.kind not in {"bma", "ranked"}:
            raise ValueError(f"unsupported null kind {self.kind!r}")
        if int(self.iterations) < 1:
            raise ValueError("null iterations must be positive")
        if int(self.seed) < 0:
            raise ValueError("null seed must be resolved before artifact creation")
        if int(self.ddof) < 0:
            raise ValueError("null ddof must be non-negative")
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

    def to_dict(self) -> dict:
        payload = asdict(self)
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
        return artifacts.combine_fingerprints(
            "andes_null_spec_v1",
            components,
        )

    @classmethod
    def from_dict(cls, payload):
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
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError("null spec is missing: " + ", ".join(missing))
        return cls(
            kind=str(payload["kind"]),
            iterations=int(payload["iterations"]),
            seed=int(payload["seed"]),
            sampling=str(payload["sampling"]),
            ddof=int(payload["ddof"]),
            embedding_hash=str(payload["embedding_hash"]),
            population_hashes=tuple(
                str(value) for value in payload["population_hashes"]
            ),
            ranked_hash=(
                None
                if payload.get("ranked_hash") is None
                else str(payload["ranked_hash"])
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
            f"unsupported null artifact version {metadata.get('artifact_version')!r}"
        )
    if metadata.get("kind") != expected_kind:
        raise ValueError(
            f"expected {expected_kind} null, found {metadata.get('kind')!r}"
        )

    means = np.load(root / "means.npy", allow_pickle=False)
    stds = np.load(root / "stds.npy", allow_pickle=False)
    present = np.load(root / "present.npy", allow_pickle=False)
    expected_shape = tuple(int(value) for value in metadata.get("shape", []))
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
    means: np.ndarray
    stds: np.ndarray
    present: np.ndarray

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
    def from_mapping(cls, cache, spec: NullSpec):
        values = cache.cache if hasattr(cache, "cache") else cache
        if not values:
            raise ValueError("cannot create a BMA null model from an empty cache")
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

    @classmethod
    def from_builder(cls, cache):
        """Freeze a populated numerical builder into a typed null model."""
        metadata = dict(getattr(cache, "metadata", {}))
        spec = NullSpec(
            kind="bma",
            iterations=int(metadata["ite"]),
            seed=int(metadata["seed"]),
            sampling=str(metadata.get("null_sampling", "per_size_pair")),
            ddof=int(metadata.get("std_ddof", 1)),
            embedding_hash=str(metadata["embedding_hash"]),
            population_hashes=(
                str(metadata["population1_hash"]),
                str(metadata["population2_hash"]),
            ),
        )
        return cls.from_mapping(cache, spec)

    def to_mapping(self) -> dict[tuple[int, int], tuple[float, float]]:
        rows, columns = np.nonzero(self.present)
        return {
            (int(m), int(k)): (
                float(self.means[m, k]),
                float(self.stds[m, k]),
            )
            for m, k in zip(rows, columns)
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
        left = np.asarray(left_sizes, dtype=np.int32)
        right = np.asarray(right_sizes, dtype=np.int32)
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
    means: np.ndarray
    stds: np.ndarray
    present: np.ndarray

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
    def from_mapping(cls, cache, spec: NullSpec):
        values = cache.cache if hasattr(cache, "cache") else cache
        if not values:
            raise ValueError("cannot create a ranked null model from an empty cache")
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

    @classmethod
    def from_builder(cls, cache):
        """Freeze a populated numerical builder into a typed null model."""
        metadata = dict(getattr(cache, "metadata", {}))
        spec = NullSpec(
            kind="ranked",
            iterations=int(metadata["ite"]),
            seed=int(metadata["seed"]),
            sampling="prefix_coupled",
            ddof=int(metadata.get("std_ddof", 1)),
            embedding_hash=str(metadata["embedding_hash"]),
            population_hashes=(str(metadata["population_hash"]),),
            ranked_hash=str(metadata["ranked_emb_hash"]),
        )
        return cls.from_mapping(cache, spec)

    def to_mapping(self) -> dict[int, tuple[float, float]]:
        return {
            int(size): (float(self.means[size]), float(self.stds[size]))
            for size in np.flatnonzero(self.present)
        }

    def standardize(self, true_scores, sizes, *, out=None):
        scores = np.asarray(true_scores)
        sizes = np.asarray(sizes, dtype=np.int32)
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
