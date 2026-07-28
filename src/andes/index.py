"""
Persistent exact ANDES database indexes and index command-line operations.

Builds a reusable index for one embedding plus one GMT database:

  B[g, t] = max_{x in term_t} sim(g, x)
  M[t, g] = 1 if gene g belongs to term t

Then one query gene set Q can be scored exactly against every indexed term:

  query_to_terms = B[Q, :].sum(axis=0)
  best_to_query = max_{q in Q} sim(g, q) for every gene g
  terms_to_query = M @ best_to_query
  true = (query_to_terms + terms_to_query) / (|Q| + term_sizes)

This is standard symmetric BMA. The feature is intentionally separate from
andes.py so the all-vs-all implementation remains the baseline path.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import data as ld
from . import bma as func
from . import artifacts
from . import application
from .nulls import BmaNullModel
from .provenance import RunProvenance, write_result_sidecar

INDEX_VERSION = 2

_INDEX_FILES = (
    "metadata.json",
    "genes.json",
    "terms.json",
    "embedding.npy",
    "bestmatch.npy",
    "sizes.npy",
    "background.npy",
    "membership.npz",
    "term_indices.npz",
)

_METADATA_KEYS = {
    "index_version",
    "n_genes",
    "embedding_dim",
    "n_terms",
    "embedding_hash",
    "gene_list_hash",
    "terms_hash",
    "sizes_hash",
    "term_indices_hash",
    "background_hash",
    "membership_hash",
    "bestmatch_hash",
}


def _hash_array(arr, digest_size=16):
    return artifacts.hash_array(arr, digest_size=digest_size)


def _hash_strings(items, digest_size=16):
    return artifacts.hash_strings(items, digest_size=digest_size)


def _hash_sparse_csr(matrix, digest_size=16):
    """Hash a CSR matrix including shape, dtype, and canonical sparse payload."""
    return artifacts.hash_sparse_csr(matrix, digest_size=digest_size)


def _invalid_index(index_dir, message):
    return ValueError(f"invalid ANDES index at {Path(index_dir)}: {message}")


def _database_fingerprint(index):
    offsets = np.empty(len(index.sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.asarray(index.sizes, dtype=np.int64), out=offsets[1:])
    return artifacts.combine_fingerprints(
        "andes_genesets_v1",
        {
            "background": index.metadata["background_hash"],
            "embedding": _embedding_fingerprint(index),
            "members": index.metadata["term_indices_hash"],
            "offsets": artifacts.hash_array(offsets),
            "terms": index.metadata["terms_hash"],
        },
    )


def _embedding_fingerprint(index):
    return artifacts.combine_fingerprints(
        "andes_embedding_v1",
        {
            "genes": index.metadata["gene_list_hash"],
            "vectors": index.metadata["embedding_hash"],
        },
    )


def _query_fingerprint(query_indices):
    return artifacts.combine_fingerprints(
        "andes_query_v1",
        {"indices": artifacts.hash_array(np.asarray(query_indices, dtype=np.int32))},
    )


def _null_spec_payload(cache):
    return None if cache is None else BmaNullModel.from_builder(cache).spec.to_dict()


def _require_float32_matrix(name, value):
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must have dtype float32, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr)


def _canonical_int_indices(name, value, n_genes, allow_empty=False):
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index array")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must contain at least one index")
    # Validate in the source integer width. Casting first could wrap a large
    # uint64/int64 value into an apparently valid int32 embedding row.
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= int(n_genes)):
        raise ValueError(f"{name} contains an index outside [0, {int(n_genes)})")
    return np.unique(arr.astype(np.int32, copy=False))


def _validate_build_inputs(E_unit, gene_list, terms, term_indices, background):
    """Validate and canonicalize the public index-builder inputs."""
    E_unit = _require_float32_matrix("E_unit", E_unit)
    n_genes = E_unit.shape[0]
    if n_genes == 0 or E_unit.shape[1] == 0:
        raise ValueError("E_unit must have at least one row and one column")
    norms = np.linalg.norm(E_unit, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-6):
        raise ValueError("E_unit rows must be unit-normalized")

    gene_list = list(gene_list)
    if len(gene_list) != n_genes:
        raise ValueError(
            "embedding rows do not match gene-list length: "
            f"{n_genes} != {len(gene_list)}"
        )
    if any(not isinstance(gene, str) or not gene for gene in gene_list):
        raise ValueError("gene_list must contain non-empty strings")
    if len(set(gene_list)) != len(gene_list):
        raise ValueError("gene_list contains duplicate names; gene order is ambiguous")

    terms = list(terms)
    if not terms:
        raise ValueError("terms must contain at least one term")
    if any(not isinstance(term, str) or not term for term in terms):
        raise ValueError("terms must contain non-empty strings")
    if len(set(terms)) != len(terms):
        raise ValueError("terms contains duplicate names")

    missing_terms = [term for term in terms if term not in term_indices]
    if missing_terms:
        raise ValueError(f"term_indices is missing term {missing_terms[0]!r}")
    canonical_terms = {
        term: _canonical_int_indices(
            f"term_indices[{term!r}]", term_indices[term], n_genes
        )
        for term in terms
    }
    background = _canonical_int_indices("background", background, n_genes)
    all_members = np.concatenate(list(canonical_terms.values()))
    missing_members = np.setdiff1d(
        all_members,
        background,
        assume_unique=False,
    )
    if missing_members.size:
        raise ValueError(
            f"background omits term member indices, including {int(missing_members[0])}"
        )
    return E_unit, gene_list, terms, canonical_terms, background


def _cache_payload(null_cache):
    if hasattr(null_cache, "cache"):
        return null_cache.cache
    if isinstance(null_cache, Mapping):
        return null_cache
    raise TypeError("null_cache must be a BmaNullBuilder instance or mapping")


def _validate_null_cache_axes(null_cache, index1, index2):
    """Validate cache orientation when cache metadata is available."""
    metadata = getattr(null_cache, "metadata", None)
    if not metadata:
        return
    if metadata.get("kind") != "andes_bma_null":
        raise ValueError("null cache is not an ANDES BMA cache")

    expected = {
        "embedding_hash": index1.metadata["embedding_hash"],
        "population1_hash": _hash_array(np.asarray(index1.background, dtype=np.int32)),
        "population2_hash": _hash_array(np.asarray(index2.background, dtype=np.int32)),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            axis = {
                "embedding_hash": "embedding",
                "population1_hash": "row/background-1",
                "population2_hash": "column/background-2",
            }[key]
            raise ValueError(f"null cache {axis} metadata is incompatible")


def load_embedding(emb_path, genelist_path):
    embedding = ld.load_embedding_space(emb_path, genelist_path)
    return embedding.vectors, list(embedding.genes)


def load_target_gmt(gmt_path, gene_list, min_size=10, max_size=300):
    node2index = {g: i for i, g in enumerate(gene_list)}
    raw = ld.load_gmt(gmt_path)
    indexed = ld.term2indexes(raw, node2index, upper=max_size, lower=min_size)
    indexed_np = func.preconvert_indices_to_arrays(indexed)
    terms = sorted(indexed_np)
    if not terms:
        raise ValueError("no terms passed size filters")
    background = np.asarray(
        func.get_background_indices(raw, set(gene_list), node2index), dtype=np.int32
    )
    return raw, terms, indexed_np, background


def _term_indices_to_npz_payload(terms, term_indices):
    return {
        f"arr_{i}": np.asarray(term_indices[t], dtype=np.int32)
        for i, t in enumerate(terms)
    }


def _build_andes_index_payload(
    E_unit,
    gene_list,
    terms,
    term_indices,
    background,
    out_dir,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Write one complete index payload into an existing private directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        E_unit,
        gene_list,
        terms,
        term_indices,
        background,
    ) = _validate_build_inputs(E_unit, gene_list, terms, term_indices, background)
    M = func.build_term_membership_matrix(terms, term_indices, E_unit.shape[0])
    M = M.tocsr()
    M.sort_indices()
    sizes = np.asarray([len(term_indices[t]) for t in terms], dtype=np.int32)

    np.save(out_dir / "embedding.npy", E_unit)
    np.save(out_dir / "sizes.npy", sizes)
    np.save(out_dir / "background.npy", background)
    sparse.save_npz(out_dir / "membership.npz", M)
    np.savez_compressed(
        out_dir / "term_indices.npz",
        **_term_indices_to_npz_payload(terms, term_indices),
    )
    with open(out_dir / "genes.json", "w") as fh:
        json.dump(list(gene_list), fh)
    with open(out_dir / "terms.json", "w") as fh:
        json.dump(terms, fh)

    packed = func.pack_term_indices(terms, term_indices)
    _, planned_chunk_workspace_mb = func.bestmatch_term_ranges(
        packed.sizes,
        E_unit.shape[0],
        E_unit.shape[1],
        max_workspace_mb,
    )
    bestmatch_path = out_dir / "bestmatch.npy"
    B = np.lib.format.open_memmap(
        bestmatch_path,
        mode="w+",
        dtype=np.float32,
        shape=(E_unit.shape[0], len(terms)),
    )
    next_start = 0
    peak_chunk_mb = 0.0
    for start, end, B_chunk in func.iter_gene_to_term_best_match_chunks(
        E_unit,
        E_unit,
        packed,
        max_workspace_mb=max_workspace_mb,
        show_progress=show_progress,
    ):
        start = int(start)
        end = int(end)
        B_chunk = np.asarray(B_chunk)
        if start != next_start or end <= start or end > len(terms):
            raise RuntimeError(
                "best-match chunk iterator yielded a non-contiguous term range "
                f"({start}, {end}) after column {next_start}"
            )
        expected_shape = (E_unit.shape[0], end - start)
        if B_chunk.shape != expected_shape or B_chunk.dtype != np.float32:
            raise RuntimeError(
                "best-match chunk iterator yielded "
                f"shape={B_chunk.shape}, dtype={B_chunk.dtype}; "
                f"expected shape={expected_shape}, dtype=float32"
            )
        B[:, start:end] = B_chunk
        peak_chunk_mb = max(peak_chunk_mb, B_chunk.nbytes / 1e6)
        next_start = end
    if next_start != len(terms):
        raise RuntimeError(
            f"best-match chunk iterator stopped at column {next_start} of {len(terms)}"
        )
    B.flush()

    metadata = {
        "index_version": INDEX_VERSION,
        "n_genes": int(E_unit.shape[0]),
        "embedding_dim": int(E_unit.shape[1]),
        "n_terms": len(terms),
        "embedding_hash": _hash_array(E_unit),
        "gene_list_hash": _hash_strings(gene_list),
        "terms_hash": _hash_strings(terms),
        "sizes_hash": _hash_array(sizes),
        "term_indices_hash": _hash_array(
            np.concatenate([term_indices[t] for t in terms]).astype(np.int32)
            if terms
            else np.empty(0, dtype=np.int32)
        ),
        "background_hash": _hash_array(np.sort(background)),
        "membership_hash": _hash_sparse_csr(M),
        "bestmatch_hash": _hash_array(B),
        "workspace_mb": float(planned_chunk_workspace_mb),
        "chunk_workspace_mb": float(planned_chunk_workspace_mb),
        "bestmatch_write_chunk_mb": float(peak_chunk_mb),
        "workspace_limit_mb": float(max_workspace_mb),
        "bestmatch_mb": float(B.nbytes / 1e6),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    B.flush()
    return metadata


def build_andes_index(
    E_unit,
    gene_list,
    terms,
    term_indices,
    background,
    out_dir,
    max_workspace_mb=1024,
    show_progress=False,
    overwrite=False,
):
    """Build, validate, and atomically publish an exact ANDES index."""
    final_dir = Path(out_dir)
    with artifacts.atomic_artifact_directory(
        final_dir, overwrite=bool(overwrite)
    ) as building_dir:
        metadata = _build_andes_index_payload(
            E_unit,
            gene_list,
            terms,
            term_indices,
            background,
            building_dir,
            max_workspace_mb=max_workspace_mb,
            show_progress=show_progress,
        )
        # Validate the private artifact with the same complete contract used by
        # consumers before making the final directory visible.
        load_andes_index(building_dir, mmap=True, verify_hashes=True)
    return metadata


@dataclass
class AndesIndex:
    index_dir: Path
    E_unit: np.ndarray
    gene_list: list
    gene_to_index: dict
    terms: list
    term_to_index: dict
    term_indices: dict
    sizes: np.ndarray
    background: np.ndarray
    membership: sparse.csr_matrix
    bestmatch: np.ndarray
    metadata: dict

    def ranked_bestmatch_artifact(self):
        """Return the persistent matrix with its complete scoring identity."""
        from .scoring import IndexedBestMatch

        return IndexedBestMatch(
            values=self.bestmatch,
            embedding_fingerprint=_embedding_fingerprint(self),
            database_fingerprint=_database_fingerprint(self),
        )

    def map_genes(self, genes):
        idx = []
        missing = []
        for gene in genes:
            if gene in self.gene_to_index:
                idx.append(self.gene_to_index[gene])
            else:
                missing.append(gene)
        if not idx:
            raise ValueError("query has no genes present in the indexed embedding")
        return np.asarray(sorted(set(idx)), dtype=np.int32), missing

    def _canonical_query_indices(self, query_idx):
        return _canonical_int_indices(
            "query_idx", query_idx, self.E_unit.shape[0], allow_empty=False
        )

    def validate_query_background(self, query_idx):
        """Raise if query genes cannot be normalized by the index null cache."""
        query_idx = self._canonical_query_indices(query_idx)
        missing = np.setdiff1d(query_idx, self.background, assume_unique=False)
        if missing.size:
            genes = [self.gene_list[int(i)] for i in missing[:5]]
            extra = "" if missing.size <= 5 else f" and {missing.size - 5} more"
            raise ValueError(
                "z-score queries must use genes from the indexed background; "
                f"outside-background genes: {', '.join(genes)}{extra}"
            )

    def assert_compatible_with(self, other):
        """Require the same normalized embedding and exact gene row order."""
        if not isinstance(other, AndesIndex):
            raise TypeError("other must be an AndesIndex")
        if self.gene_list != other.gene_list:
            raise ValueError(
                "index gene order mismatch; indexes must use identical genes "
                "in identical row order"
            )
        if self.metadata["gene_list_hash"] != other.metadata["gene_list_hash"]:
            raise ValueError("index gene-order hashes do not match")
        if self.E_unit.shape != other.E_unit.shape:
            raise ValueError(
                "index embedding shape mismatch: "
                f"{self.E_unit.shape} != {other.E_unit.shape}"
            )
        if self.metadata["embedding_hash"] != other.metadata["embedding_hash"]:
            raise ValueError(
                "index embedding mismatch; indexes must be built from the "
                "same normalized embedding"
            )

    def _normalize_scores(self, true_scores, row_sizes, null_cache, other=None):
        if null_cache is None:
            return None
        other = self if other is None else other
        _validate_null_cache_axes(null_cache, self, other)
        return func.zscore_matrix_from_cache(
            np.asarray(true_scores, dtype=np.float32),
            np.asarray(row_sizes, dtype=np.int32),
            other.sizes,
            _cache_payload(null_cache),
        )

    def score_query(self, query_idx, null_cache=None):
        query_idx = self._canonical_query_indices(query_idx)

        query_to_terms = np.asarray(
            self.bestmatch[query_idx, :].sum(axis=0, dtype=np.float32),
            dtype=np.float32,
        )
        sims = self.E_unit @ self.E_unit[query_idx].T
        best_to_query = sims.max(axis=1).astype(np.float32)
        terms_to_query = np.asarray(self.membership @ best_to_query, dtype=np.float32)

        true_scores = (query_to_terms + terms_to_query) / (
            float(query_idx.size) + self.sizes.astype(np.float32)
        )
        if null_cache is None:
            return true_scores.astype(np.float32), None

        self.validate_query_background(query_idx)
        zscores = self._normalize_scores(
            true_scores[None, :],
            np.asarray([query_idx.size], dtype=np.int32),
            null_cache,
        )[0]
        return true_scores.astype(np.float32), zscores

    def score_indexed_term(self, term, null_cache=None):
        """Score one indexed term using its already-persisted best-match column."""
        if isinstance(term, (int, np.integer)):
            term_pos = int(term)
            if term_pos < 0 or term_pos >= len(self.terms):
                raise IndexError(f"indexed term position {term_pos} is out of range")
            term_name = self.terms[term_pos]
        else:
            term_name = str(term)
            try:
                term_pos = self.term_to_index[term_name]
            except KeyError as exc:
                raise KeyError(
                    f"term {term_name!r} is not present in the index"
                ) from exc

        query_idx = self.term_indices[term_name]
        query_to_terms = np.asarray(
            self.bestmatch[query_idx, :].sum(axis=0, dtype=np.float32),
            dtype=np.float32,
        )
        best_to_query = np.asarray(self.bestmatch[:, term_pos], dtype=np.float32)
        terms_to_query = np.asarray(self.membership @ best_to_query, dtype=np.float32)
        true_scores = (query_to_terms + terms_to_query) / (
            float(query_idx.size) + self.sizes.astype(np.float32)
        )
        true_scores = np.asarray(true_scores, dtype=np.float32)
        if null_cache is None:
            return true_scores, None

        self.validate_query_background(query_idx)
        zscores = self._normalize_scores(
            true_scores[None, :],
            np.asarray([query_idx.size], dtype=np.int32),
            null_cache,
        )[0]
        return true_scores, zscores

    def score_queries(
        self,
        query_sets,
        null_cache=None,
        max_workspace_mb=1024,
    ):
        """Score many query sets, sharing forward aggregation and reverse GEMMs."""
        if isinstance(query_sets, Mapping):
            query_sets = list(query_sets.values())
        else:
            query_sets = list(query_sets)
        if not query_sets:
            empty = np.empty((0, len(self.terms)), dtype=np.float32)
            return empty, None

        queries = [self._canonical_query_indices(query) for query in query_sets]
        query_sizes = np.asarray([query.size for query in queries], dtype=np.int32)
        row_idx = np.repeat(
            np.arange(len(queries), dtype=np.int32), query_sizes.astype(np.int64)
        )
        col_idx = np.concatenate(queries).astype(np.int32, copy=False)
        query_membership = sparse.csr_matrix(
            (
                np.ones(col_idx.size, dtype=np.float32),
                (row_idx, col_idx),
            ),
            shape=(len(queries), self.E_unit.shape[0]),
            dtype=np.float32,
        )
        query_to_terms = np.asarray(query_membership @ self.bestmatch, dtype=np.float32)
        terms_to_queries = np.empty((len(queries), len(self.terms)), dtype=np.float32)

        budget_bytes = (
            float(max_workspace_mb) * 1e6
            if max_workspace_mb and max_workspace_mb > 0
            else float("inf")
        )
        ranges = []
        start = 0
        packed_genes = 0
        for end, size in enumerate(query_sizes, start=1):
            candidate_genes = packed_genes + int(size)
            candidate_queries = end - start
            candidate_bytes = (
                self.E_unit.shape[0]
                * (candidate_genes + candidate_queries)
                * np.dtype(np.float32).itemsize
            )
            if end - start > 1 and candidate_bytes > budget_bytes:
                ranges.append((start, end - 1))
                start = end - 1
                packed_genes = int(size)
            else:
                packed_genes = candidate_genes
        ranges.append((start, len(queries)))

        for start, end in ranges:
            local_queries = queries[start:end]
            local_sizes = query_sizes[start:end]
            flat = np.concatenate(local_queries).astype(np.int32, copy=False)
            offsets = np.empty(len(local_queries), dtype=np.int64)
            offsets[0] = 0
            if len(local_queries) > 1:
                np.cumsum(local_sizes[:-1], out=offsets[1:])

            sims = self.E_unit @ self.E_unit[flat].T
            best_to_queries = np.maximum.reduceat(sims, offsets, axis=1)
            terms_to_queries[start:end, :] = np.asarray(
                self.membership @ best_to_queries, dtype=np.float32
            ).T

        true_scores = (query_to_terms + terms_to_queries) / (
            query_sizes[:, None].astype(np.float32)
            + self.sizes[None, :].astype(np.float32)
        )
        true_scores = np.asarray(true_scores, dtype=np.float32)
        if null_cache is None:
            return true_scores, None

        for query in queries:
            self.validate_query_background(query)
        zscores = self._normalize_scores(true_scores, query_sizes, null_cache)
        return true_scores, zscores

    def compare(self, other, null_cache=None):
        """Compute exact all-vs-all BMA scores against a compatible index."""
        self.assert_compatible_with(other)
        directed_12 = np.asarray(self.membership @ other.bestmatch, dtype=np.float32)
        same_term_axis = (
            self.terms == other.terms
            and np.array_equal(self.sizes, other.sizes)
            and all(
                np.array_equal(self.term_indices[term], other.term_indices[term])
                for term in self.terms
            )
        )
        if same_term_axis:
            directed_21 = directed_12
        else:
            directed_21 = np.asarray(
                other.membership @ self.bestmatch, dtype=np.float32
            )
        denominator = self.sizes[:, None].astype(np.float32) + other.sizes[
            None, :
        ].astype(np.float32)
        true_scores = np.asarray(
            (directed_12 + directed_21.T) / denominator,
            dtype=np.float32,
        )
        if null_cache is None:
            return true_scores, None

        zscores = self._normalize_scores(
            true_scores, self.sizes, null_cache, other=other
        )
        return true_scores, zscores


def score_index_to_index(index1, index2, null_cache=None):
    """Functional wrapper for exact index-to-index scoring."""
    return index1.compare(index2, null_cache=null_cache)


def _load_json(index_dir, filename):
    try:
        with open(index_dir / filename, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid_index(index_dir, f"cannot read {filename}: {exc}") from exc


def _validate_loaded_index(
    index_dir,
    metadata,
    gene_list,
    terms,
    E_unit,
    bestmatch,
    sizes,
    background,
    membership,
    term_indices,
    verify_hashes,
):
    if not isinstance(metadata, dict):
        raise _invalid_index(index_dir, "metadata.json must contain an object")
    version = metadata.get("index_version")
    if version != INDEX_VERSION:
        raise _invalid_index(
            index_dir,
            f"unsupported index_version {version!r}; expected {INDEX_VERSION}",
        )
    missing_metadata = sorted(_METADATA_KEYS - set(metadata))
    if missing_metadata:
        raise _invalid_index(
            index_dir,
            "metadata.json is missing keys: " + ", ".join(missing_metadata),
        )

    if not isinstance(gene_list, list) or any(
        not isinstance(gene, str) or not gene for gene in gene_list
    ):
        raise _invalid_index(index_dir, "genes.json must contain non-empty strings")
    if len(set(gene_list)) != len(gene_list):
        raise _invalid_index(index_dir, "genes.json contains duplicate gene names")
    if not isinstance(terms, list) or any(
        not isinstance(term, str) or not term for term in terms
    ):
        raise _invalid_index(index_dir, "terms.json must contain non-empty strings")
    if len(set(terms)) != len(terms):
        raise _invalid_index(index_dir, "terms.json contains duplicate term names")

    n_genes = len(gene_list)
    n_terms = len(terms)
    embedding_dim = int(metadata.get("embedding_dim", -1))
    expected_shapes = {
        "embedding.npy": (n_genes, embedding_dim),
        "bestmatch.npy": (n_genes, n_terms),
        "sizes.npy": (n_terms,),
    }
    arrays = {
        "embedding.npy": E_unit,
        "bestmatch.npy": bestmatch,
        "sizes.npy": sizes,
    }
    for filename, expected_shape in expected_shapes.items():
        if arrays[filename].shape != expected_shape:
            raise _invalid_index(
                index_dir,
                f"{filename} has shape {arrays[filename].shape}; "
                f"expected {expected_shape}",
            )
    if E_unit.dtype != np.float32:
        raise _invalid_index(
            index_dir, f"embedding.npy has dtype {E_unit.dtype}; expected float32"
        )
    if bestmatch.dtype != np.float32:
        raise _invalid_index(
            index_dir, f"bestmatch.npy has dtype {bestmatch.dtype}; expected float32"
        )
    if sizes.dtype != np.int32:
        raise _invalid_index(
            index_dir, f"sizes.npy has dtype {sizes.dtype}; expected int32"
        )
    if background.ndim != 1 or background.dtype != np.int32:
        raise _invalid_index(
            index_dir,
            "background.npy must be a one-dimensional int32 array",
        )
    if membership.shape != (n_terms, n_genes):
        raise _invalid_index(
            index_dir,
            f"membership.npz has shape {membership.shape}; "
            f"expected {(n_terms, n_genes)}",
        )
    if membership.dtype != np.float32:
        raise _invalid_index(
            index_dir,
            f"membership.npz has dtype {membership.dtype}; expected float32",
        )

    scalar_metadata = {
        "n_genes": n_genes,
        "embedding_dim": E_unit.shape[1],
        "n_terms": n_terms,
    }
    for key, expected in scalar_metadata.items():
        if metadata.get(key) != expected:
            raise _invalid_index(
                index_dir,
                f"metadata {key}={metadata.get(key)!r}; expected {expected}",
            )

    if background.size == 0:
        raise _invalid_index(index_dir, "background.npy is empty")
    if np.any(background < 0) or np.any(background >= n_genes):
        raise _invalid_index(index_dir, "background.npy contains out-of-range indices")
    if not np.array_equal(background, np.unique(background)):
        raise _invalid_index(
            index_dir, "background.npy must contain sorted unique indices"
        )

    membership = membership.tocsr(copy=False)
    membership.sort_indices()
    if membership.nnz and not np.all(membership.data == 1.0):
        raise _invalid_index(index_dir, "membership.npz must contain binary values")
    for i, term in enumerate(terms):
        idx = term_indices.get(term)
        if idx is None:
            raise _invalid_index(index_dir, f"term_indices.npz is missing {term!r}")
        if idx.ndim != 1 or idx.dtype != np.int32:
            raise _invalid_index(
                index_dir,
                f"term indices for {term!r} must be a one-dimensional int32 array",
            )
        if idx.size == 0:
            raise _invalid_index(index_dir, f"term {term!r} has no gene indices")
        if int(idx[0]) < 0 or int(idx[-1]) >= n_genes:
            raise _invalid_index(
                index_dir, f"term {term!r} contains an out-of-range gene index"
            )
        if not np.array_equal(idx, np.unique(idx)):
            raise _invalid_index(
                index_dir, f"term {term!r} indices must be sorted and unique"
            )
        if np.setdiff1d(idx, background, assume_unique=True).size:
            raise _invalid_index(
                index_dir,
                f"background.npy omits members of term {term!r}",
            )
        if int(sizes[i]) != idx.size:
            raise _invalid_index(
                index_dir,
                f"sizes.npy disagrees with term {term!r}: "
                f"{int(sizes[i])} != {idx.size}",
            )
        row_start = int(membership.indptr[i])
        row_end = int(membership.indptr[i + 1])
        if not np.array_equal(membership.indices[row_start:row_end], idx):
            raise _invalid_index(
                index_dir, f"membership.npz disagrees with term {term!r}"
            )

    if verify_hashes:
        concatenated = np.concatenate([term_indices[term] for term in terms])
        expected_hashes = {
            "gene_list_hash": _hash_strings(gene_list),
            "terms_hash": _hash_strings(terms),
            "sizes_hash": _hash_array(sizes),
            "term_indices_hash": _hash_array(concatenated.astype(np.int32, copy=False)),
            "background_hash": _hash_array(background),
            "membership_hash": _hash_sparse_csr(membership),
        }
        if verify_hashes is True:
            if not np.isfinite(E_unit).all():
                raise _invalid_index(
                    index_dir,
                    "embedding.npy contains non-finite values",
                )
            if not np.allclose(
                np.linalg.norm(E_unit, axis=1),
                1.0,
                rtol=2e-5,
                atol=2e-6,
            ):
                raise _invalid_index(
                    index_dir,
                    "embedding.npy rows are not unit-normalized",
                )
            if not np.isfinite(bestmatch).all():
                raise _invalid_index(
                    index_dir,
                    "bestmatch.npy contains non-finite values",
                )
            expected_hashes.update(
                {
                    "embedding_hash": _hash_array(E_unit),
                    "bestmatch_hash": _hash_array(bestmatch),
                }
            )
        for key, expected in expected_hashes.items():
            if metadata.get(key) != expected:
                raise _invalid_index(
                    index_dir,
                    f"{key} mismatch; the index payload may be corrupt or stale",
                )
    return membership


def load_andes_index(index_dir, mmap=False, verify_hashes="metadata"):
    """Load a structurally validated index.

    The default ``"metadata"`` mode verifies all small identity payloads while
    leaving the large embedding and best-match arrays memory-mapped. Pass
    ``True`` for a full corruption audit or ``False`` for shape/dtype checks
    only. Index construction always performs the full audit before publishing.
    """
    if verify_hashes not in {False, True, "metadata"}:
        raise ValueError("verify_hashes must be False, True, or 'metadata'")
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        raise _invalid_index(index_dir, "index directory does not exist")
    missing = [name for name in _INDEX_FILES if not (index_dir / name).is_file()]
    if missing:
        raise _invalid_index(index_dir, "missing required files: " + ", ".join(missing))

    metadata = _load_json(index_dir, "metadata.json")
    gene_list = _load_json(index_dir, "genes.json")
    terms = _load_json(index_dir, "terms.json")
    mmap_mode = "r" if mmap else None
    try:
        E_unit = np.load(
            index_dir / "embedding.npy",
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        bestmatch = np.load(
            index_dir / "bestmatch.npy",
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        sizes = np.load(index_dir / "sizes.npy", allow_pickle=False)
        background = np.load(index_dir / "background.npy", allow_pickle=False)
        membership = sparse.load_npz(index_dir / "membership.npz").tocsr()
        with np.load(index_dir / "term_indices.npz", allow_pickle=False) as packed:
            expected_keys = {f"arr_{i}" for i in range(len(terms))}
            actual_keys = set(packed.files)
            if actual_keys != expected_keys:
                raise _invalid_index(
                    index_dir,
                    "term_indices.npz keys do not match terms.json",
                )
            term_indices = {
                term: np.asarray(packed[f"arr_{i}"]) for i, term in enumerate(terms)
            }
    except ValueError as exc:
        if str(exc).startswith("invalid ANDES index"):
            raise
        raise _invalid_index(index_dir, f"cannot load array payload: {exc}") from exc
    except OSError as exc:
        raise _invalid_index(index_dir, f"cannot load array payload: {exc}") from exc

    membership = _validate_loaded_index(
        index_dir,
        metadata,
        gene_list,
        terms,
        E_unit,
        bestmatch,
        sizes,
        background,
        membership,
        term_indices,
        verify_hashes,
    )
    gene_to_index = {gene: i for i, gene in enumerate(gene_list)}
    term_to_index = {term: i for i, term in enumerate(terms)}

    return AndesIndex(
        index_dir=index_dir,
        E_unit=E_unit,
        gene_list=gene_list,
        gene_to_index=gene_to_index,
        terms=terms,
        term_to_index=term_to_index,
        term_indices=term_indices,
        sizes=sizes,
        background=background,
        membership=membership,
        bestmatch=bestmatch,
        metadata=metadata,
    )


def load_query_genes(path):
    genes = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split("\t")
            if len(tokens) >= 3:
                genes.extend(tok for tok in tokens[2:] if tok)
            else:
                genes.extend(line.split())
    return genes


def load_query_sets(path):
    """Load one query set per GMT row, preserving file order."""
    names = []
    gene_sets = []
    seen = set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            tokens = line.split("\t")
            if len(tokens) < 3:
                raise ValueError(
                    f"{path}:{lineno}: batch queries must use GMT format "
                    "(name, description, genes...)"
                )
            name = tokens[0].strip()
            genes = [gene for gene in tokens[2:] if gene]
            if not name:
                raise ValueError(f"{path}:{lineno}: query name is empty")
            if name in seen:
                raise ValueError(f"{path}:{lineno}: duplicate query name {name!r}")
            if not genes:
                raise ValueError(f"{path}:{lineno}: query {name!r} has no genes")
            seen.add(name)
            names.append(name)
            gene_sets.append(genes)
    if not names:
        raise ValueError(f"{path}: no query sets found")
    return names, gene_sets


def load_or_build_query_cache(
    index,
    query_size,
    cache_path,
    ite=1000,
    seed=12345,
    null_mode="prefix",
    no_build=False,
    verbose=True,
):
    """Load/build a BMA null cache for one indexed query size.

    This uses the index background on both null axes, which is the reusable
    server-side setting. Axis-specific backgrounds should use the compare or
    null application commands.
    """
    return _load_or_build_index_cache(
        index,
        index,
        np.asarray([query_size], dtype=np.int32),
        index.sizes,
        cache_path,
        ite=ite,
        seed=seed,
        null_mode=null_mode,
        no_build=no_build,
        verbose=verbose,
        label=f"indexed-query (query_size={int(query_size)})",
    )


def load_or_build_batch_query_cache(
    index,
    query_sizes,
    cache_path,
    ite=1000,
    seed=12345,
    null_mode="prefix",
    no_build=False,
    verbose=True,
):
    """Load/build one cache covering all distinct batch-query sizes."""
    return _load_or_build_index_cache(
        index,
        index,
        query_sizes,
        index.sizes,
        cache_path,
        ite=ite,
        seed=seed,
        null_mode=null_mode,
        no_build=no_build,
        verbose=verbose,
        label="indexed batch-query",
    )


def load_or_build_comparison_cache(
    index1,
    index2,
    cache_path,
    ite=1000,
    seed=12345,
    null_mode="prefix",
    no_build=False,
    verbose=True,
):
    """Load/build an axis-aware null cache for an index comparison."""
    index1.assert_compatible_with(index2)
    return _load_or_build_index_cache(
        index1,
        index2,
        index1.sizes,
        index2.sizes,
        cache_path,
        ite=ite,
        seed=seed,
        null_mode=null_mode,
        no_build=no_build,
        verbose=verbose,
        label="index comparison",
    )


def _load_or_build_index_cache(
    index1,
    index2,
    row_sizes,
    column_sizes,
    cache_path,
    ite,
    seed,
    null_mode,
    no_build,
    verbose,
    label,
):
    cache_path = Path(cache_path)
    cache = func.BmaNullBuilder()
    seed = func.BmaNullBuilder.resolve_seed(seed)
    null_sampling = "prefix_coupled" if null_mode == "prefix" else "per_size_pair"
    expected = func.BmaNullBuilder.build_metadata(
        index1.E_unit,
        index1.background,
        index2.background,
        ite,
        seed,
        null_sampling=null_sampling,
    )
    size_pairs = {
        (int(m), int(k))
        for m in np.unique(np.asarray(row_sizes, dtype=np.int32))
        for k in np.unique(np.asarray(column_sizes, dtype=np.int32))
    }

    if cache_path.exists():
        if cache_path.is_dir():
            cache.load_artifact(cache_path)
        else:
            cache.load_artifact(cache_path)

    metadata_ok, reason = cache.metadata_matches(expected)
    missing = (
        [pair for pair in size_pairs if pair not in cache.cache]
        if metadata_ok
        else sorted(size_pairs)
    )
    if metadata_ok and not missing:
        _validate_null_cache_axes(cache, index1, index2)
        return cache
    if no_build:
        raise RuntimeError(
            f"cache is missing {len(missing)} size pairs or has incompatible "
            f"metadata ({reason})"
        )

    if verbose:
        print(
            f"Building {label} null cache: {len(size_pairs)} size pairs, "
            f"ite={ite}, null_mode={null_mode}"
        )
    if null_mode == "prefix":
        cache.precompute_prefix(
            index1.E_unit,
            index1.background,
            size_pairs,
            ite=ite,
            seed=seed,
            population_idx2=index2.background,
            verbose=verbose,
        )
    else:
        cache.precompute(
            index1.E_unit,
            index1.background,
            size_pairs,
            ite=ite,
            seed=seed,
            population_idx2=index2.background,
            verbose=verbose,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_artifact(cache_path, overwrite=cache_path.exists())
    _validate_null_cache_axes(cache, index1, index2)
    return cache


def write_query_results(out_path, index, true_scores, zscores=None, top_k=0):
    out_path = Path(out_path)
    order = np.arange(len(index.terms))
    if zscores is not None:
        sort_values = zscores
    else:
        sort_values = true_scores
    if top_k and top_k > 0 and top_k < len(order):
        order = np.argpartition(sort_values, -top_k)[-top_k:]
    order = order[np.argsort(sort_values[order])[::-1]]

    data = {
        "term": [index.terms[i] for i in order],
        "size": index.sizes[order],
        "true_score": true_scores[order],
    }
    if zscores is not None:
        data["z_score"] = zscores[order]
    df = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    application.write_dataframe_atomic(df, out_path, index=False)
    return df


def write_batch_query_results(
    out_path,
    index,
    query_names,
    true_scores,
    zscores=None,
    top_k=0,
):
    """Write long-form results for a batch of query sets."""
    true_scores = np.asarray(true_scores)
    if true_scores.shape != (len(query_names), len(index.terms)):
        raise ValueError("true_scores shape does not match queries x indexed terms")
    if zscores is not None and np.asarray(zscores).shape != true_scores.shape:
        raise ValueError("zscores shape does not match true_scores")

    frames = []
    for i, name in enumerate(query_names):
        sort_values = true_scores[i] if zscores is None else np.asarray(zscores)[i]
        order = np.arange(len(index.terms))
        if top_k and 0 < top_k < len(order):
            order = np.argpartition(sort_values, -top_k)[-top_k:]
        order = order[np.argsort(sort_values[order], kind="stable")[::-1]]
        data = {
            "query": [name] * len(order),
            "term": [index.terms[j] for j in order],
            "size": index.sizes[order],
            "true_score": true_scores[i, order],
        }
        if zscores is not None:
            data["z_score"] = np.asarray(zscores)[i, order]
        frames.append(pd.DataFrame(data))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    application.write_dataframe_atomic(df, out_path, index=False)
    return df


def write_comparison_results(
    out_path,
    index1,
    index2,
    true_scores,
    zscores=None,
):
    """Write an index comparison as labeled CSV or binary NPY plus labels."""
    out_path = Path(out_path)
    values = np.asarray(true_scores if zscores is None else zscores, dtype=np.float32)
    expected_shape = (len(index1.terms), len(index2.terms))
    if values.shape != expected_shape:
        raise ValueError(
            f"comparison matrix has shape {values.shape}; expected {expected_shape}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".npy":
        artifacts.save_npy_atomic(out_path, values)
        artifacts.write_json_atomic(out_path.with_suffix(".rows.json"), index1.terms)
        artifacts.write_json_atomic(out_path.with_suffix(".columns.json"), index2.terms)
        return values

    df = pd.DataFrame(values, index=index1.terms, columns=index2.terms)
    application.write_dataframe_atomic(df, out_path, index_label="term")
    return df


def cmd_build(args):
    E_unit, gene_list = load_embedding(args.emb, args.genelist)
    _, terms, term_indices, background = load_target_gmt(
        args.geneset, gene_list, min_size=args.min_size, max_size=args.max_size
    )
    print(f"Building ANDES index: {len(gene_list)} genes x {len(terms)} terms")
    metadata = build_andes_index(
        E_unit,
        gene_list,
        terms,
        term_indices,
        background,
        args.out,
        max_workspace_mb=args.query_memory_mb,
        show_progress=args.verbose,
        overwrite=args.overwrite,
    )
    print(f"Wrote index to {args.out}")
    print(f"Best-match matrix: {metadata['bestmatch_mb']:.1f} MB")
    print(f"Peak temporary build chunk: ~{metadata['chunk_workspace_mb']:.1f} MB")


def cmd_query(args):
    index = load_andes_index(args.index, mmap=not args.no_mmap)
    if args.term is not None:
        try:
            term_pos = index.term_to_index[args.term]
        except KeyError as exc:
            raise ValueError(f"term {args.term!r} is not present in the index") from exc
        query_idx = index.term_indices[args.term]
        missing = []
        print(
            f"Indexed-term query: {args.term} "
            f"({len(query_idx)} genes, column {term_pos})"
        )
    else:
        genes = load_query_genes(args.genes)
        query_idx, missing = index.map_genes(genes)
        print(
            f"Query genes: {len(query_idx)} matched"
            + (f", {len(missing)} missing" if missing else "")
        )

    cache = None
    if not args.no_zscore:
        index.validate_query_background(query_idx)
        cache_path = args.cache or str(Path(args.index) / "bma_query_null.null")
        cache = load_or_build_query_cache(
            index,
            len(query_idx),
            cache_path,
            ite=args.ite,
            seed=args.seed,
            null_mode=args.null_mode,
            no_build=args.no_cache_build,
            verbose=args.verbose,
        )

    if args.term is not None:
        true_scores, zscores = index.score_indexed_term(args.term, cache)
    else:
        true_scores, zscores = index.score_query(query_idx, cache)
    df = write_query_results(
        args.out,
        index,
        true_scores,
        zscores=zscores,
        top_k=args.top_k,
    )
    provenance = RunProvenance(
        method="andes_index_query",
        score_engine=("indexed_term" if args.term is not None else "indexed_query"),
        score_kind="z_score" if zscores is not None else "true_score",
        numeric_dtype="float32",
        accumulator_dtype="float32",
        embedding_fingerprint=index.metadata["embedding_hash"],
        left_database_fingerprint=_query_fingerprint(query_idx),
        right_database_fingerprint=_database_fingerprint(index),
        null_spec=_null_spec_payload(cache),
        extra={
            "query_size": int(len(query_idx)),
            "top_k": int(args.top_k),
            "missing_genes": int(len(missing)),
        },
    )
    write_result_sidecar(args.out, provenance)
    print(f"Wrote {len(df)} results to {args.out}")


def cmd_batch_query(args):
    index = load_andes_index(args.index, mmap=not args.no_mmap)
    query_names, gene_sets = load_query_sets(args.queries)
    query_indices = []
    missing_total = 0
    for name, genes in zip(query_names, gene_sets):
        try:
            idx, missing = index.map_genes(genes)
        except ValueError as exc:
            raise ValueError(f"query {name!r}: {exc}") from exc
        query_indices.append(idx)
        missing_total += len(missing)
    print(
        f"Batch queries: {len(query_names)} sets, "
        f"{sum(len(idx) for idx in query_indices)} matched memberships"
        + (f", {missing_total} missing genes" if missing_total else "")
    )

    cache = None
    if not args.no_zscore:
        for idx in query_indices:
            index.validate_query_background(idx)
        cache_path = args.cache or str(Path(args.index) / "bma_query_null.null")
        cache = load_or_build_batch_query_cache(
            index,
            [len(idx) for idx in query_indices],
            cache_path,
            ite=args.ite,
            seed=args.seed,
            null_mode=args.null_mode,
            no_build=args.no_cache_build,
            verbose=args.verbose,
        )

    true_scores, zscores = index.score_queries(
        query_indices,
        null_cache=cache,
        max_workspace_mb=args.query_memory_mb,
    )
    df = write_batch_query_results(
        args.out,
        index,
        query_names,
        true_scores,
        zscores=zscores,
        top_k=args.top_k,
    )
    batch_fingerprint = artifacts.combine_fingerprints(
        "andes_batch_query_v1",
        {
            str(i): artifacts.hash_array(indices)
            for i, indices in enumerate(query_indices)
        },
    )
    provenance = RunProvenance(
        method="andes_index_batch_query",
        score_engine="indexed_batch",
        score_kind="z_score" if zscores is not None else "true_score",
        numeric_dtype="float32",
        accumulator_dtype="float32",
        embedding_fingerprint=index.metadata["embedding_hash"],
        left_database_fingerprint=batch_fingerprint,
        right_database_fingerprint=_database_fingerprint(index),
        null_spec=_null_spec_payload(cache),
        extra={
            "queries": len(query_names),
            "top_k": int(args.top_k),
            "missing_genes": int(missing_total),
        },
    )
    write_result_sidecar(args.out, provenance)
    print(f"Wrote {len(df)} results to {args.out}")


def cmd_compare(args):
    index1 = load_andes_index(args.index1, mmap=not args.no_mmap)
    index2 = load_andes_index(args.index2, mmap=not args.no_mmap)
    index1.assert_compatible_with(index2)
    print(f"Comparing indexes: {len(index1.terms)} x {len(index2.terms)} terms")

    cache = None
    if not args.no_zscore:
        cache_path = args.cache or str(Path(args.out).with_suffix(".null"))
        cache = load_or_build_comparison_cache(
            index1,
            index2,
            cache_path,
            ite=args.ite,
            seed=args.seed,
            null_mode=args.null_mode,
            no_build=args.no_cache_build,
            verbose=args.verbose,
        )
    true_scores, zscores = index1.compare(index2, null_cache=cache)
    write_comparison_results(
        args.out,
        index1,
        index2,
        true_scores,
        zscores=zscores,
    )
    provenance = RunProvenance(
        method="andes_index_compare",
        score_engine="index_to_index",
        score_kind="z_score" if zscores is not None else "true_score",
        numeric_dtype="float32",
        accumulator_dtype="float32",
        embedding_fingerprint=index1.metadata["embedding_hash"],
        left_database_fingerprint=_database_fingerprint(index1),
        right_database_fingerprint=_database_fingerprint(index2),
        symmetric_reuse=(
            index1.terms == index2.terms
            and index1.metadata["term_indices_hash"]
            == index2.metadata["term_indices_hash"]
        ),
        null_spec=_null_spec_payload(cache),
        extra={
            "rows": len(index1.terms),
            "columns": len(index2.terms),
        },
    )
    write_result_sidecar(args.out, provenance)
    kind = "z-score" if zscores is not None else "true-score"
    print(f"Wrote {kind} matrix to {args.out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build/query persistent ANDES indexes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="build a reusable target database index")
    build.add_argument("--emb", required=True)
    build.add_argument("--genelist", required=True)
    build.add_argument("--geneset", required=True, help="target GMT database")
    build.add_argument("--out", required=True, help="output index directory")
    build.add_argument("--min", dest="min_size", type=int, default=10)
    build.add_argument("--max", dest="max_size", type=int, default=300)
    build.add_argument(
        "--query-memory-mb",
        type=float,
        default=1024.0,
        help="target cap for temporary best-match construction chunks",
    )
    build.add_argument("--verbose", action="store_true")
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing index directory",
    )
    build.set_defaults(func=cmd_build)

    query = sub.add_parser(
        "query", help="score one query gene set or indexed term against an index"
    )
    query.add_argument("--index", required=True, help="index directory from build")
    query_source = query.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--genes", help="query genes file or one-line GMT")
    query_source.add_argument(
        "--term", help="indexed term name (uses the persisted best-match column)"
    )
    query.add_argument("--out", required=True)
    query.add_argument("--top-k", type=int, default=0, help="0 writes all terms")
    query.add_argument("--cache", default="", help="query null cache path")
    query.add_argument("--ite", type=int, default=1000)
    query.add_argument("--seed", type=int, default=12345)
    query.add_argument("--null-mode", choices=["prefix", "pairwise"], default="prefix")
    query.add_argument("--no-cache-build", action="store_true")
    query.add_argument("--no-zscore", action="store_true")
    query.add_argument("--no-mmap", action="store_true")
    query.add_argument("--verbose", action="store_true")
    query.set_defaults(func=cmd_query)

    batch = sub.add_parser(
        "batch-query", help="score a GMT file of query sets against an index"
    )
    batch.add_argument("--index", required=True, help="index directory from build")
    batch.add_argument("--queries", required=True, help="query sets in GMT format")
    batch.add_argument("--out", required=True)
    batch.add_argument("--top-k", type=int, default=0, help="per-query; 0 writes all")
    batch.add_argument("--cache", default="", help="query null cache path")
    batch.add_argument("--ite", type=int, default=1000)
    batch.add_argument("--seed", type=int, default=12345)
    batch.add_argument("--null-mode", choices=["prefix", "pairwise"], default="prefix")
    batch.add_argument(
        "--query-memory-mb",
        type=float,
        default=1024.0,
        help="target cap for temporary batched-query similarity chunks",
    )
    batch.add_argument("--no-cache-build", action="store_true")
    batch.add_argument("--no-zscore", action="store_true")
    batch.add_argument("--no-mmap", action="store_true")
    batch.add_argument("--verbose", action="store_true")
    batch.set_defaults(func=cmd_batch_query)

    compare = sub.add_parser(
        "compare", help="score every term in one index against every term in another"
    )
    compare.add_argument("--index1", required=True, help="row index directory")
    compare.add_argument("--index2", required=True, help="column index directory")
    compare.add_argument("--out", required=True, help="CSV or .npy score matrix")
    compare.add_argument("--cache", default="", help="axis-aware null cache path")
    compare.add_argument("--ite", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=12345)
    compare.add_argument(
        "--null-mode", choices=["prefix", "pairwise"], default="prefix"
    )
    compare.add_argument("--no-cache-build", action="store_true")
    compare.add_argument("--no-zscore", action="store_true")
    compare.add_argument("--no-mmap", action="store_true")
    compare.add_argument("--verbose", action="store_true")
    compare.set_defaults(func=cmd_compare)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
