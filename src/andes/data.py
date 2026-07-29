"""
Canonical embedding and gene-set data models plus validated input parsing.

GMT format: one gene set per line, tab-separated.
  col 0 : term ID
  col 1 : term description (skipped)
  col 2+: gene IDs

Importing the data models has no process-wide side effects.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import artifacts

FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]
Int64Array: TypeAlias = NDArray[np.int64]
TermPosition: TypeAlias = int | np.integer[Any]
_ScalarT = TypeVar("_ScalarT", bound=np.generic)


def _readonly_array(
    values: ArrayLike,
    dtype: type[_ScalarT],
) -> NDArray[_ScalarT]:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _term_position(position: object, term_count: int) -> int:
    """Validate a positional term-axis index without coercing identifiers."""
    if isinstance(position, (bool, np.bool_)) or not isinstance(
        position, (Integral, np.integer)
    ):
        raise TypeError("gene-set position must be an integer")
    resolved = int(position)
    if resolved < 0 or resolved >= term_count:
        raise IndexError(f"gene-set position {resolved} is out of range")
    return resolved


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Canonical normalized embedding and its exact gene-row identity."""

    vectors: FloatArray
    genes: tuple[str, ...]
    gene_to_index: Mapping[str, int]
    vector_hash: str
    gene_order_hash: str
    fingerprint: str

    @classmethod
    def from_arrays(
        cls,
        vectors: ArrayLike,
        genes: Iterable[object],
        *,
        normalize: bool = True,
    ) -> EmbeddingSpace:
        values = np.asarray(vectors)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("embedding vectors must have shape (genes, dimensions)")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("embedding vectors must be numeric")
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("embedding vectors contain a non-finite value")

        gene_names = tuple(str(gene) for gene in genes)
        if len(gene_names) != values.shape[0]:
            raise ValueError(
                "embedding gene count does not match the number of vector rows"
            )
        if any(not gene for gene in gene_names):
            raise ValueError("embedding gene identifiers must be non-empty")
        if len(set(gene_names)) != len(gene_names):
            raise ValueError("embedding gene identifiers must be unique")

        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("embedding contains a zero-length vector")
        if normalize:
            values = values / norms
        elif not np.allclose(
            norms[:, 0],
            1.0,
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ValueError("normalize=False requires unit-normalized embedding rows")

        values = _readonly_array(values, np.float32)
        gene_order_hash = artifacts.hash_strings(gene_names)
        vector_hash = artifacts.hash_array(values)
        components = {"genes": gene_order_hash, "vectors": vector_hash}
        return cls(
            vectors=values,
            genes=gene_names,
            gene_to_index=MappingProxyType(
                {gene: i for i, gene in enumerate(gene_names)}
            ),
            vector_hash=vector_hash,
            gene_order_hash=gene_order_hash,
            fingerprint=artifacts.combine_fingerprints(
                "andes_embedding_v1", components
            ),
        )


@dataclass(frozen=True, slots=True)
class PackedTermAxis:
    """Term-aligned zero-copy views used by numerical scoring kernels."""

    terms: tuple[str, ...]
    members: IntArray
    offsets: Int64Array
    sizes: IntArray

    def members_at(self, position: TermPosition) -> IntArray:
        """Return members for one positional term axis entry."""
        position = _term_position(position, len(self.terms))
        start = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        return self.members[start:end]


@dataclass(frozen=True, slots=True)
class GeneSetDatabase:
    """Canonical packed gene-set database aligned to one embedding."""

    n_genes: int
    embedding_fingerprint: str
    terms: tuple[str, ...]
    members: IntArray
    offsets: Int64Array
    sizes: IntArray
    background: IntArray
    background_policy: str
    term_to_index: Mapping[str, int]
    background_hash: str
    fingerprint: str

    @classmethod
    def from_index_mapping(
        cls,
        term_to_members,
        *,
        n_genes: int,
        embedding_fingerprint: str,
        terms=None,
        background=None,
        background_policy=None,
        min_size: int = 1,
        max_size: int | None = None,
    ):
        n_genes = int(n_genes)
        min_size = int(min_size)
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        embedding_fingerprint = str(embedding_fingerprint)
        if not embedding_fingerprint:
            raise ValueError("embedding_fingerprint must not be empty")
        if min_size < 1:
            raise ValueError("min_size must be positive")
        if max_size is None:
            max_size = n_genes
        max_size = int(max_size)
        if max_size < min_size:
            raise ValueError("max_size must be greater than or equal to min_size")

        ordered_terms = tuple(term_to_members.keys()) if terms is None else tuple(terms)
        if any(not isinstance(term, str) or not term for term in ordered_terms):
            raise ValueError("gene-set term identifiers must be non-empty strings")
        if len(set(ordered_terms)) != len(ordered_terms):
            raise ValueError("gene-set term identifiers must be unique")

        kept_terms = []
        arrays = []
        for term in ordered_terms:
            if term not in term_to_members:
                raise KeyError(f"missing membership for term {term!r}")
            values = np.asarray(term_to_members[term])
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise TypeError(
                    f"term {term!r} members must be a one-dimensional integer array"
                )
            canonical = np.unique(values.astype(np.int64, copy=False))
            if canonical.size and (
                int(canonical[0]) < 0 or int(canonical[-1]) >= n_genes
            ):
                raise IndexError(
                    f"term {term!r} contains an out-of-range embedding index"
                )
            if min_size <= canonical.size <= max_size:
                kept_terms.append(term)
                arrays.append(canonical.astype(np.int32, copy=False))

        if not kept_terms:
            raise ValueError("no gene sets remain after canonicalization and filtering")

        sizes = np.asarray([array.size for array in arrays], dtype=np.int32)
        offsets = np.empty(len(arrays) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])
        members = np.concatenate(arrays).astype(np.int32, copy=False)

        if background is None:
            if background_policy != "retained_members":
                raise ValueError(
                    "background_policy='retained_members' is required when "
                    "background is inferred"
                )
            background_values = np.unique(members)
            resolved_background_policy = "retained_members"
        else:
            if not isinstance(background_policy, str) or not background_policy:
                raise ValueError(
                    "background_policy must describe an explicitly supplied background"
                )
            if background_policy == "retained_members":
                raise ValueError(
                    "background_policy='retained_members' cannot accompany an "
                    "explicit background"
                )
            raw_background = np.asarray(background)
            if raw_background.ndim != 1 or raw_background.dtype.kind not in "iu":
                raise TypeError("background must be a one-dimensional integer array")
            background_values = np.unique(raw_background.astype(np.int64, copy=False))
            if background_values.size and (
                int(background_values[0]) < 0 or int(background_values[-1]) >= n_genes
            ):
                raise IndexError("background contains an out-of-range embedding index")
            background_values = background_values.astype(np.int32, copy=False)
            resolved_background_policy = background_policy
        if background_values.size == 0:
            raise ValueError("gene-set background must not be empty")
        missing_from_background = np.setdiff1d(
            members,
            background_values,
            assume_unique=False,
        )
        if missing_from_background.size:
            raise ValueError(
                "gene-set background omits retained member indices, including "
                f"{int(missing_from_background[0])}"
            )

        term_names = tuple(kept_terms)
        members = _readonly_array(members, np.int32)
        offsets = _readonly_array(offsets, np.int64)
        sizes = _readonly_array(sizes, np.int32)
        background_values = _readonly_array(background_values, np.int32)
        background_hash = artifacts.hash_array(background_values)
        components = {
            "background": background_hash,
            "embedding": embedding_fingerprint,
            "members": artifacts.hash_array(members),
            "offsets": artifacts.hash_array(offsets),
            "terms": artifacts.hash_strings(term_names),
        }
        return cls(
            n_genes=n_genes,
            embedding_fingerprint=embedding_fingerprint,
            terms=term_names,
            members=members,
            offsets=offsets,
            sizes=sizes,
            background=background_values,
            background_policy=resolved_background_policy,
            term_to_index=MappingProxyType(
                {term: i for i, term in enumerate(term_names)}
            ),
            background_hash=background_hash,
            fingerprint=artifacts.combine_fingerprints("andes_genesets_v1", components),
        )

    def members_at(self, position: TermPosition) -> IntArray:
        """Return read-only members for one positional term axis entry."""
        position = _term_position(position, len(self.terms))
        start = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        return self.members[start:end]

    def members_for(self, term: str) -> IntArray:
        """Return read-only members for one term identifier."""
        if not isinstance(term, str):
            raise TypeError("term must be a string identifier")
        try:
            position = self.term_to_index[term]
        except KeyError as exc:
            raise KeyError(f"unknown gene-set term {term!r}") from exc
        return self.members_at(position)

    @property
    def packed_axis(self) -> PackedTermAxis:
        """Return the canonical packed arrays without copying or rehashing."""
        return PackedTermAxis(
            terms=self.terms,
            members=self.members,
            offsets=self.offsets,
            sizes=self.sizes,
        )

    def select_terms(self, terms) -> GeneSetDatabase:
        """Return a canonical subset while preserving the original background."""
        requested = tuple(str(term) for term in terms)
        if not requested:
            raise ValueError("term selection must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("term selection must not contain duplicates")
        missing = [term for term in requested if term not in self.term_to_index]
        if missing:
            raise KeyError(f"unknown gene-set term {missing[0]!r}")
        return GeneSetDatabase.from_index_mapping(
            {term: self.members_for(term) for term in requested},
            n_genes=self.n_genes,
            embedding_fingerprint=self.embedding_fingerprint,
            terms=requested,
            background=self.background,
            background_policy="preserved",
        )


def _iter_gmt_rows(file):
    """Yield strictly validated GMT rows with source line numbers."""
    path = str(file)
    seen = set()
    with open(file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                raise ValueError(f"{path}:{line_number}: blank GMT row")
            tokens = [token.strip() for token in line.split("\t")]
            if len(tokens) < 3:
                raise ValueError(
                    f"{path}:{line_number}: GMT row requires at least "
                    "term, description, and one gene"
                )
            term = tokens[0]
            if not term:
                raise ValueError(f"{path}:{line_number}: empty GMT term identifier")
            if term in seen:
                raise ValueError(f"{path}:{line_number}: duplicate GMT term {term!r}")
            genes = tokens[2:]
            while genes and not genes[-1]:
                genes.pop()
            if any(not gene for gene in genes):
                raise ValueError(f"{path}:{line_number}: empty GMT gene identifier")
            seen.add(term)
            yield term, tokens[1], genes


def load_gmt(file):
    """Parse a GMT file into a mapping from term IDs to gene ID lists.

    The description field (column 1) is discarded. Gene IDs are returned as
    raw strings; callers apply node2index mapping via term2indexes.
    """
    return {term: genes for term, _, genes in _iter_gmt_rows(file)}


def term2indexes(go_dict, node2index, upper=300, lower=5):
    """Map gene-set gene IDs to embedding indices and filter by size.

    Genes not present in node2index (i.e., absent from the embedding) are
    dropped. Duplicate genes (including distinct IDs mapped to the same row)
    are removed before applying the inclusive size bounds. Terms with fewer
    than `lower` or more than `upper` unique surviving genes are excluded.

    Parameters
    ----------
    go_dict : dict {str: list of str}
        Raw gene sets from load_gmt.
    node2index : dict {str: int}
        Maps gene IDs to embedding rows. Must return -1 (or raise
        KeyError caught by .get) for unknown genes.
    upper, lower : int
        Inclusive size bounds after filtering.

    Returns
    -------
    dict {str: numpy.ndarray}
        Filtered gene sets as sorted, C-contiguous int32 index arrays.
    """
    ret = {}
    for key, genes in go_dict.items():
        unique_indices = {
            int(index) for gene in genes if (index := node2index.get(gene, -1)) != -1
        }
        if lower <= len(unique_indices) <= upper:
            ret[key] = np.asarray(sorted(unique_indices), dtype=np.int32)
    return ret


def load_embedding_space(
    embedding_path: str | Path,
    gene_list_path: str | Path,
) -> EmbeddingSpace:
    """Load CSV/NPY embeddings and return one validated normalized domain object."""
    embedding_path = str(embedding_path)
    if embedding_path.lower().endswith(".npy"):
        vectors = np.load(embedding_path, allow_pickle=False)
    else:
        vectors = np.loadtxt(embedding_path, delimiter=",", dtype=np.float32)
    with open(gene_list_path, encoding="utf-8") as handle:
        genes = [line.strip() for line in handle]
    return EmbeddingSpace.from_arrays(vectors, genes, normalize=True)


def load_ranked_indices(path, gene_to_index) -> IntArray:
    """Load one unique ranked gene universe aligned to an embedding.

    The first tab-separated field is the gene identifier. Additional fields
    are ignored. Genes absent from the embedding are dropped, while duplicate
    mapped genes are rejected because they make ranked traversal ambiguous.
    """
    ranked = []
    first_seen_line = {}
    with open(path, encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t", strict=True)
        for line_number, row in enumerate(rows, start=1):
            if not row or not row[0].strip():
                raise ValueError(f"{path}:{line_number}: empty ranked gene")
            gene = row[0].strip()
            index = gene_to_index.get(gene)
            if index is None:
                continue
            index = int(index)
            if index in first_seen_line:
                raise ValueError(
                    f"{path}:{line_number}: duplicate ranked gene {gene!r}; "
                    f"embedding row first appeared on line {first_seen_line[index]}"
                )
            first_seen_line[index] = line_number
            ranked.append(index)
    if not ranked:
        raise ValueError("ranked list has no genes in the embedding")
    return np.asarray(ranked, dtype=np.int32)


def load_gene_set_database(
    gmt_path: str | Path,
    embedding: EmbeddingSpace,
    *,
    min_size: int = 10,
    max_size: int = 300,
    sort_terms: bool = False,
) -> GeneSetDatabase:
    """Load and align a GMT database to an embedding in one validated step."""
    return load_gene_set_databases(
        [gmt_path],
        embedding,
        min_size=min_size,
        max_size=max_size,
        sort_terms=sort_terms,
    )


def load_gene_set_databases(
    gmt_paths: Sequence[str | Path],
    embedding: EmbeddingSpace,
    *,
    min_size: int = 10,
    max_size: int = 300,
    sort_terms: bool = False,
) -> GeneSetDatabase:
    """Load several GMT files as one canonical, validated database."""
    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    paths = tuple(gmt_paths)
    if not paths:
        raise ValueError("at least one GMT path is required")
    raw = {}
    for path in paths:
        loaded = load_gmt(path)
        duplicate_terms = set(raw).intersection(loaded)
        if duplicate_terms:
            duplicate = min(duplicate_terms)
            raise ValueError(f"duplicate term across GMT files: {duplicate!r}")
        raw.update(loaded)
    indexed = term2indexes(
        raw,
        embedding.gene_to_index,
        upper=max_size,
        lower=min_size,
    )
    if not indexed:
        raise ValueError("no terms from the supplied GMT files passed the size filters")
    terms = tuple(sorted(indexed) if sort_terms else indexed)
    background = np.unique(
        np.fromiter(
            (
                embedding.gene_to_index[gene]
                for genes in raw.values()
                for gene in genes
                if gene in embedding.gene_to_index
            ),
            dtype=np.int32,
        )
    )
    return GeneSetDatabase.from_index_mapping(
        indexed,
        n_genes=len(embedding.genes),
        embedding_fingerprint=embedding.fingerprint,
        terms=terms,
        background=background,
        background_policy="all_mapped_genes",
        min_size=1,
    )
