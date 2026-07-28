"""
Canonical embedding and gene-set data models plus validated input parsing.

GMT format: one gene set per line, tab-separated.
  col 0 : term ID
  col 1 : term description (skipped)
  col 2+: gene IDs

This module is side-effect-free and does not import numerical scoring kernels.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import artifacts


def _readonly_array(values, dtype):
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Canonical normalized embedding and its exact gene-row identity."""

    vectors: np.ndarray
    genes: tuple[str, ...]
    gene_to_index: MappingProxyType
    vector_hash: str
    gene_order_hash: str
    fingerprint: str

    @classmethod
    def from_arrays(cls, vectors, genes, *, normalize: bool = True):
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
class GeneSetDatabase:
    """Canonical packed gene-set database aligned to one embedding."""

    embedding_fingerprint: str
    terms: tuple[str, ...]
    members: np.ndarray
    offsets: np.ndarray
    sizes: np.ndarray
    background: np.ndarray
    term_to_index: MappingProxyType
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
            background_values = np.unique(members)
        else:
            raw_background = np.asarray(background)
            if raw_background.ndim != 1 or raw_background.dtype.kind not in "iu":
                raise TypeError("background must be a one-dimensional integer array")
            background_values = np.unique(raw_background.astype(np.int64, copy=False))
            if background_values.size and (
                int(background_values[0]) < 0 or int(background_values[-1]) >= n_genes
            ):
                raise IndexError("background contains an out-of-range embedding index")
            background_values = background_values.astype(np.int32, copy=False)
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
            embedding_fingerprint=embedding_fingerprint,
            terms=term_names,
            members=members,
            offsets=offsets,
            sizes=sizes,
            background=background_values,
            term_to_index=MappingProxyType(
                {term: i for i, term in enumerate(term_names)}
            ),
            background_hash=background_hash,
            fingerprint=artifacts.combine_fingerprints("andes_genesets_v1", components),
        )

    def indices(self, term_or_position) -> np.ndarray:
        """Return the read-only canonical members for one term."""
        if isinstance(term_or_position, str):
            try:
                position = self.term_to_index[term_or_position]
            except KeyError as exc:
                raise KeyError(f"unknown gene-set term {term_or_position!r}") from exc
        else:
            position = int(term_or_position)
        if position < 0 or position >= len(self.terms):
            raise IndexError(f"gene-set position {position} is out of range")
        start = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        return self.members[start:end]

    def as_index_mapping(self) -> dict[str, np.ndarray]:
        """Return term-aligned array views for compatibility with old kernels."""
        return {
            term: self.indices(position) for position, term in enumerate(self.terms)
        }


def _iter_gmt_rows(file):
    """Yield strictly validated GMT rows with source line numbers."""
    path = str(file)
    seen = set()
    with open(file, "r", encoding="utf-8") as handle:
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
            if any(not gene for gene in genes):
                raise ValueError(f"{path}:{line_number}: empty GMT gene identifier")
            seen.add(term)
            yield term, tokens[1], genes


def load_gmt(file):
    """Parse a GMT file and return a dict mapping term ID → list of gene IDs.

    The description field (column 1) is discarded.  Gene IDs are returned as
    raw strings; callers apply node2index mapping via term2indexes.
    """
    return {term: genes for term, _, genes in _iter_gmt_rows(file)}


def term2name(file_name):
    """Parse a GMT file and return a dict mapping term ID → description string (column 1)."""
    return {term: description for term, description, _ in _iter_gmt_rows(file_name)}


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
        Gene ID → embedding row index mapping.  Must return -1 (or raise
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


def load_embedding_space(embedding_path, gene_list_path) -> EmbeddingSpace:
    """Load CSV/NPY embeddings and return one validated normalized domain object."""
    embedding_path = str(embedding_path)
    if embedding_path.lower().endswith(".npy"):
        vectors = np.load(embedding_path, allow_pickle=False)
    else:
        vectors = np.loadtxt(embedding_path, delimiter=",", dtype=np.float32)
    with open(gene_list_path, "r", encoding="utf-8") as handle:
        genes = [line.strip() for line in handle]
    return EmbeddingSpace.from_arrays(vectors, genes, normalize=True)


def load_gene_set_database(
    gmt_path,
    embedding: EmbeddingSpace,
    *,
    min_size=10,
    max_size=300,
    sort_terms=False,
) -> GeneSetDatabase:
    """Load and align a GMT database to an embedding in one validated step."""
    if not isinstance(embedding, EmbeddingSpace):
        raise TypeError("embedding must be an EmbeddingSpace")
    raw = load_gmt(gmt_path)
    indexed = term2indexes(
        raw,
        embedding.gene_to_index,
        upper=max_size,
        lower=min_size,
    )
    if not indexed:
        raise ValueError(f"no terms from {gmt_path} passed the size filters")
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
        min_size=1,
    )
