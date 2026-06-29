"""
andes_index.py — persistent exact ANDES database indexes

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
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import sparse

import load_data as ld
import func_optimized as func

INDEX_VERSION = 1


def _hash_array(arr, digest_size=16):
    arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _hash_strings(items, digest_size=16):
    h = hashlib.blake2b(digest_size=digest_size)
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def load_embedding(emb_path, genelist_path):
    raw = np.loadtxt(emb_path, delimiter=",", dtype=np.float32)
    with open(genelist_path, "r") as fh:
        gene_list = [line.strip() for line in fh]
    if len(gene_list) != raw.shape[0]:
        raise ValueError("embedding rows do not match gene-list length")
    E_unit = np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32)
    return E_unit, gene_list


def load_target_gmt(gmt_path, gene_list, min_size=10, max_size=300):
    node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(gene_list)})
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
    return {f"arr_{i}": np.asarray(term_indices[t], dtype=np.int32) for i, t in enumerate(terms)}


def build_andes_index(
    E_unit,
    gene_list,
    terms,
    term_indices,
    background,
    out_dir,
    max_workspace_mb=1024,
    show_progress=False,
):
    """Build and persist an exact one-database ANDES index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    terms = list(terms)
    term_indices = {
        t: np.asarray(term_indices[t], dtype=np.int32) for t in terms
    }
    background = np.asarray(background, dtype=np.int32)
    E_unit = np.ascontiguousarray(E_unit, dtype=np.float32)

    blocks = func.precompute_term_embedding_blocks(E_unit, term_indices)
    B, workspace_mb = func.gene_to_term_best_match_matrix(
        E_unit,
        terms,
        blocks,
        max_workspace_mb=max_workspace_mb,
        show_progress=show_progress,
    )
    M = func.build_term_membership_matrix(terms, term_indices, E_unit.shape[0])
    sizes = np.asarray([len(term_indices[t]) for t in terms], dtype=np.int32)

    np.save(out_dir / "embedding.npy", E_unit)
    np.save(out_dir / "bestmatch.npy", B)
    np.save(out_dir / "sizes.npy", sizes)
    np.save(out_dir / "background.npy", background)
    sparse.save_npz(out_dir / "membership.npz", M)
    np.savez_compressed(
        out_dir / "term_indices.npz", **_term_indices_to_npz_payload(terms, term_indices)
    )
    with open(out_dir / "genes.json", "w") as fh:
        json.dump(list(gene_list), fh)
    with open(out_dir / "terms.json", "w") as fh:
        json.dump(terms, fh)

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
        "workspace_mb": float(workspace_mb),
        "bestmatch_mb": float(B.nbytes / 1e6),
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return metadata


@dataclass
class AndesIndex:
    index_dir: Path
    E_unit: np.ndarray
    gene_list: list
    gene_to_index: dict
    terms: list
    term_indices: dict
    sizes: np.ndarray
    background: np.ndarray
    membership: sparse.csr_matrix
    bestmatch: np.ndarray
    metadata: dict

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

    def validate_query_background(self, query_idx):
        """Raise if query genes cannot be normalized by the index null cache."""
        query_idx = np.asarray(query_idx, dtype=np.int32)
        missing = np.setdiff1d(query_idx, self.background, assume_unique=False)
        if missing.size:
            genes = [self.gene_list[int(i)] for i in missing[:5]]
            extra = "" if missing.size <= 5 else f" and {missing.size - 5} more"
            raise ValueError(
                "z-score queries must use genes from the indexed background; "
                f"outside-background genes: {', '.join(genes)}{extra}"
            )

    def score_query(self, query_idx, null_cache=None):
        query_idx = np.asarray(query_idx, dtype=np.int32)
        if query_idx.size == 0:
            raise ValueError("query_idx must contain at least one gene")

        query_to_terms = np.asarray(
            self.bestmatch[query_idx, :].sum(axis=0), dtype=np.float32
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

        zscores = func.zscore_matrix_from_cache(
            true_scores[None, :],
            np.asarray([query_idx.size], dtype=np.int32),
            self.sizes,
            null_cache.cache,
        )[0]
        return true_scores.astype(np.float32), zscores


def load_andes_index(index_dir, mmap=False):
    index_dir = Path(index_dir)
    with open(index_dir / "metadata.json") as fh:
        metadata = json.load(fh)
    with open(index_dir / "genes.json") as fh:
        gene_list = json.load(fh)
    with open(index_dir / "terms.json") as fh:
        terms = json.load(fh)

    mmap_mode = "r" if mmap else None
    E_unit = np.load(index_dir / "embedding.npy", mmap_mode=mmap_mode)
    bestmatch = np.load(index_dir / "bestmatch.npy", mmap_mode=mmap_mode)
    sizes = np.load(index_dir / "sizes.npy")
    background = np.load(index_dir / "background.npy")
    membership = sparse.load_npz(index_dir / "membership.npz").tocsr()
    packed = np.load(index_dir / "term_indices.npz")
    term_indices = {
        term: np.asarray(packed[f"arr_{i}"], dtype=np.int32)
        for i, term in enumerate(terms)
    }
    gene_to_index = {g: i for i, g in enumerate(gene_list)}

    return AndesIndex(
        index_dir=index_dir,
        E_unit=E_unit,
        gene_list=gene_list,
        gene_to_index=gene_to_index,
        terms=terms,
        term_indices=term_indices,
        sizes=np.asarray(sizes, dtype=np.int32),
        background=np.asarray(background, dtype=np.int32),
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
    server-side setting. Axis-specific old-ANDES backgrounds should still use
    andes.py/precompute_cache.py.
    """
    cache_path = Path(cache_path)
    cache = func.NullCacheBMA()
    seed = func.NullCacheBMA.resolve_seed(seed)
    null_sampling = "prefix_coupled" if null_mode == "prefix" else "per_size_pair"
    expected = func.NullCacheBMA.build_metadata(
        index.E_unit,
        index.background,
        index.background,
        ite,
        seed,
        null_sampling=null_sampling,
    )
    size_pairs = {(int(query_size), int(k)) for k in np.unique(index.sizes)}

    if cache_path.exists():
        cache.load(cache_path)

    metadata_ok, _ = cache.metadata_matches(expected)
    missing = [pair for pair in size_pairs if pair not in cache.cache]
    if metadata_ok and not missing:
        return cache
    if no_build:
        raise RuntimeError(
            f"cache is missing {len(missing)} size pairs or has incompatible metadata"
        )

    if verbose:
        print(
            f"Building indexed-query null cache: {len(size_pairs)} size pairs, "
            f"query_size={query_size}, ite={ite}, null_mode={null_mode}"
        )
    if null_mode == "prefix":
        cache.precompute_prefix(
            index.E_unit,
            index.background,
            size_pairs,
            ite=ite,
            seed=seed,
            population_idx2=index.background,
            verbose=verbose,
        )
    else:
        cache.precompute(
            index.E_unit,
            index.background,
            size_pairs,
            ite=ite,
            seed=seed,
            population_idx2=index.background,
            verbose=verbose,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.save(cache_path)
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
    df.to_csv(out_path, index=False)
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
    )
    print(f"Wrote index to {args.out}")
    print(f"Best-match matrix: {metadata['bestmatch_mb']:.1f} MB")


def cmd_query(args):
    index = load_andes_index(args.index, mmap=not args.no_mmap)
    genes = load_query_genes(args.genes)
    query_idx, missing = index.map_genes(genes)
    print(
        f"Query genes: {len(query_idx)} matched"
        + (f", {len(missing)} missing" if missing else "")
    )

    cache = None
    if not args.no_zscore:
        index.validate_query_background(query_idx)
        cache_path = args.cache or str(Path(args.index) / "bma_query_null.pkl")
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

    true_scores, zscores = index.score_query(query_idx, cache)
    df = write_query_results(
        args.out,
        index,
        true_scores,
        zscores=zscores,
        top_k=args.top_k,
    )
    print(f"Wrote {len(df)} results to {args.out}")


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
    build.add_argument("--query-memory-mb", type=float, default=1024.0)
    build.add_argument("--verbose", action="store_true")
    build.set_defaults(func=cmd_build)

    query = sub.add_parser("query", help="score one query gene set against an index")
    query.add_argument("--index", required=True, help="index directory from build")
    query.add_argument("--genes", required=True, help="query genes file or one-line GMT")
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

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
