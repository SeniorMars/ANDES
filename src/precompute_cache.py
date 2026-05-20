"""
precompute_cache.py
Build and persist null-distribution caches for many GMT files at once,
so a server can serve ANDES queries without per-request Monte Carlo.

Two cache families
------------------
1. BMA caches:  one per (embedding, background) tuple.
   Reusable for ANY pair of GMT files that share the same embedding +
   background population. Keyed by (m, k) gene-set sizes.

2. ES caches:   one per (embedding, background, ranked_list) tuple.
   Tied to a specific ranked list (and so a specific experiment), but
   reusable across all GMT files queried against that ranking.

Both are content-addressed: the cache filename encodes a hash of the
inputs that determine the null. Rebuilding with the same hash inputs
short-circuits.

Layout
------
  CACHE_ROOT/
    bma/
      manifest.json
      <emb_hash>__<pop_hash>__ite<N>__seed<S>.pkl
    es/
      manifest.json
      <emb_hash>__<pop_hash>__<ranked_hash>__ite<N>__seed<S>.pkl

Usage
-----
  # Build BMA cache for one embedding × union of many GMTs
  python precompute_cache.py bma \
      --emb data/embedding/node2vec_consensus.csv \
      --genelist data/embedding/consensus_node.txt \
      --gmt data/gene_sets/*.gmt \
      --workers 8

  # Build ES cache for one embedding × one ranked list × many GMTs
  python precompute_cache.py es \
      --emb data/embedding/node2vec_consensus.csv \
      --genelist data/embedding/consensus_node.txt \
      --gmt data/gene_sets/*.gmt \
      --ranked data/expression/GSE3467_rank.txt \
      --workers 8

  # List existing caches
  python precompute_cache.py list

  # Verify a cache against current size pairs (no rebuild, just report gaps)
  python precompute_cache.py verify bma --emb ... --gmt ...

Server-side query pattern
-------------------------
  cache = NullCacheBMA(); cache.load(cache_path_for(emb_id, pop_id))
  for term1, term2 in pairs:
      score = compute_bma(...)
      z     = cache.get_zscore(score, m, k)        # O(1)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import glob
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import load_data as ld
import func_optimized as func_new
from func_gsea import (
    NullCacheESBetter as NullCacheES,
    compute_ranked_emb,
    warmup_numba_es,
)

CACHE_ROOT = Path(os.environ.get("ANDES_CACHE_ROOT", "./andes_cache"))


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def _hash_array(a: np.ndarray, n_bytes: int = 8) -> str:
    """Stable short hash of an ndarray's contents."""
    h = hashlib.blake2b(digest_size=n_bytes)
    h.update(a.tobytes(order="C"))
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    return h.hexdigest()


def _hash_iterable(items, n_bytes: int = 8) -> str:
    h = hashlib.blake2b(digest_size=n_bytes)
    for x in items:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()


def emb_id(E_unit: np.ndarray, gene_list: list[str]) -> str:
    """Hash of embedding + gene order. Two embeddings with the same numeric
    content but different gene order will get different ids (intentional)."""
    return _hash_array(E_unit) + "_" + _hash_iterable(gene_list)[:8]


def pop_id(pop: np.ndarray) -> str:
    """Hash of background population indices."""
    return _hash_array(np.sort(pop))


def ranked_id(ranked_idx: np.ndarray) -> str:
    """Hash of ranked list (ordered)."""
    return _hash_array(ranked_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_manifest(path: Path, manifest: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding(emb_path: str, genelist_path: str):
    raw = np.loadtxt(emb_path, delimiter=",", dtype=np.float32)
    with open(genelist_path) as fh:
        gene_list = [line.strip() for line in fh]
    if len(gene_list) != raw.shape[0]:
        raise ValueError("embedding rows do not match gene-list length")
    E_unit = np.ascontiguousarray(
        func_new.l2_normalize_rows(raw), dtype=np.float32
    )
    return E_unit, gene_list


def load_gmt_to_indices(gmt_paths, gene_list, min_size, max_size):
    """Load and merge multiple GMTs. Returns:
        union_geneset: dict[term -> list[gene_str]]   (across all files)
        union_indices: dict[term -> np.int32 array]   (after embedding intersect)
        per_file_terms: dict[gmt_path -> list[term]]  (so we can report per file)
    """
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(gene_list)})
    union_geneset = {}
    per_file_terms = {}

    for gmt in gmt_paths:
        d = ld.load_gmt(gmt)
        union_geneset.update(d)
        per_file_terms[gmt] = list(d.keys())

    union_indices_set = ld.term2indexes(
        union_geneset, g_node2index, upper=max_size, lower=min_size
    )
    union_indices = func_new.preconvert_indices_to_arrays(union_indices_set)
    return union_geneset, union_indices, per_file_terms


def background_pop(union_geneset, gene_list, gene_list_set):
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(gene_list)})
    all_genes = set().union(*union_geneset.values())
    all_genes &= gene_list_set
    return np.array(sorted(g_node2index[g] for g in all_genes), dtype=np.int32)


def load_ranked(ranked_path: str, gene_list_set, gene_list):
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(gene_list)})
    df = pd.read_csv(ranked_path, sep="\t", index_col=0, header=None)
    return np.array(
        [g_node2index[str(g)] for g in df.index if str(g) in gene_list_set],
        dtype=np.int32,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_bma(args):
    print("=" * 70)
    print("BUILD BMA NULL CACHE")
    print("=" * 70)

    gmt_paths = sorted(set(p for pat in args.gmt for p in glob.glob(pat)))
    if not gmt_paths:
        print("No GMT files matched.")
        sys.exit(1)
    print(f"GMT files ({len(gmt_paths)}):")
    for p in gmt_paths:
        print(f"  {p}")

    E_unit, gene_list = load_embedding(args.emb, args.genelist)
    gene_list_set = set(gene_list)
    print(f"\nEmbedding: {E_unit.shape}  ({E_unit.nbytes/1e6:.1f} MB)")

    union_geneset, union_indices, per_file = load_gmt_to_indices(
        gmt_paths, gene_list, args.min, args.max
    )
    pop = background_pop(union_geneset, gene_list, gene_list_set)
    print(f"Terms (after size filter): {len(union_indices)}")
    print(f"Background: {len(pop)} genes")

    sizes = sorted({len(arr) for arr in union_indices.values()})
    size_pairs = {(m, k) for m in sizes for k in sizes}
    print(f"Unique gene-set sizes: {len(sizes)}  → {len(size_pairs)} (m,k) pairs")

    eid = emb_id(E_unit, gene_list)
    pid = pop_id(pop)
    fname = f"{eid[:16]}__{pid[:16]}__ite{args.ite}__seed{args.seed}.pkl"
    cache_dir = CACHE_ROOT / "bma"
    cache_path = cache_dir / fname
    manifest_path = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCache target: {cache_path}")

    cache = func_new.NullCacheBMA()

    if cache_path.exists() and not args.rebuild:
        cache.load(str(cache_path))
        missing = [pair for pair in size_pairs if pair not in cache.cache]
        print(f"Loaded existing cache: {len(cache.cache)} entries")
        print(f"Missing for this GMT union: {len(missing)} pairs")
        if not missing:
            print("All needed pairs present. No work to do.")
            _record_manifest(manifest_path, fname, gmt_paths, args, eid, pid)
            return
        size_pairs = set(missing)

    print(f"\nBuilding {len(size_pairs)} entries with {args.workers} workers...")
    func_new.warmup_numba()
    t0 = time.perf_counter()

    if args.workers > 1:
        cache.precompute_parallel(
            E_unit, pop, size_pairs,
            ite=args.ite, seed=args.seed, verbose=True,
            n_workers=args.workers,
            chunk_size=None if args.chunk_size <= 0 else args.chunk_size,
        )
    else:
        cache.precompute(
            E_unit, pop, size_pairs,
            ite=args.ite, seed=args.seed, verbose=True,
        )

    elapsed = time.perf_counter() - t0
    print(f"\nBuild time: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    cache.save(str(cache_path))
    _record_manifest(manifest_path, fname, gmt_paths, args, eid, pid)
    print(f"Saved cache to {cache_path}")


def cmd_es(args):
    print("=" * 70)
    print("BUILD ES NULL CACHE")
    print("=" * 70)

    gmt_paths = sorted(set(p for pat in args.gmt for p in glob.glob(pat)))
    if not gmt_paths:
        print("No GMT files matched.")
        sys.exit(1)

    E_unit, gene_list = load_embedding(args.emb, args.genelist)
    gene_list_set = set(gene_list)

    union_geneset, union_indices, per_file = load_gmt_to_indices(
        gmt_paths, gene_list, args.min, args.max
    )
    pop = background_pop(union_geneset, gene_list, gene_list_set)
    sizes = sorted({len(arr) for arr in union_indices.values()})
    print(f"\nTerms: {len(union_indices)}   Sizes: {len(sizes)}")

    ranked_idx = load_ranked(args.ranked, gene_list_set, gene_list)
    if len(ranked_idx) == 0:
        print("Ranked list empty after filtering to embedding genes.")
        sys.exit(1)
    print(f"Ranked list: {len(ranked_idx)} genes")

    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)

    eid = emb_id(E_unit, gene_list)
    pid = pop_id(pop)
    rid = ranked_id(ranked_idx)
    fname = f"{eid[:16]}__{pid[:16]}__{rid[:16]}__ite{args.ite}__seed{args.seed}.pkl"
    cache_dir = CACHE_ROOT / "es"
    cache_path = cache_dir / fname
    manifest_path = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCache target: {cache_path}")

    cache = NullCacheES()
    if cache_path.exists() and not args.rebuild:
        cache.load(str(cache_path))
        missing = [m for m in sizes if m not in cache.cache]
        print(f"Loaded existing cache: {len(cache)} entries")
        print(f"Missing sizes: {len(missing)}")
        if not missing:
            print("All needed sizes present. No work to do.")
            _record_manifest(manifest_path, fname, gmt_paths, args, eid, pid,
                             ranked_path=args.ranked, rid=rid)
            return
        sizes = missing

    func_new.warmup_numba()
    warmup_numba_es()

    t0 = time.perf_counter()
    if args.workers > 1:
        cache.precompute_parallel(
            E_unit, pop, sizes, ranked_emb,
            ite=args.ite, seed=args.seed, verbose=True,
            n_workers=args.workers,
            chunk_size=None if args.chunk_size <= 0 else args.chunk_size,
        )
    else:
        cache.precompute(
            E_unit, pop, sizes, ranked_emb,
            ite=args.ite, seed=args.seed, verbose=True,
        )
    elapsed = time.perf_counter() - t0
    print(f"\nBuild time: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    cache.save(str(cache_path))
    _record_manifest(manifest_path, fname, gmt_paths, args, eid, pid,
                     ranked_path=args.ranked, rid=rid)
    print(f"Saved cache to {cache_path}")


def _record_manifest(manifest_path, fname, gmt_paths, args, eid, pid,
                     ranked_path=None, rid=None):
    manifest = load_manifest(manifest_path)
    entry = {
        "file": fname,
        "ite": args.ite,
        "seed": args.seed,
        "min_size": args.min,
        "max_size": args.max,
        "emb_id": eid,
        "pop_id": pid,
        "gmts": gmt_paths,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if ranked_path is not None:
        entry["ranked_path"] = ranked_path
        entry["ranked_id"] = rid
    manifest[fname] = entry
    save_manifest(manifest_path, manifest)


def cmd_list(args):
    print(f"Cache root: {CACHE_ROOT.resolve()}")
    for kind in ("bma", "es"):
        mpath = CACHE_ROOT / kind / "manifest.json"
        m = load_manifest(mpath)
        print(f"\n[{kind}] {len(m)} cache(s)")
        for fname, info in m.items():
            f = (CACHE_ROOT / kind / fname)
            size_kb = f.stat().st_size / 1024 if f.exists() else 0
            print(f"  {fname}  ({size_kb:.1f} KB)")
            print(f"    built_at : {info.get('built_at')}")
            print(f"    ite/seed : {info.get('ite')}/{info.get('seed')}")
            print(f"    gmts     : {len(info.get('gmts', []))} files")


def cmd_verify(args):
    """Report which (m,k) or m sizes would be missing for a given query setup."""
    kind = args.kind
    if kind == "bma":
        gmt_paths = sorted(set(p for pat in args.gmt for p in glob.glob(pat)))
        E_unit, gene_list = load_embedding(args.emb, args.genelist)
        gene_list_set = set(gene_list)
        union_geneset, union_indices, _ = load_gmt_to_indices(
            gmt_paths, gene_list, args.min, args.max
        )
        pop = background_pop(union_geneset, gene_list, gene_list_set)
        sizes = sorted({len(arr) for arr in union_indices.values()})
        size_pairs = {(m, k) for m in sizes for k in sizes}

        eid = emb_id(E_unit, gene_list)
        pid = pop_id(pop)
        fname = f"{eid[:16]}__{pid[:16]}__ite{args.ite}__seed{args.seed}.pkl"
        cache_path = CACHE_ROOT / "bma" / fname

        if not cache_path.exists():
            print(f"Cache file does not exist: {cache_path}")
            sys.exit(1)
        cache = func_new.NullCacheBMA()
        cache.load(str(cache_path))
        missing = [pair for pair in size_pairs if pair not in cache.cache]
        print(f"Need {len(size_pairs)} (m,k) pairs; cache has {len(cache.cache)};"
              f" missing {len(missing)}")
        if missing:
            print("Sample missing:", missing[:10])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared
    def add_common(s):
        s.add_argument("--emb",       required=True)
        s.add_argument("--genelist",  required=True)
        s.add_argument("--gmt",       required=True, nargs="+",
                       help="One or more GMT files or globs")
        s.add_argument("--min",       type=int, default=10)
        s.add_argument("--max",       type=int, default=300)
        s.add_argument("--ite",       type=int, default=1000)
        s.add_argument("--seed",      type=int, default=12345)
        s.add_argument("--workers",   type=int, default=8)
        s.add_argument("--chunk_size", type=int, default=0)
        s.add_argument("--rebuild",   action="store_true",
                       help="Discard any existing matching cache and rebuild")

    sb = sub.add_parser("bma", help="Build BMA cache (set vs set)")
    add_common(sb)
    sb.set_defaults(func=cmd_bma)

    se = sub.add_parser("es", help="Build ES cache (set vs ranked list)")
    add_common(se)
    se.add_argument("--ranked", required=True)
    se.set_defaults(func=cmd_es)

    sl = sub.add_parser("list", help="List existing caches")
    sl.set_defaults(func=cmd_list)

    sv = sub.add_parser("verify", help="Check coverage of an existing cache")
    sv.add_argument("kind", choices=["bma"])  # es could be added similarly
    add_common(sv)
    sv.set_defaults(func=cmd_verify)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
