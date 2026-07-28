#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from andes import data as ld
from andes import bma as func


def load_embedding(path, genelist_path):
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    with open(genelist_path) as fh:
        genes = [line.strip() for line in fh]
    return np.ascontiguousarray(func.l2_normalize_rows(raw), dtype=np.float32), genes


def sample_pairs_by_size(terms, term2idx, n_pairs, seed):
    rng = np.random.default_rng(seed)
    groups = {
        "small": [t for t in terms if len(term2idx[t]) < 30],
        "medium": [t for t in terms if 30 <= len(term2idx[t]) < 80],
        "large": [t for t in terms if len(term2idx[t]) >= 80],
    }
    out = {}
    for name, bucket in groups.items():
        if len(bucket) < 2:
            continue
        pairs = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(bucket), size=2, replace=False)
            pairs.append((bucket[int(i)], bucket[int(j)]))
        out[name] = pairs
    return out


def time_threshold(E_unit, idx, blocks, pairs, threshold, repeats):
    max_m = max(len(idx[t]) for pair in pairs for t in pair)
    ws = func.BMAWorkspaceMax(max_m, max_m, E_unit.shape[1])
    start = time.perf_counter()
    checksum = 0.0
    for _ in range(repeats):
        for t1, t2 in pairs:
            x = idx[t1]
            y = idx[t2]
            if len(x) * len(y) < threshold:
                checksum += func.compute_bma_numba(E_unit, x, y)
            else:
                checksum += func.compute_bma_blocks_ws(
                    blocks[t1], blocks[t2], ws.views(len(x), len(y))
                )
    elapsed = time.perf_counter() - start
    return elapsed / (len(pairs) * repeats), checksum


def main():
    p = argparse.ArgumentParser(description="Benchmark BMA Numba/BLAS threshold")
    p.add_argument("--emb", default="data/embedding/node2vec_consensus.csv")
    p.add_argument("--genelist", default="data/embedding/consensus_node.txt")
    p.add_argument("--geneset", default="data/gene_sets/hsa_experimental_eval_BP_propagated.gmt")
    p.add_argument("--min", dest="min_size", type=int, default=10)
    p.add_argument("--max", dest="max_size", type=int, default=300)
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--thresholds", default="0,400,1000,2500,5000,10000")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    E_unit, genes = load_embedding(args.emb, args.genelist)
    func.warmup_numba()
    node_set = set(genes)
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(genes)})
    geneset = ld.load_gmt(args.geneset)
    idx = func.preconvert_indices_to_arrays(
        ld.term2indexes(geneset, g_node2index, upper=args.max_size, lower=args.min_size)
    )
    terms = list(idx)
    blocks = func.precompute_term_embedding_blocks(E_unit, idx)
    pair_groups = sample_pairs_by_size(terms, idx, args.pairs, args.seed)
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]

    report = {"thresholds": thresholds, "groups": {}}
    for group_name, pairs in pair_groups.items():
        print(f"\n{group_name} ({len(pairs)} pairs)")
        rows = []
        for threshold in thresholds:
            seconds_per_pair, checksum = time_threshold(
                E_unit, idx, blocks, pairs, threshold, args.repeats
            )
            ms = seconds_per_pair * 1000
            rows.append({"threshold": threshold, "ms_per_pair": ms, "checksum": checksum})
            print(f"  threshold={threshold:6d}  {ms:8.4f} ms/pair")
        best = min(rows, key=lambda r: r["ms_per_pair"])
        print(f"  best threshold: {best['threshold']}")
        report["groups"][group_name] = rows

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
