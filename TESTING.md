# Testing and Profiling

Use two levels of checks:

```bash
python -m unittest discover -s tests
```

This is the fast correctness suite. It validates BMA scoring, GSEA scoring,
cache metadata round trips, separate ANDES null backgrounds, cost chunking, and
the benchmark JSON harness on synthetic data.

For staged performance profiling, use `bench_end_to_end.py`:

```bash
python bench_end_to_end.py andes \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset1 data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --geneset2 data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --ite 200 \
  --workers 8 \
  --query-workers 8 \
  --cache reports/andes_cache.pkl \
  --json-out reports/andes.json \
  --profile-out reports/andes.prof
```

For GSEA:

```bash
python bench_end_to_end.py gsea \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --rankedlist data/expression/GSE3467_rank.txt \
  --ite 1000 \
  --workers 8 \
  --cache reports/gsea_cache.pkl \
  --json-out reports/gsea.json \
  --profile-out reports/gsea.prof
```

The JSON report records stage timings for load, normalize, Numba warmup,
gene-set parsing, cache build, query scoring, and save. The `.prof` file can be
opened with `snakeviz` or inspected with:

```bash
python -m pstats reports/andes.prof
```

Use `--skip-cache-build` when isolating query scoring, and `--limit-terms1`,
`--limit-terms2`, or `--limit-terms` for quick profiling passes.

Use `--cache` on repeated runs. Cache files include metadata for the embedding,
background, ranked list where relevant, iteration count, and seed; invalid cache
files are ignored and rebuilt.

Use `--seed -1` when you want OS entropy instead of a fixed seed. The resolved
seed is written into JSON/config output and cache metadata.

To tune the BMA Numba/BLAS cutoff on a machine:

```bash
python bench_bma_threshold.py \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset data/gene_sets/hsa_experimental_eval_BP_propagated.gmt
```

Then pass the best cutoff to `andes.py` or `bench_end_to_end.py` with
`--numba-threshold`.
