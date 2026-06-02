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

To compare BMA modes on Mochi, run the same input with different mode flags and
separate cache paths. The default BMA benchmark mode is now
`--query-mode bestmatch --null-mode prefix`.

```bash
# Previous optimized baseline
python benchmarks/bench_end_to_end.py andes ... \
  --query-mode batched \
  --null-mode pairwise \
  --cache reports/andes_pairwise.pkl \
  --json-out reports/andes_batched_pairwise.json

# Prefix-coupled null build, same batched query scorer
python benchmarks/bench_end_to_end.py andes ... \
  --query-mode batched \
  --null-mode prefix \
  --cache reports/andes_prefix.pkl \
  --json-out reports/andes_batched_prefix.json

# Exact gene-to-term best-match query scorer
python benchmarks/bench_end_to_end.py andes ... \
  --query-mode bestmatch \
  --null-mode pairwise \
  --cache reports/andes_pairwise.pkl \
  --json-out reports/andes_bestmatch_pairwise.json

# Both BMA prototypes
python benchmarks/bench_end_to_end.py andes ... \
  --query-mode bestmatch \
  --null-mode prefix \
  --cache reports/andes_prefix.pkl \
  --json-out reports/andes_bestmatch_prefix.json

# Equivalent to the new default BMA benchmark mode
python benchmarks/bench_end_to_end.py andes ... \
  --cache reports/andes_prefix.pkl \
  --json-out reports/andes_default.json
```

For fair query timing, build the cache once and then rerun without changing the
cache path; compare the `query_scoring` stage. For fair cache-build timing, use
a fresh cache path or remove the old cache first and compare the `cache_build`
stage. BMA JSON reports include `query_mode`,
`null_mode`, `cache_loaded`, `cache_entries`, and `query_workspace_mb`.

To test the persistent one-query index path, build the target database index
once, then query it with a plain one-gene-per-line file or a one-line GMT:

```bash
python src/andes_index.py build \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --out reports/go_bp.index \
  --verbose

python src/andes_index.py query \
  --index reports/go_bp.index \
  --genes query_genes.txt \
  --out reports/query_vs_go_bp.csv \
  --top-k 50 \
  --ite 1000 \
  --null-mode prefix
```

The first indexed query may build `reports/go_bp.index/bma_query_null.pkl` for
that query size. Later queries with the same size and index reuse it.

To validate that the optimized BMA statistics are correct, use the diagnostic
validator. It checks direct true-score equivalence, prefix-vs-pairwise null
estimates for selected sizes, and old-style-vs-optimized z-score/ranking
agreement on a subset:

```bash
python benchmarks/validate_andes_optimizations.py \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset1 data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --geneset2 data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --limit-terms1 50 \
  --limit-terms2 50 \
  --z-ite 1000 \
  --null-ite 5000 \
  --json-out reports/andes_validation_50x50.json
```

Equivalent justfile shortcut:

```bash
just validate-andes-here 50 1000 5000
```

Expected behavior: true-score equivalence should be essentially exact
up to float32 roundoff. Prefix and pairwise null estimates should be close but
not identical. End-to-end z-scores should show rank/correlation agreement, not
bit-for-bit equality, because old ANDES uses independent per-pair Monte Carlo
nulls and different numerical conventions.

For ranked ANDES, the ES null cache is already prefix-coupled. Compare the
current scorer with the exact ranked best-match scorer:

```bash
python benchmarks/bench_end_to_end.py gsea ... \
  --score-mode batched \
  --cache reports/gsea.pkl \
  --json-out reports/gsea_batched.json

python benchmarks/bench_end_to_end.py gsea ... \
  --score-mode bestmatch \
  --cache reports/gsea.pkl \
  --json-out reports/gsea_bestmatch.json
```

Ranked JSON reports include `score_mode`, `cache_loaded`, `cache_entries`, and
`query_workspace_mb`.

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
