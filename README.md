# ANDES

ANDES compares gene sets in a normalized embedding space and performs
embedding-based ranked enrichment analysis.

The package provides one optimized implementation for each workflow:

- database comparison uses exact streamed gene-to-term best matches;
- ranked enrichment streams only the required best-match rows, or traverses a
  persistent index when one is supplied;
- BMA and ranked nulls use prefix-coupled Monte Carlo sampling.

For a guided example, see [demo.ipynb](demo.ipynb).

## Installation

ANDES supports Python 3.11 and 3.12 on POSIX systems. Artifact concurrency
uses POSIX advisory file locks.

```sh
uv sync --dev
```

GEO download support is optional:

```sh
uv sync --extra geo
```

## Compare gene-set databases

```sh
uv run andes compare \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset1 first.gmt \
  --geneset2 second.gmt \
  --out scores.npy \
  --ite 1000
```

The scorer canonicalizes memberships, detects self-comparisons before building
representations, restricts work to source-gene unions, streams best-match
chunks into directed score matrices, and combines the two directions row by
row. `--query-memory-mb` controls the temporary chunk target and defaults to
128 decimal MB. Inputs, outputs, and library overhead remain part of total
process RSS.

Use `--query-blas-threads` for the scoring GEMMs and
`--null-blas-threads` for null construction. A value of zero for query BLAS
threads leaves the installed BLAS runtime unchanged.

## Ranked enrichment

For a pre-ranked gene list:

```sh
uv run andes enrich \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset pathways.gmt \
  --rankedlist ranked_genes.txt \
  --out enrichment.csv \
  --ite 1000
```

Expression input can be ranked with vectorized binary OLS:

```sh
uv run andes enrich \
  --emb embedding.npy \
  --genelist genes.txt \
  --geneset pathways.gmt \
  --expressionfile expression.tsv \
  --out enrichment.csv \
  --empr \
  --n-permutations 1000
```

The expression file is tab-separated. Its first row contains only the binary
condition labels. The second row is the table header: its first field names
the gene-ID column, followed by one sample column per condition label.
Additional metadata columns may follow the samples:

```text
0       0       1       1
gene    s1      s2      s3      s4      metadata
101     5.1     5.3     8.7     8.9     batch_a
102     3.2     3.0     2.1     2.2     batch_b
```

The CLI rejects duplicate gene IDs, nonnumeric sample values, ambiguous
sample columns, and degenerate designs.

Ranked-null construction automatically selects up to eight workers.
`--null-memory-mb` sets one memory allowance shared by those workers. Use
`--workers` for an explicit count, `--worker-blas-threads` for each null
worker, and `--query-blas-threads` for direct scoring.

## Persistent indexes

Build an index once:

```sh
uv run andes index build \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset pathways.gmt \
  --out pathways.index
```

Prebuild the BMA null rows that the query service will accept. The range is
inclusive:

```sh
uv run andes null bma \
  --index pathways.index \
  --query-sizes 1:300 \
  --cache-root /srv/andes/cache \
  --ite 1000
```

Query genes or an indexed term. Index consumers default to
`--cache-policy require`, so a request fails if its null row was not
prebuilt:

```sh
uv run andes index query \
  --index pathways.index \
  --genes query_genes.txt \
  --out query.csv \
  --cache-root /srv/andes/cache \
  --top-k 50

uv run andes index query \
  --index pathways.index \
  --term "GO:0008150" \
  --cache-root /srv/andes/cache \
  --out term_query.csv
```

Batch queries and index-to-index comparisons are also available:

```sh
uv run andes index batch-query \
  --index pathways.index \
  --queries queries.gmt \
  --cache-root /srv/andes/cache \
  --out batch.csv

uv run andes null bma \
  --index first.index \
  --index2 second.index \
  --cache-root /srv/andes/cache

uv run andes index compare \
  --index1 first.index \
  --index2 second.index \
  --cache-root /srv/andes/cache \
  --out comparison.npy
```

Supplying `--index` to `andes enrich` automatically selects indexed ranked
traversal. Its null is ranking-specific, so prebuild it for each deployed
ranking:

```sh
uv run andes null es \
  --index pathways.index \
  --ranked ranked_genes.txt \
  --cache-root /srv/andes/cache

uv run andes enrich \
  --index pathways.index \
  --rankedlist ranked_genes.txt \
  --cache-root /srv/andes/cache \
  --out enrichment.csv
```

Without an index, ranked best-match rows are streamed and consumed without
materializing a full index.

Single, batched, and index-to-index queries use `--query-memory-mb` to bound
temporary GEMM and index slices. Oversized queries are split into gene chunks
before their similarity matrices are formed. The limit excludes required
result arrays, so programmatic callers should batch very large query
collections. Index construction uses the same 128 MB default, reuses its
numerical workspaces across chunks, and validates dense payloads in bounded
row blocks.

Index build, query, batch-query, and compare commands accept
`--query-blas-threads`. Query and compare commands also accept
`--null-blas-threads` for an on-demand BMA null.

Indexes and null models are versioned, validated artifacts. Large arrays can
be memory-mapped. Normal loads validate structure, shapes, dtypes, and compact
identity payloads without scanning the dense arrays. Audit an artifact's full
embedding and best-match hashes explicitly:

```sh
uv run andes index verify --index pathways.index --full
```

### Server deployment

Set one cache root for every builder and query process:

```sh
export ANDES_CACHE_ROOT=/srv/andes/cache
```

Use a controlled build job to publish indexes and prebuild null coverage.
Then mount the index directories and completed cache root read-only and run
query workers with the default `require` policy. Queries memory-map the large
index arrays, perform metadata validation at startup, and never write into an
index or null artifact. Content-addressed null artifacts live under
`$ANDES_CACHE_ROOT/bma` and `$ANDES_CACHE_ROOT/ranked`.

Artifact publication is atomic. Null writers targeting the same artifact are
serialized with an exclusive filesystem lock, and null readers take a shared
lock while loading. Full dense-array hash audits belong in the build/publish
job; routine index startup uses the bounded metadata audit. Publish an index
before exposing its path to workers. Every published index and null has a
stable sibling lock file; copy or mount the artifact and lock together. Index
loaders hold a shared lock while opening and validating every payload, so a
controlled replacement cannot mix generations. A required-cache query fails
if its lock file is missing. An incompatible explicit cache path is rejected
unless the controlled builder is run with `--rebuild-cache`.
The shared filesystem must provide POSIX advisory locks and same-filesystem
atomic renames.

For predictable concurrency, use process workers, load one memory-mapped index
per worker, and establish the BLAS thread limit when each worker starts.
Do not change BLAS thread limits independently inside concurrent request
threads.

## Null artifacts

Comparison and enrichment commands resolve compatible null models
from `--cache-root`, `ANDES_CACHE_ROOT`, or `cache/`, in that order. For
repeated workloads, the same artifacts can be built, listed, or audited
explicitly:

```sh
uv run andes null bma \
  --index pathways.index \
  --query-sizes 1:300 \
  --ite 1000

uv run andes null list
uv run andes null verify bma \
  --index pathways.index \
  --query-sizes 1:300 \
  --ite 1000
```

Artifact identity includes the embedding, background population, sampling
method, seed, iteration count, and ranked-list identity when applicable.
Incompatible or incomplete metadata causes an error.
Requested sizes control an artifact's coverage. Its scientific identity
determines the path, and a controlled builder can extend coverage over time.

## Numerical and statistical semantics

Similarities and persisted best matches use `float32`; ranked means,
cumulative sums, and Monte Carlo null moments use `float64`. Empirical
exceedance counters use `int64`. Ranked scoring is signed and uses one
documented first-near-tie rule in standalone, indexed, and null paths.
Each prefix-coupled null estimate has a valid marginal distribution. Estimates
across sizes are correlated. Null standard deviations use `ddof=1`, and a
zero null standard deviation maps to a z-score of zero. Null construction
requires at least two Monte Carlo iterations.

## Output

CSV is supported for interoperability. For large matrices, `.npy` avoids text
serialization overhead and is accompanied by row, column, and provenance JSON
files. The provenance manifest hashes every component, including the axis
labels, and records the numerical engine, fingerprints, null definition, seed,
precision, and symmetry reuse.

## Development

```sh
just check
uv run python benchmarks/benchmark_optimized.py
uv run python benchmarks/benchmark_optimized.py --real-data
uv run python benchmarks/bench_end_to_end.py
```

The first benchmark isolates the numerical kernels; the end-to-end harness
measures cold/warm CLI workflows and validates their artifacts. See
[TESTING.md](TESTING.md) for benchmark options.

The focused matrix-regression experiment compares DrugBank-by-OMIM z-scores
with a saved reference matrix and can optionally recompute AUPRC against a
binary ground-truth table. The ranked-expression and figure-level paper
workflow is outside its scope. It enforces checked-in coverage, finite-value,
correlation, and error thresholds and returns a nonzero status on failure;
`--report-only` keeps exploratory runs non-blocking:

```sh
just validate-drug-disease-matrix
```

## Citation

If you use ANDES, please cite:

> Li L, Dannenfelser R, Cruz C, Yao V. A best-match approach for gene set
> analyses in embedding spaces. Genome Research. 2024.

[PubMed record](https://pubmed.ncbi.nlm.nih.gov/39231608/)
