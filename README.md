# ANDES

### Algorithm for Network Data Embedding and Similarity analysis

ANDES is a Python package and command-line application for comparing gene sets
in precomputed embedding spaces. It includes a consensus protein–protein
interaction network embedding generated with node2vec and a sample gene-set
database (Gene Ontology Biological Process gene sets for _Homo sapiens_).

### Features

- Gene-set similarity: Compute pairwise similarity scores between two gene sets in embedding space.
- Embedding-based GSEA: Perform a ranked Gene Set Enrichment Analysis (GSEA) using embedding-derived gene rankings.

The numerical library lives in `src/andes/`. Exact scoring, null calibration,
artifact persistence, and command orchestration are separate modules. The demo
Jupyter notebook (`demo.ipynb`) shows sample usage.

## Citation

If you use ANDES in your work, please cite:
> [A best-match approach for gene set analyses in embedding spaces.](https://pubmed.ncbi.nlm.nih.gov/39231608/)
Li L, Dannenfelser R, Cruz C, Yao V. Genome Research. 2024.

## Installation

1. Install conda if you haven't already
2. Create and activate the ANDES environment:
   
```sh
conda env create -f env.yml
conda activate ANDES
```

For development with `uv`, use:

```sh
uv sync --dev
```

Paper-validation plots and GEO downloads are optional:

```sh
uv sync --extra experiments --extra geo
```

## Usage

To quickly get started we recommend looking at our `demo.ipynb`. Alternatively, ANDES can be run
from the command line in both modes with the following commands.

Compute similarity between all pairs of genesets in two databases / gmt files:

```sh
uv run andes compare --emb embedding_file.csv --genelist embedding_gene_ids.txt --geneset1 first_gene_set_database.gmt --geneset2 second_gene_set_database.gmt --out output_file.csv -n num_processor
```

Compute a ranked-based comparison for a geneset database (such as Gene Ontology) given a ranked list of genes 

```sh
uv run andes enrich --emb embedding_file.csv --genelist embedding_gene_ids.txt --geneset gene_set_database.gmt --rankedlist ranked_genes.txt --out output_file.csv -n num_processor
```

## Optimized BMA modes

`andes compare` defaults to the fastest exact BMA configuration from Mochi
benchmarks: prefix-coupled null construction and exact gene-to-term best-match
query scoring.

```sh
# Default exact BMA path
uv run andes compare ... --query-mode bestmatch --null-mode prefix

# Previous optimized baseline, useful for regression comparisons
uv run andes compare ... --query-mode batched --null-mode pairwise
```

The two modes optimize different parts of ANDES:

- `--query-mode bestmatch` targets true-score computation after a null cache is
  available.
- `--null-mode prefix` targets null-cache construction. Once the cache exists,
  query-time null normalization is just a lookup by `(m, k)` either way.

Main-query and null-worker parallelism are controlled separately:

```sh
uv run andes compare ... \
  --query-blas-threads 4 \
  --worker-blas-threads 1 \
  --workers 8
```

The exact `bestmatch` query uses one Python thread and the requested BLAS width.
Outer-threaded `batched` and `pairwise` modes default to one BLAS thread per
Python worker. A zero `--query-blas-threads` keeps the runtime BLAS default for
`bestmatch`.

### Exact gene-to-term best-match scoring

Standard BMA for term pair `X_i, Y_j` is:

```text
GS(X_i, Y_j) =
  (sum_{x in X_i} max_{y in Y_j} sim(x, y)
 + sum_{y in Y_j} max_{x in X_i} sim(x, y))
 / (|X_i| + |Y_j|)
```

For large database-vs-database scoring, many source terms repeatedly ask the
same question: what is gene `g`'s best match to target term `Y_j`? The
best-match prototype precomputes that reusable value:

```text
B2[g, j] = max_{y in Y_j} sim(g, y)
```

If `M1` is a sparse term-by-gene membership matrix for database 1, then every
directed score from database 1 to database 2 is:

```text
D12 = M1 @ B2
D12[i, j] = sum_{x in X_i} B2[x, j]
```

The reverse direction is symmetric:

```text
B1[g, i] = max_{x in X_i} sim(g, x)
D21 = M2 @ B1
```

The exact ANDES true-score matrix is then:

```text
GS = (D12 + D21.T) / (sizes1[:, None] + sizes2[None, :])
```

This preserves the standard BMA score exactly. The one-shot implementation
canonicalizes and packs term memberships, restricts best-match rows to genes
used by the source database, and streams target-term chunks directly into
`D = M @ B_chunk`. It therefore never retains a complete dense `B` matrix.
`--query-memory-mb` sets the target size for temporary similarity chunks (one
indivisible term can exceed a smaller target).

For self-vs-self comparisons, symmetry is detected before representations are
built. One packed term axis and one streamed directed matrix are reused, with
`GS = (D + D.T) / (sizes[:, None] + sizes[None, :])`.

The rough cost comparison is:

```text
pairwise all-vs-all:
  O(d * sum_i |X_i| * sum_j |Y_j|)

gene-to-term best-match:
  O(d * |union_i X_i| * sum_j |Y_j|)
  + O(d * |union_j Y_j| * sum_i |X_i|)
  + sparse aggregation
```

This is most attractive when many terms reuse the same genes and the full
all-vs-all score matrix is needed. For a small number of queried pairs, the
default batched/pairwise scorer may still be faster.

### Prefix-coupled BMA null cache

The BMA null cache stores one entry per requested size pair:

```text
cache[(m, k)] = (mu, sigma)
```

The pairwise null builder estimates each `(m, k)` distribution independently.
The prefix-coupled default instead draws one random permutation of each
background per Monte Carlo iteration, computes the largest needed prefix matrix,
and updates every requested prefix size pair from that matrix.

The pairwise builder does:

```text
for each size pair (m, k):
  repeat ite times:
    sample random X of size m
    sample random Y of size k
    compute BMA(X, Y)
```

For the GO BP benchmark with 62,001 requested size pairs and `ite=1000`, that
is roughly:

```text
62,001 * 1000 = 62,001,000 random BMA simulations
```

Even with batched sampling and batched GEMM, that is a large amount of repeated
Monte Carlo work.

The prefix-coupled builder does:

```text
repeat ite times:
  sample one random ordered background prefix for axis 1
  sample one random ordered background prefix for axis 2
  take max_m and max_k genes
  compute one max_m x max_k similarity matrix
  use cumulative max/sum to read out every requested (m, k)
```

So with `ite=1000`, it performs 1000 prefix matrix simulations, not 62 million
independent size-pair simulations.

For one iteration, with maximum requested sizes `M1` and `M2`:

```text
X = E[perm1[:M1]]
Y = E[perm2[:M2]]
A = X @ Y.T
```

Cumulative maxima recover every prefix BMA score:

```text
row_best = maximum.accumulate(A, axis=1)
row_sum  = cumsum(row_best, axis=0)

col_best = maximum.accumulate(A, axis=0)
col_sum  = cumsum(col_best, axis=1)

score[m-1, k-1] =
  (row_sum[m-1, k-1] + col_sum[m-1, k-1]) / (m + k)
```

This is valid because the first `m` elements of a random permutation are a
uniform sample without replacement of size `m`. Null estimates for different
sizes are correlated because they share permutations, but each individual
`(m, k)` estimate is marginally valid.

Prefix-coupled nulls therefore preserve the correct marginal null distribution
for each size pair `(m, k)`, but estimates across different size pairs are
correlated because they are derived from shared random permutations. This is
acceptable for ANDES z-scoring, which uses each size-pair mean and standard
deviation independently, but prefix and per-size-pair caches should not be
expected to match bit-for-bit.

The rough cost comparison is:

```text
default null build:
  O(ite * sum_{(m,k) requested} m * k * d)

prefix-coupled null build:
  O(ite * max_m * max_k * d)
  + cumulative max/cumsum overhead
```

So prefix mode is likely faster when many size pairs are requested, especially
when requested sizes are dense. It may be slower for a small or sparse set of
size pairs that includes one large `(max_m, max_k)` pair.

Prefix-built caches are marked with metadata
`null_sampling="prefix_coupled"`. Default caches use
`null_sampling="per_size_pair"`, so the two cache types are not accidentally
mixed. After a cache is accepted as valid, both modes use the same query-time
lookup by `(m, k)`.

### Equivalence and caveats versus old ANDES

The optimized BMA path preserves the standard `distinct=False` ANDES statistic
and the marginal null distribution for each size pair `(m, k)`. Old ANDES draws
independent random sets for every term pair:

```text
X ~ uniform size-m subset of population1
Y ~ uniform size-k subset of population2
score = BMA(X, Y)
```

The prefix-coupled cache uses prefixes of random permutations. For any fixed
`m` and `k`, those prefixes are also uniform samples without replacement, so
each individual `(m, k)` null distribution is preserved.

The optimized output is still not expected to match old ANDES bit-for-bit:

- Old ANDES estimates an independent Monte Carlo null for every term pair. The
  optimized cache reuses one null estimate for all term pairs with the same
  `(m, k)`.
- Prefix-coupled caches also correlate null estimates across different size
  pairs because they share random permutation prefixes. This changes Monte
  Carlo noise correlation, not the marginal null for any one `(m, k)`.
- Old ANDES uses `np.std(back_scores)` with NumPy's default `ddof=0`; the
  optimized BMA cache stores sample standard deviations with `ddof=1`.
- Old ANDES computes a full sklearn cosine matrix from float64-loaded
  embeddings. The optimized implementation normalizes rows as float32 and uses
  dot products, so low-order numerical differences are expected.
- Old ANDES can produce `inf` or `nan` when the null standard deviation is zero.
  The optimized z-score lookup returns `0` for zero-sigma null entries.
- The size-pair null-cache argument applies to standard `distinct=False` ANDES.
  Dynamic per-pair overlap removal would change the effective target set and
  should use a separate scorer/null strategy.

Validation should therefore target statistical agreement, rank stability, and
biological conclusions rather than exact equality with old ANDES outputs.

## Persistent ANDES database indexes

For repeated one-gene-set-vs-database queries, use `andes index`. This is a
separate feature from the all-vs-all `andes compare` path. It builds and
persists the target database structures:

```text
B[g, t] = max_{x in term_t} sim(g, x)
M[t, g] = 1 if gene g belongs to term t
```

Build an index once:

```sh
uv run andes index build \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --out reports/go_bp.index
```

Then score one query gene set against every indexed term:

```sh
uv run andes index query \
  --index reports/go_bp.index \
  --genes query_genes.txt \
  --out reports/query_vs_go_bp.csv \
  --top-k 50
```

The same artifact supports indexed terms, batches, and full index comparisons:

```sh
# Reuse an indexed term's existing B column
uv run andes index query \
  --index reports/go_bp.index --term GO:0008150 \
  --out reports/term_vs_go_bp.csv

# One GMT row per query set
uv run andes index batch-query \
  --index reports/go_bp.index --queries query_sets.gmt \
  --out reports/queries_vs_go_bp.csv --top-k 50

# Both indexes must use the identical normalized embedding and gene row order
uv run andes index compare \
  --index1 reports/db1.index --index2 reports/db2.index \
  --out reports/db1_vs_db2.npy
```

Index construction writes `bestmatch.npy` incrementally with a memory map.
Loading validates the format version, shapes, dtypes, gene and term order,
memberships, backgrounds, and payload hashes. Binary comparison output includes
`.rows.json` and `.columns.json` label sidecars. Version-1 indexes must be
rebuilt.

The exact query formula is:

```text
query_to_terms = B[query_idx, :].sum(axis=0)
best_to_query  = (E @ E[query_idx].T).max(axis=1)
terms_to_query = M @ best_to_query
true           = (query_to_terms + terms_to_query) / (len(query_idx) + sizes)
```

This preserves standard ANDES BMA exactly. It is faster for repeated queries
because the database-side best-match matrix `B` is reused from disk. The query
command can also build/load a BMA null cache for the query size, using the
index background as the shared reusable null population. For axis-specific
old-ANDES database backgrounds, continue using `andes compare`.

## Optimized ranked ANDES modes

`andes enrich` uses a prefix-coupled ES null cache by default. It supports
both an in-memory ranked best-match scorer and direct traversal of a persistent
index:

```sh
uv run andes enrich ... --score-mode bestmatch

uv run andes enrich \
  --index reports/go_bp.index --score-mode indexed \
  --rankedlist ranked.tsv --out reports/ranked_vs_go_bp.csv
```

For a ranked list `R` and term `X_t`, ranked ANDES first computes:

```text
S[r, t] = max_{x in X_t} sim(R[r], x)
```

The enrichment score for term `t` is then:

```text
centered[:, t] = S[:, t] - mean(S[:, t])
running[:, t]  = cumsum(centered[:, t])
ES[t]          = running[argmax(abs(running[:, t])), t]
```

The default ranked scorer groups terms by size and computes each group's
similarity blocks. The `bestmatch` prototype instead builds ranked-gene-to-term
best-match blocks directly, then computes all ES traces column-wise. This is
exact for the existing ranked ANDES score and can be faster when many terms are
scored against the same ranked list.

The rough true-score cost comparison is:

```text
default ranked scorer:
  O(d * ranked_length * sum_t |X_t|)
  grouped into size batches

ranked best-match scorer:
  O(d * ranked_length * sum_t |X_t|)
  reused as S[ranked_position, term] blocks before column-wise ES
```

The asymptotic dot-product count is similar for a single ranked list, so speed
depends on batching, memory layout, and term reuse. The best-match formulation
becomes more attractive when the same gene-set database is scored against many
ranked lists, because the reusable full matrix
`B[g, t] = max_{x in X_t} sim(g, x)` can be cached or chunked and sliced as
`B[ranked_idx, :]`. Indexed mode performs exactly that slice in bounded term
chunks and recomputes no embedding similarities.

For empirical expression permutations:

```sh
uv run andes enrich \
  --index reports/go_bp.index --score-mode indexed \
  --expressionfile expression.tsv --empr \
  --n-permutations 1000 --permutation-batch-size 32 \
  --out reports/expression_vs_go_bp.csv
```

Two-group OLS statistics are computed for a permutation batch with one
expression-by-condition GEMM, expression genes are mapped to embedding rows
once, and each permuted ranking traverses cached `B`. Only per-term exceedance
counts are retained; no permutation-by-term score matrix is materialized.
Use `--query-blas-threads` for ranked scoring and OLS, and
`--worker-blas-threads` for null construction. The legacy
`--blas-threads` spelling remains an alias for worker BLAS threads.

The ranked null cache is already prefix-coupled. For each Monte Carlo
iteration it samples one random gene-set prefix up to `max_m`, computes one
`max_m x ranked_length` similarity block, and updates all requested gene-set
sizes. This changes null construction from roughly:

```text
O(ite * sum_m m * ranked_length * d)
```

to:

```text
O(ite * max_m * ranked_length * d)
```

while preserving a normal cache lookup interface:

```text
cache[m] = (mu, sigma)
```

## Library API and artifacts

The public library uses canonical, validated domain objects:

```python
from andes.data import load_embedding_space, load_gene_set_database
from andes.scoring import score_bma_matrix

embedding = load_embedding_space("embedding.csv", "genes.txt")
database = load_gene_set_database("sets.gmt", embedding)
result = score_bma_matrix(embedding, database, database)
```

Exact scores and null calibration are intentionally separate. Use
`BmaNullModel` or `RankedNullModel` from `andes.nulls` to standardize an exact
result. New null artifacts are versioned directories containing JSON metadata
and non-pickled NumPy arrays. Index artifacts validate versions, shapes, dtypes,
canonical memberships, fingerprints, and payload hashes before use.

Every result written by the application layer also receives a
`.metadata.json` provenance sidecar recording the method, engine, numerical
types, null specification, input fingerprints, and output hash.

The supported command hierarchy is:

```text
andes compare
andes enrich
andes index build|query|batch-query|compare
andes null bma|es|list|verify
```

Legacy scalar implementations are retained only under `experiments/legacy/`
as benchmark references. Production code imports from `andes` and uses the
unified CLI.
