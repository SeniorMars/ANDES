# ANDES

### Algorithm for Network Data Embedding and Similarity analysis

ANDES is a suite of standalone scripts for comparing similarity between gene sets using precomputed gene embeddings.
It includes a consensus protein–protein interaction network embedding generated with node2vec and a sample
geneset database (Gene Ontology Biological Process genesets for _Homo Sapiens_).

### Features

- Gene-set similarity: Compute pairwise similarity scores between two gene sets in embedding space.
- Embedding-based GSEA: Perform a ranked Gene Set Enrichment Analysis (GSEA) using embedding-derived gene rankings.

These functions are implemented in `src/set_analysis_fun.py` and the demo
jupyter notebook (`demo.ipynb`) shows sample usage.

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

## Usage

To quickly get started we recommend looking at our `demo.ipynb`. Alternatively, ANDES can be run
from the command line in both modes with the following commands.

Compute similarity between all pairs of genesets in two databases / gmt files:

```sh
python src/andes.py --emb embedding_file.csv --genelist embedding_gene_ids.txt --geneset1 first_gene_set_database.gmt --geneset2 second_gene_set_database.gmt --out output_file.csv -n num_processor
```

Compute a ranked-based comparison for a geneset database (such as Gene Ontology) given a ranked list of genes 

```sh
python src/andes_gsea.py --emb embedding_file.csv --genelist embedding_gene_ids.txt --geneset gene_set_database.gmt --rankedlist ranked_genes.txt --out output_file.csv -n num_processor
```

## Optimized BMA modes

`src/andes.py` defaults to the fastest exact BMA configuration from Mochi
benchmarks: prefix-coupled null construction and exact gene-to-term best-match
query scoring.

```sh
# Default exact BMA path
python src/andes.py ... --query-mode bestmatch --null-mode prefix

# Previous optimized baseline, useful for regression comparisons
python src/andes.py ... --query-mode batched --null-mode pairwise
```

The two modes optimize different parts of ANDES:

- `--query-mode bestmatch` targets true-score computation after a null cache is
  available.
- `--null-mode prefix` targets null-cache construction. Once the cache exists,
  query-time null normalization is just a lookup by `(m, k)` either way.

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

This preserves the standard BMA score exactly. It can be faster than pairwise
scoring because each gene-to-term best-match value is computed once and reused
across all source terms. The main cost is memory for dense `N_genes x N_terms`
best-match blocks, so the implementation chunks target terms using
`--query-memory-mb`.

For self-vs-self comparisons, the best-match path builds the directed
gene-to-term matrix once. If `geneset1 == geneset2`, then `B1 == B2` and the
score is computed as `GS = (D + D.T) / (sizes[:, None] + sizes[None, :])`,
avoiding the second best-match block pass.

The rough cost comparison is:

```text
pairwise all-vs-all:
  O(d * sum_i |X_i| * sum_j |Y_j|)

gene-to-term best-match:
  O(d * N_genes * sum_j |Y_j|)
  + O(d * N_genes * sum_i |X_i|)
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

For repeated one-gene-set-vs-database queries, use `src/andes_index.py`. This
is a separate feature from the all-vs-all `src/andes.py` path. It builds and
persists the target database structures:

```text
B[g, t] = max_{x in term_t} sim(g, x)
M[t, g] = 1 if gene g belongs to term t
```

Build an index once:

```sh
python src/andes_index.py build \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --out reports/go_bp.index
```

Then score one query gene set against every indexed term:

```sh
python src/andes_index.py query \
  --index reports/go_bp.index \
  --genes query_genes.txt \
  --out reports/query_vs_go_bp.csv \
  --top-k 50
```

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
old-ANDES database backgrounds, continue using `src/andes.py`.

## Optimized ranked ANDES modes

`src/andes_gsea.py` already uses a prefix-coupled ES null cache by default.
An optional exact ranked best-match matrix scorer is available:

```sh
python src/andes_gsea.py ... --score-mode bestmatch
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
`B[ranked_idx, :]`.

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
