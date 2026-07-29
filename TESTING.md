# Testing and benchmarking

## Correctness

Run the complete local quality gate:

```sh
just check
```

This runs Ruff linting, Ruff's formatting check, standard-mode BasedPyright
analysis, and the full pytest suite. Type checking covers the package, tests,
benchmarks, experiments, and tools. Run only the numerical and artifact tests
with `uv run pytest -q`.

The suite emphasizes mathematical and artifact invariants:

- streamed BMA agrees with a scalar reference;
- standalone and indexed ranked scoring agree with a scalar reference;
- canonicalized inputs are deterministic and filter after deduplication;
- prefix null construction is reproducible for a fixed seed;
- index and null artifacts reject corruption and incompatibility;
- vectorized binary OLS agrees with Statsmodels;
- empirical permutation counts agree with a direct implementation.

Performance assertions are deliberately excluded from unit tests.

## Numerical benchmark

Use the in-process benchmark for numerical scorer measurements:

```sh
uv run python benchmarks/benchmark_optimized.py
```

Adjust the generated workload and save machine-readable output:

```sh
uv run python benchmarks/benchmark_optimized.py \
  --genes 5000 \
  --dimensions 256 \
  --terms 1000 \
  --term-size 50 \
  --ranked-length 5000 \
  --repeats 3 \
  --workspace-mb 128 \
  --json-out reports/optimized.json
```

It measures the current scoring paths:

- streamed exact all-vs-all BMA;
- streamed standalone ranked best-match scoring;
- persistent index numerical construction and atomic publication;
- metadata-verified and full-audit memory-mapped index loading;
- indexed ranked traversal over the published artifact.

The script warms the ranked Numba kernels before timing and reports median
wall time, cumulative process peak RSS, runtime versions, BLAS identity, and
planner estimates for diagnosis. It releases large results between repetitions
and measures only the current implementation.

The in-process RSS value is cumulative across all stages. Use the subprocess
benchmark below for stage-level memory regression checks.

Use the bundled propagated GO database to exercise realistic membership
overlap and duplicate sets:

```sh
uv run python benchmarks/benchmark_optimized.py \
  --real-data \
  --repeats 3 \
  --workspace-mb 128 \
  --json-out reports/go-bp-optimized.json
```

The JSON report records total membership occurrences, unique member genes,
the redundancy ratio, duplicate term sets, input time, index construction,
publication, metadata-load and full-audit time, and each scoring path
separately. Standalone and indexed ranked scores must agree as signed values
under the documented accumulation and tie contract.

## End-to-end benchmark

Use the subprocess benchmark to measure complete CLI workflows:

```sh
uv run python benchmarks/bench_end_to_end.py \
  --ite 100 \
  --repeats 3 \
  --null-workers 0 \
  --null-memory-mb 128 \
  --workspace-mb 128 \
  --json-out reports/end-to-end.json
```

This measures artifact-cold and artifact-warm `andes compare` and
`andes enrich` invocations, persistent index construction, index-to-index
comparison, and indexed ranked enrichment. Each sample includes input loading,
artifact validation, scoring, provenance generation, and output serialization.

The harness checks each run's outputs and artifacts:

- warm runs must leave their null artifacts byte-for-byte and timestamp
  unchanged;
- every output component, including matrix axis labels, must match its
  scientific provenance manifest;
- warm results must agree with cold results;
- index-to-index scores must agree with streamed one-shot scores;
- indexed ranked scores must agree with standalone signed scores;
- persistent indexes must remain unchanged when consumed.

Temporary artifacts are removed by default. Pass a path that does not yet
exist to `--artifacts-dir` to retain them for inspection. Use
`--workflow compare` or `--workflow enrich` to isolate one workflow. The
equivalent project commands are `just benchmark` for numerical kernels and
`just benchmark-end-to-end` for complete CLI workflows.

Use a report from the same workload, runtime, BLAS configuration, and machine
as a measured peak-RSS gate:

```sh
uv run python benchmarks/bench_end_to_end.py \
  --ite 100 \
  --repeats 3 \
  --workspace-mb 128 \
  --baseline-json reports/end-to-end-baseline.json \
  --max-rss-regression-percent 5 \
  --json-out reports/end-to-end-current.json
```

The gate compares parent CLI-process peak RSS from each fresh subprocess,
using the median peak for repeated stages. `RUSAGE_CHILDREN` reports a child
high-water mark, so child measurements remain diagnostic. Ranked-null
concurrency is bounded separately by `--null-memory-mb`. The gate uses
measured RSS and records planned workspace for diagnosis.

## Focused profiling

Use the CLI itself for end-to-end profiling:

```sh
time uv run andes compare \
  --emb data/embedding/node2vec_consensus.csv \
  --genelist data/embedding/consensus_node.txt \
  --geneset1 data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --geneset2 data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
  --out reports/go_bp.npy \
  --cache reports/go_bp.null \
  --query-blas-threads 4 \
  --null-blas-threads 1
```

Build the cache once before isolating query performance. Use a new artifact
path when measuring null construction.

## Drug-disease matrix regression

The matrix experiment uses the streamed scorer and prefix-coupled null. It
validates a frozen DrugBank-by-OMIM matrix. The paper's ranked-expression
analysis and figures are outside its scope:

```sh
uv run python experiments/validate_drug_disease_matrix.py \
  --out-dir reports/drug_disease_validation
```

It writes aligned optimized z-scores, differences from the saved paper
matrix, summary agreement metrics, and provenance. The command fails when its
checked-in coverage, finite-value, correlation, or error envelope is not met.
Use `--report-only` only for exploratory comparisons. Pass `--truth` with a
binary drug-by-disease CSV to also compare per-disease AUPRC.
