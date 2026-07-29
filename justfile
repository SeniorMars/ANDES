python := env_var_or_default("ANDES_PYTHON", "uv run python")
embedding := "data/embedding/node2vec_consensus.csv"
genes := "data/embedding/consensus_node.txt"
gene_sets := "data/gene_sets/hsa_experimental_eval_BP_propagated.gmt"
ranked := "data/expression/GSE3467_rank.txt"

default:
    just --list

sync:
    uv sync --dev

test:
    {{python}} -m pytest -q

lint:
    uv run ruff check src tests benchmarks experiments tools

format:
    uv run ruff format src tests benchmarks experiments tools

format-check:
    uv run ruff format --check src tests benchmarks experiments tools

typecheck:
    uv run basedpyright

check: lint format-check typecheck test

benchmark:
    {{python}} benchmarks/benchmark_optimized.py

benchmark-end-to-end:
    {{python}} benchmarks/bench_end_to_end.py

validate-drug-disease-matrix:
    {{python}} experiments/validate_drug_disease_matrix.py \
        --out-dir reports/drug_disease_validation

compare-demo:
    mkdir -p reports
    {{python}} -m andes.cli compare \
        --emb {{embedding}} \
        --genelist {{genes}} \
        --geneset1 {{gene_sets}} \
        --geneset2 {{gene_sets}} \
        --out reports/compare.npy \
        --cache reports/compare.null

enrich-demo:
    mkdir -p reports
    {{python}} -m andes.cli enrich \
        --emb {{embedding}} \
        --genelist {{genes}} \
        --geneset {{gene_sets}} \
        --rankedlist {{ranked}} \
        --out reports/enrichment.csv

download-geo geo_cache="/tmp/andes-geo":
    {{python}} tools/download_geo_expression.py \
        --geo2kegg paper/data/geo2kegg.txt \
        --out-dir data/expression \
        --cache {{geo_cache}}
