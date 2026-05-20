remote_host := "mochi"
remote_dir := "/home/cjh16/ANDES"
local_dir := justfile_directory()
user := "cjh16"
python     := env_var_or_default("ANDES_PYTHON", "uv run python")
remote_fish := env_var_or_default("ANDES_FISH", "/home/cjh16/.local/bin/fish")
excludes := "--exclude-from=" + local_dir + "/.rsyncignore"

query_gmt := "/grain/rad4/github/ANDES/rd_test_data/test_cases.gmt"
go_bp_gmt := "/grain/resources/gene_ontology/output/2025-03-16/hsa_ALL_BP_direct.gmt"
drugbank_gmt := "/grain/ll84/setmatch/data/gmt/drugbank.2301.gmt"
kegg_cpdb_gmt := "/grain/ll84/setmatch/data/gmt/KEGG_CPDB.gmt"
omim_gmt := "/grain/ll84/setmatch/data/gmt/omim.20231030.prop.gmt"

emb := "data/embedding/node2vec_consensus.csv"
genelist := "data/embedding/consensus_node.txt"
remote_results := "remote_results"
remote_cache := "remote_cache"
remote_benchmarks := "remote_benchmarks"
workers := "32"
query_memory_mb := "32768"
ite := "1000"

# end-to-end benchmark defaults
bench_gmt := "data/gene_sets/hsa_experimental_eval_BP_propagated.gmt"
kegg_gmt  := "paper/results/enrichment_analysis/GSEA/gene_sets.gmt"
reports   := "reports"

# List available commands
default:
    just --list

# ── Download GEO expression data ──────────────────────────────────────────────

# Download all GEO expression datasets and generate _rank.txt files (run on server)
download-geo-here geo_cache="/tmp/geo_cache":
    mkdir -p data/expression
    {{python}} src/download_geo_expression.py \
        --geo2kegg paper/data/geo2kegg.txt \
        --out-dir  data/expression \
        --cache    {{geo_cache}}

# Download a single GEO dataset (for testing or re-running failures)
download-geo-one-here geo geo_cache="/tmp/geo_cache":
    mkdir -p data/expression
    {{python}} src/download_geo_expression.py \
        --geo2kegg paper/data/geo2kegg.txt \
        --out-dir  data/expression \
        --cache    {{geo_cache}} \
        --geos     {{geo}}

# Download all GEO datasets remotely
download-geo geo_cache="/tmp/geo_cache":
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just download-geo-here {{geo_cache}}"'

# Pull code/scripts from mochi, excluding generated data/results
pull:
    rsync -avz {{remote_host}}:{{remote_dir}}/ {{local_dir}}/ {{excludes}}

# Dry-run push so you can inspect what would change
push-dry:
    rsync -avzn {{local_dir}}/ {{remote_host}}:{{remote_dir}}/ {{excludes}}

# Push local code/scripts to mochi
push:
    rsync -avz {{local_dir}}/ {{remote_host}}:{{remote_dir}}/ {{excludes}}

# Push with deletion (destructive on mochi). Use push-dry first to inspect.
push-delete:
    rsync -avz --delete {{local_dir}}/ {{remote_host}}:{{remote_dir}}/ {{excludes}}

# Dry-run of push-delete — shows what would be added, updated, or removed.
push-delete-dry:
    rsync -avzn --delete {{local_dir}}/ {{remote_host}}:{{remote_dir}}/ {{excludes}}

# Dry-run: show what pull-delete would change locally.
pull-delete-dry:
    rsync -avzn --delete {{remote_host}}:{{remote_dir}}/ {{local_dir}}/ {{excludes}}

# Pull from remote with deletion (destructive locally). Use when mochi is source of truth.
pull-delete:
    rsync -avz --delete {{remote_host}}:{{remote_dir}}/ {{local_dir}}/ {{excludes}}

# Run the local unit tests
test:
    {{python}} -m unittest discover -s tests

# Create/update the uv environment
uv-sync:
    uv sync

# Run the unit tests remotely in /home/cjh16/ANDES
remote-test:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && {{python}} -m unittest discover -s tests"'

# Open a shell in the remote project dir
ssh:
    ssh -t {{remote_host}} 'cd {{remote_dir}} && exec $$SHELL'

# Check your Slurm queue
queue:
    ssh {{remote_host}} 'squeue -u {{user}}'

# Watch queue, refresh every 5 seconds
watch-queue:
    ssh -t {{remote_host}} 'watch -n 5 squeue -u {{user}}'

# Submit a Slurm job remotely
submit job:
    ssh {{remote_host}} 'cd {{remote_dir}} && sbatch {{job}}'

# Run an ANDES query-vs-background test remotely. Pass a GMT, output CSV, and cache PKL.
andes bg out cache workers=workers:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && mkdir -p {{remote_results}} {{remote_cache}} && {{python}} src/andes.py --emb {{emb}} --genelist {{genelist}} --geneset1 {{query_gmt}} --geneset2 {{bg}} --out {{out}} --cache {{cache}} --ite {{ite}} --workers {{workers}} --query-workers {{workers}} --query-mode batched --query-memory-mb {{query_memory_mb}} --verbose"'

# Test query GMT against GO BP direct background
andes-go-bp:
    just andes {{go_bp_gmt}} {{remote_results}}/test_cases_vs_go_bp.csv {{remote_cache}}/test_cases_vs_go_bp.pkl {{workers}}

# Test query GMT against DrugBank
andes-drugbank:
    just andes {{drugbank_gmt}} {{remote_results}}/test_cases_vs_drugbank.csv {{remote_cache}}/test_cases_vs_drugbank.pkl {{workers}}

# Test query GMT against KEGG/CPDB
andes-kegg-cpdb:
    just andes {{kegg_cpdb_gmt}} {{remote_results}}/test_cases_vs_kegg_cpdb.csv {{remote_cache}}/test_cases_vs_kegg_cpdb.pkl {{workers}}

# Test query GMT against OMIM propagated
andes-omim:
    just andes {{omim_gmt}} {{remote_results}}/test_cases_vs_omim.csv {{remote_cache}}/test_cases_vs_omim.pkl {{workers}}

# Run all four query-vs-background tests on mochi
andes-all: andes-go-bp andes-drugbank andes-kegg-cpdb andes-omim

# Run a small remote smoke benchmark over all PI cases
pi-benchmark-smoke:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && {{python}} benchmarks/benchmark_pi_cases.py --out-dir {{remote_benchmarks}}/pi_cases_smoke --ite {{ite}} --workers 16 --query-workers 16 --query-memory-mb 4096 --limit-queries 5 --limit-background 50 --verbose"'

# Run a small smoke benchmark in the current shell, for when you are already on mochi
pi-benchmark-smoke-here:
    {{python}} benchmarks/benchmark_pi_cases.py --out-dir {{remote_benchmarks}}/pi_cases_smoke --ite {{ite}} --workers 16 --query-workers 16 --query-memory-mb 4096 --limit-queries 5 --limit-background 50 --verbose

# Run the full PI benchmark suite remotely
pi-benchmark:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && {{python}} benchmarks/benchmark_pi_cases.py --out-dir {{remote_benchmarks}}/pi_cases --ite {{ite}} --workers {{workers}} --query-workers {{workers}} --query-memory-mb {{query_memory_mb}} --verbose"'

# Run the full PI benchmark suite in the current shell, for when you are already on mochi
pi-benchmark-here:
    {{python}} benchmarks/benchmark_pi_cases.py --out-dir {{remote_benchmarks}}/pi_cases --ite {{ite}} --workers {{workers}} --query-workers {{workers}} --query-memory-mb {{query_memory_mb}} --verbose

# Run one PI benchmark case remotely: go_bp, drugbank, kegg_cpdb, or omim
pi-benchmark-one case:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && {{python}} benchmarks/benchmark_pi_cases.py --case {{case}} --out-dir {{remote_benchmarks}}/pi_cases --ite {{ite}} --workers {{workers}} --query-workers {{workers}} --query-memory-mb {{query_memory_mb}} --verbose"'

# Run one PI benchmark case in the current shell: go_bp, drugbank, kegg_cpdb, or omim
pi-benchmark-one-here case:
    {{python}} benchmarks/benchmark_pi_cases.py --case {{case}} --out-dir {{remote_benchmarks}}/pi_cases --ite {{ite}} --workers {{workers}} --query-workers {{workers}} --query-memory-mb {{query_memory_mb}} --verbose

# Submit the full PI benchmark suite as a Slurm job
submit-pi-benchmark:
    ssh {{remote_host}} 'cd {{remote_dir}} && mkdir -p logs && sbatch benchmarks/submit_pi_benchmark.sh'

# Push code, then run a small PI benchmark smoke test
push-pi-smoke: push pi-benchmark-smoke

# Push code, then submit full PI benchmark to Slurm
push-submit-pi-benchmark: push submit-pi-benchmark

# Push code, then run the GO BP remote test
push-andes-go-bp: push andes-go-bp

# Pull remote result CSVs and logs incrementally
pull-results:
    mkdir -p {{local_dir}}/remote_results {{local_dir}}/logs
    rsync -avz --update {{remote_host}}:{{remote_dir}}/{{remote_results}}/ {{local_dir}}/remote_results/
    rsync -avz --update {{remote_host}}:{{remote_dir}}/logs/ {{local_dir}}/logs/

# Pull PI benchmark summaries, JSON reports, score matrices, and logs
pull-benchmarks:
    mkdir -p {{local_dir}}/{{remote_benchmarks}} {{local_dir}}/logs
    rsync -avz --update {{remote_host}}:{{remote_dir}}/{{remote_benchmarks}}/ {{local_dir}}/{{remote_benchmarks}}/
    rsync -avz --update {{remote_host}}:{{remote_dir}}/logs/ {{local_dir}}/logs/

# Pull remote cache PKLs. These may be large.
pull-caches:
    mkdir -p {{local_dir}}/remote_cache
    rsync -avz --update {{remote_host}}:{{remote_dir}}/{{remote_cache}}/ {{local_dir}}/remote_cache/

# ── end-to-end benchmarks ────────────────────────────────────────────────────

# Run ANDES set-vs-set benchmark (already on server / in current shell)
bench-andes-here workers="8":
    mkdir -p {{reports}}
    {{python}} benchmarks/bench_end_to_end.py andes \
        --emb {{emb}} \
        --genelist {{genelist}} \
        --geneset1 {{bench_gmt}} \
        --geneset2 {{bench_gmt}} \
        --ite {{ite}} \
        --workers {{workers}} \
        --query-workers {{workers}} \
        --query-mode batched \
        --query-memory-mb {{query_memory_mb}} \
        --cache {{reports}}/andes_cache.pkl \
        --json-out {{reports}}/andes.json \
        --profile-out {{reports}}/andes.prof \
        --verbose

# Run ANDES set-vs-set benchmark remotely via ssh
bench-andes workers="8":
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just bench-andes-here {{workers}}"'

bench-andes-smoke-here:
    just bench-andes-here 4

bench-andes-smoke:
    just bench-andes 4

# Run ANDES-GSEA benchmark on a single ranked list (already on server / in current shell)
# Usage: just bench-gsea-here rankedlist=data/expression/GSE3467_rank.txt
bench-gsea-here rankedlist workers=workers:
    mkdir -p {{reports}}
    {{python}} benchmarks/bench_end_to_end.py gsea \
        --emb {{emb}} \
        --genelist {{genelist}} \
        --geneset {{kegg_gmt}} \
        --rankedlist {{rankedlist}} \
        --ite {{ite}} \
        --workers {{workers}} \
        --cache {{reports}}/gsea_cache.pkl \
        --json-out {{reports}}/gsea.json \
        --profile-out {{reports}}/gsea.prof \
        --verbose

# Run ANDES-GSEA benchmark remotely
bench-gsea rankedlist workers=workers:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just bench-gsea-here {{rankedlist}} {{workers}}"'

# ── GEO2KEGG benchmark (GSEA-ANDES) ─────────────────────────────────────────

# Run GEO2KEGG benchmark (already on server / in current shell)
bench-geo2kegg-here workers=workers:
    mkdir -p {{reports}}
    {{python}} benchmarks/bench_geo2kegg.py \
        --ite {{ite}} \
        --workers {{workers}} \
        --out {{reports}}/geo2kegg.csv \
        --verbose

# Run GEO2KEGG benchmark on a single GEO dataset (quick sanity check)
bench-geo2kegg-one-here geo workers=workers:
    mkdir -p {{reports}}
    {{python}} benchmarks/bench_geo2kegg.py \
        --geos {{geo}} \
        --ite {{ite}} \
        --workers {{workers}} \
        --out {{reports}}/geo2kegg_{{geo}}.csv \
        --verbose

bench-geo2kegg-smoke-here:
    just bench-geo2kegg-one-here GSE3467 4

# Remote variants
bench-geo2kegg workers=workers:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just bench-geo2kegg-here {{workers}}"'

bench-geo2kegg-one geo workers=workers:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just bench-geo2kegg-one-here {{geo}} {{workers}}"'

bench-geo2kegg-smoke:
    just bench-geo2kegg-one GSE3467 4

# ── Speedup benchmarks ───────────────────────────────────────────────────────

# Smoke: quick old-vs-new comparison on small synthetic data
bench-speedup-smoke-here:
    mkdir -p {{reports}}
    {{python}} benchmarks/bench_speedup.py \
        --n-genes 500 --dim 64 --n-terms 20 --ite {{ite}} --workers 4 \
        --json-out {{reports}}/speedup_smoke.json

# Full speedup benchmark (realistic problem size, takes several minutes)
bench-speedup-here n_genes="3000" workers=workers:
    mkdir -p {{reports}}
    {{python}} benchmarks/bench_speedup.py \
        --n-genes {{n_genes}} --dim 128 --n-terms 80 --n-pairs 300 \
        --ite {{ite}} --workers {{workers}} \
        --json-out {{reports}}/speedup.json

# Remote variants
bench-speedup-smoke:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just bench-speedup-smoke-here"'

bench-speedup n_genes="3000" workers=workers:
    ssh {{remote_host}} '{{remote_fish}} -l -c "cd {{remote_dir}} && just bench-speedup-here {{n_genes}} {{workers}}"'

# Push code then run each benchmark suite
push-bench-andes: push bench-andes
push-bench-geo2kegg: push bench-geo2kegg
push-bench-speedup: push bench-speedup
