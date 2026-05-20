#!/usr/bin/env bash
# Run all four ANDES/GSEA benchmarks (new and old) sequentially, then verify
# that z-scores from new and old pipelines agree (Spearman ρ ≥ 0.90).
# Output is tee'd to a timestamped log file so you can watch live and review later.
#
# Usage:
#   bash benchmarks/run_benchmarks.sh [workers]
#
# Default workers: 32. Override: bash benchmarks/run_benchmarks.sh 8

set -euo pipefail

WORKERS=${1:-32}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS="$ROOT/reports"
LOG="$REPORTS/bench_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$REPORTS"

EMB="$ROOT/data/embedding/node2vec_consensus.csv"
GENELIST="$ROOT/data/embedding/consensus_node.txt"
GMT="$ROOT/data/gene_sets/hsa_experimental_eval_BP_propagated.gmt"
RANKEDLIST="$ROOT/data/expression/GSE3467_rank.txt"

PYTHON="${ANDES_PYTHON:-uv run python}"
BENCH="$ROOT/benchmarks/bench_end_to_end.py"
COMPARE="$ROOT/benchmarks/compare_scores.py"

run() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $label  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "════════════════════════════════════════════════════════════"
    "$@"
    echo ""
    echo "  $label finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

{
echo "ANDES benchmark suite — $(date)"
echo "Workers: $WORKERS"
echo "Host:    $(hostname)"
echo ""

run "andes (new)" $PYTHON "$BENCH" andes \
    --emb "$EMB" --genelist "$GENELIST" \
    --geneset1 "$GMT" --geneset2 "$GMT" \
    --ite 1000 --workers "$WORKERS" \
    --query-workers "$WORKERS" --query-mode batched \
    --query-memory-mb 32768 \
    --out "$REPORTS/andes_new_scores.csv" \
    --json-out "$REPORTS/andes_new.json" \
    --verbose

run "gsea (new)" $PYTHON "$BENCH" gsea \
    --emb "$EMB" --genelist "$GENELIST" \
    --geneset "$GMT" --rankedlist "$RANKEDLIST" \
    --ite 1000 --workers "$WORKERS" \
    --out "$REPORTS/gsea_new_scores.csv" \
    --json-out "$REPORTS/gsea_new.json" \
    --verbose

run "andes_old" $PYTHON "$BENCH" andes_old \
    --emb "$EMB" --genelist "$GENELIST" \
    --geneset1 "$GMT" --geneset2 "$GMT" \
    --ite 1000 --workers "$WORKERS" \
    --out "$REPORTS/andes_old_scores.csv" \
    --json-out "$REPORTS/andes_old.json" \
    --verbose

run "gsea_old" $PYTHON "$BENCH" gsea_old \
    --emb "$EMB" --genelist "$GENELIST" \
    --geneset "$GMT" --rankedlist "$RANKEDLIST" \
    --ite 1000 --workers "$WORKERS" \
    --out "$REPORTS/gsea_old_scores.csv" \
    --json-out "$REPORTS/gsea_old.json" \
    --verbose

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Z-SCORE AGREEMENT CHECK  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "════════════════════════════════════════════════════════════"
$PYTHON "$COMPARE" \
    --andes-new "$REPORTS/andes_new_scores.csv" \
    --andes-old "$REPORTS/andes_old_scores.csv" \
    --gsea-new  "$REPORTS/gsea_new_scores.csv"  \
    --gsea-old  "$REPORTS/gsea_old_scores.csv"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All benchmarks complete  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "════════════════════════════════════════════════════════════"

} 2>&1 | tee "$LOG"

echo ""
echo "Log saved to $LOG"
