#!/usr/bin/env bash
# Run PI-requested ANDES benchmarks on mochi (no SLURM).
#
# Usage: bash benchmarks/submit_pi_benchmark.sh [workers]
#
# Workers default to nproc. Results go to remote_benchmarks/pi_cases/.

set -euo pipefail

WORKERS=${1:-$(nproc)}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
LOG="$ROOT/logs/pi_benchmark_$(date +%Y%m%d_%H%M%S).log"
PYTHON="${ANDES_PYTHON:-uv run python}"

cd "$ROOT"
mkdir -p logs remote_benchmarks/pi_cases

echo "PI benchmark — $(date)  workers=$WORKERS  log=$LOG"

{
    $PYTHON benchmarks/benchmark_pi_cases.py \
        --workers         "$WORKERS" \
        --query-workers   "$WORKERS" \
        --query-memory-mb 8192 \
        --ite             1000 \
        --out-dir         remote_benchmarks/pi_cases \
        --verbose
} 2>&1 | tee "$LOG"

echo ""
echo "Done. Log saved to $LOG"
