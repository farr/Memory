#!/bin/bash -l
#
# Full-catalog validation run: all 176 events, 10 samples each, fresh output dir.
# Tests fixes for ISCO retry, f_start consistency, and C-stderr capture.
#
# Submit: sbatch --array=0-175 scripts/submit_script_check.sh

#SBATCH -J memory_check
#SBATCH -p genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/home/misi/src/Memory

#SBATCH -o results/memory_gwtc_check/logs/%x_%A_%a.out
#SBATCH -e results/memory_gwtc_check/logs/%x_%A_%a.err

set -euo pipefail

export LD_PRELOAD=/lib64/libgslcblas.so.0
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}

EVENT_FILE="scripts/events.txt"
OUTDIR="results/memory_gwtc_check"

mkdir -p "$OUTDIR/logs"

EVENT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EVENT_FILE")

echo "Analyzing event: $EVENT"
echo "Node: $(hostname)"

uv run python scripts/compute_gw_memory_for_GWTC.py \
    --output-dir "$OUTDIR" \
    --events "$EVENT" \
    --multiprocess_samples True \
    --max-samples 10 \
    --overwrite
