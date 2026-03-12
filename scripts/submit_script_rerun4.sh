#!/bin/bash -l
#SBATCH -J memory
#SBATCH -p genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=80:00:00
#SBATCH --chdir=/mnt/home/misi/src/Memory

#SBATCH -o results/memory_gwtc/logs/%x_%A_%a.out
#SBATCH -e results/memory_gwtc/logs/%x_%A_%a.err

set -euo pipefail

# pyseobnr / pygsl_lite requires libgslcblas to be preloaded on this system
export LD_PRELOAD=/lib64/libgslcblas.so.0

# pyseobnr uses quaternionic/numba; point cache to a per-job writable directory
# to avoid "no locator available" crashes in multiprocessing workers
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}

EVENT_FILE="scripts/events_rerun4.txt"
OUTDIR="results/memory_gwtc"

mkdir -p "$OUTDIR/logs"

# Slurm array index -> event
EVENT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EVENT_FILE")

echo "Analyzing event: $EVENT"
echo "Node: $(hostname)"

uv run python scripts/compute_gw_memory_for_GWTC.py \
    --output-dir "$OUTDIR" \
    --events "$EVENT" \
    --multiprocess_samples True \
    --max-samples 10
