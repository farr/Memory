#!/bin/bash
# Source this script to activate the project environment:
#   source setenv.sh
#
# Sets up:
#   - uv .venv
#   - LAL_DATA_PATH pointing at the local data/ directory
#   - LD_PRELOAD for libgslcblas (required by pyseobnr / pygsl_lite)
#   - NUMBA_CACHE_DIR in /tmp (avoids "no locator available" errors in
#     multiprocessing workers when the home filesystem is slow or NFS-mounted)
#   - Thread-count variables pinned to 1 (numpy/OpenBLAS/MKL/numexpr each
#     try to spawn their own thread pools; pin them so they don't fight)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- virtual environment ---
source "$SCRIPT_DIR/.venv/bin/activate"

# --- waveform data (NRSur7dq4, SEOBNRv4ROM, …) ---
export LAL_DATA_PATH="$SCRIPT_DIR/data${LAL_DATA_PATH:+:$LAL_DATA_PATH}"

# --- pyseobnr / pygsl_lite: needs libgslcblas preloaded on this system ---
export LD_PRELOAD=/lib64/libgslcblas.so.0

# --- numba cache: use a per-session writable directory in /tmp ---
export NUMBA_CACHE_DIR="/tmp/numba_cache_$$"

# --- pin internal thread pools to 1 so multiprocessing works correctly ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
