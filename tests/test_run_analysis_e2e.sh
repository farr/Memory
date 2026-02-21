#!/usr/bin/env bash

set -euo pipefail

# End-to-end smoke test for scripts/run_hierarchical_analysis.py using a tiny dataset.
#
# This script can optionally download a small remote subset first (via
# tests/get_test_data.sh), then run run_hierarchical_analysis.py with small MCMC settings.
#
# Available analyses (--analyze):
#   astro  — astrophysical population only; does NOT require --memory-dir
#   memory — TGR-only model; requires --memory-dir
#   joint  — astrophysical + TGR; requires --memory-dir
#
# Default: astro only (no memory results needed for a basic smoke test).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/test_e2e}"
ANALYZE="${ANALYZE:-astro}"
NUM_EVENTS="${NUM_EVENTS:-2}"
DOWNLOAD_IF_MISSING=true
FORCE_DOWNLOAD=false

N_WARMUP="${N_WARMUP:-20}"
N_SAMPLE="${N_SAMPLE:-20}"
N_CHAINS="${N_CHAINS:-1}"
N_SAMPLES_PER_EVENT="${N_SAMPLES_PER_EVENT:-200}"

MEMORY_DIR=""
PARAM_KEY=""

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --data-root PATH              Data root (default: ${DATA_ROOT})"
    echo "  --analyze ANALYSIS...         Analyses to run: astro, memory, joint (default: ${ANALYZE})"
    echo "  --param-key KEY               HDF5 group name filter for posteriors"
    echo "  --memory-dir PATH             Directory with per-event memory results"
    echo "                                (required for --analyze memory or joint)"
    echo "  --num-events N                Number of posterior files if downloading (default: ${NUM_EVENTS})"
    echo "  --no-download                 Do not auto-download data if missing"
    echo "  --force-download              Re-download subset before running"
    echo "  --n-warmup N                  Warmup samples (default: ${N_WARMUP})"
    echo "  --n-sample N                  Posterior samples (default: ${N_SAMPLE})"
    echo "  --n-chains N                  Chains (default: ${N_CHAINS})"
    echo "  --n-samples-per-event N       Posterior samples per event (default: ${N_SAMPLES_PER_EVENT})"
    echo "  -h, --help                    Show help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --analyze)
            ANALYZE="$2"
            shift 2
            ;;
        --param-key)
            PARAM_KEY="$2"
            shift 2
            ;;
        --memory-dir)
            MEMORY_DIR="$2"
            shift 2
            ;;
        --num-events)
            NUM_EVENTS="$2"
            shift 2
            ;;
        --no-download)
            DOWNLOAD_IF_MISSING=false
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --n-warmup)
            N_WARMUP="$2"
            shift 2
            ;;
        --n-sample)
            N_SAMPLE="$2"
            shift 2
            ;;
        --n-chains)
            N_CHAINS="$2"
            shift 2
            ;;
        --n-samples-per-event)
            N_SAMPLES_PER_EVENT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate --analyze values
for a in ${ANALYZE}; do
    if [[ "${a}" != "astro" && "${a}" != "memory" && "${a}" != "joint" ]]; then
        echo "--analyze must be one or more of: astro, memory, joint" >&2
        exit 1
    fi
done

# memory and joint require --memory-dir
if echo "${ANALYZE}" | grep -qE "(memory|joint)"; then
    if [[ -z "${MEMORY_DIR}" ]]; then
        echo "--memory-dir is required when analyzing 'memory' or 'joint'" >&2
        exit 1
    fi
fi

POSTERIOR_GLOB="${DATA_ROOT}/posteriors/*.h5"
INJECTION_FILE="$(python3 -c "import glob, os; files=sorted(glob.glob(os.path.join('${DATA_ROOT}', 'selection', '*.hdf'))); print(files[0] if files else '')")"

if [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
    "${SCRIPT_DIR}/get_test_data.sh" --num-events "${NUM_EVENTS}" --outdir "${DATA_ROOT}"
elif [[ -z "${INJECTION_FILE}" || -z "$(python3 -c "import glob; print('x' if glob.glob('${POSTERIOR_GLOB}') else '')")" ]]; then
    if [[ "${DOWNLOAD_IF_MISSING}" == "true" ]]; then
        "${SCRIPT_DIR}/get_test_data.sh" --num-events "${NUM_EVENTS}" --outdir "${DATA_ROOT}"
        INJECTION_FILE="$(python3 -c "import glob, os; files=sorted(glob.glob(os.path.join('${DATA_ROOT}', 'selection', '*.hdf'))); print(files[0])")"
    else
        echo "Missing dataset in ${DATA_ROOT} and --no-download was provided." >&2
        exit 1
    fi
fi

OUTDIR="${DATA_ROOT}/results_$(echo "${ANALYZE}" | tr ' ' '_')"
mkdir -p "${OUTDIR}"

FIRST_POSTERIOR="$(python3 -c "import glob; files=sorted(glob.glob('${POSTERIOR_GLOB}')); print(files[0] if files else '')")"
if [[ -z "${FIRST_POSTERIOR}" ]]; then
    echo "No posterior files found at ${POSTERIOR_GLOB}" >&2
    exit 1
fi

echo "Running end-to-end smoke test"
echo "  analyze: ${ANALYZE}"
echo "  data root: ${DATA_ROOT}"
echo "  output: ${OUTDIR}"

if command -v uv >/dev/null 2>&1; then
    UV_WITH_DEFAULT="jax numpyro arviz corner"
    UV_WITH="${UV_WITH:-${UV_WITH_DEFAULT}}"
    RUNNER=(uv run)
    for dep in ${UV_WITH}; do
        RUNNER+=(--with "${dep}")
    done
    RUNNER+=(python)
else
    RUNNER=(python)
fi

# Use CPU by default for portability in CI/local test environments.
export TGRPOP_PLATFORM="${TGRPOP_PLATFORM:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

EXTRA_ARGS=()
if [[ -n "${PARAM_KEY}" ]]; then
    EXTRA_ARGS+=(--param-key "${PARAM_KEY}")
fi
if [[ -n "${MEMORY_DIR}" ]]; then
    EXTRA_ARGS+=(--memory-dir "${MEMORY_DIR}")
fi

"${RUNNER[@]}" "${REPO_DIR}/scripts/run_hierarchical_analysis.py" \
    "${POSTERIOR_GLOB}" \
    --injection-file "${INJECTION_FILE}" \
    --analyze ${ANALYZE} \
    --n-warmup "${N_WARMUP}" \
    --n-sample "${N_SAMPLE}" \
    --n-chains "${N_CHAINS}" \
    --n-samples-per-event "${N_SAMPLES_PER_EVENT}" \
    --no-plots \
    --force \
    --outdir "${OUTDIR}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Smoke test completed."
echo "Results in: ${OUTDIR}"
