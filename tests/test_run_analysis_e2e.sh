#!/usr/bin/env bash

set -euo pipefail

# End-to-end smoke test for scripts/run_analysis.py using a tiny dataset.
#
# This script can optionally download a small remote subset first (via
# tests/get_test_data.sh), then run run_analysis.py with small MCMC settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/test_e2e}"
PARAMETER="${PARAMETER:-dchi_2}"
MODEL="${MODEL:-both}"
NUM_EVENTS="${NUM_EVENTS:-2}"
DOWNLOAD_IF_MISSING=true
FORCE_DOWNLOAD=false

N_WARMUP="${N_WARMUP:-20}"
N_SAMPLE="${N_SAMPLE:-20}"
N_CHAINS="${N_CHAINS:-1}"
N_SAMPLES_PER_EVENT="${N_SAMPLES_PER_EVENT:-200}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --data-root PATH           Data root (default: ${DATA_ROOT})"
    echo "  --parameter NAME           Test parameter (default: ${PARAMETER})"
    echo "  --model {joint,tgr,both}   Model to run (default: ${MODEL})"
    echo "  --num-events N             Number of posterior files if downloading (default: ${NUM_EVENTS})"
    echo "  --no-download              Do not auto-download data if missing"
    echo "  --force-download           Re-download subset before running"
    echo "  --n-warmup N               Warmup samples (default: ${N_WARMUP})"
    echo "  --n-sample N               Posterior samples (default: ${N_SAMPLE})"
    echo "  --n-chains N               Chains (default: ${N_CHAINS})"
    echo "  --n-samples-per-event N    Posterior samples per event (default: ${N_SAMPLES_PER_EVENT})"
    echo "  -h, --help                 Show help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --parameter)
            PARAMETER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
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

if [[ "${MODEL}" != "joint" && "${MODEL}" != "tgr" && "${MODEL}" != "both" ]]; then
    echo "--model must be one of: joint, tgr, both" >&2
    exit 1
fi

POSTERIOR_GLOB="${DATA_ROOT}/posteriors/*.h5"
INJECTION_FILE="$(python3 -c "import glob, os, sys; files=sorted(glob.glob(os.path.join('${DATA_ROOT}', 'selection', '*.hdf'))); print(files[0] if files else '')")"

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

OUTDIR="${DATA_ROOT}/results_${PARAMETER}_${MODEL}"
mkdir -p "${OUTDIR}"

FIRST_POSTERIOR="$(python3 -c "import glob; files=sorted(glob.glob('${POSTERIOR_GLOB}')); print(files[0] if files else '')")"
if [[ -z "${FIRST_POSTERIOR}" ]]; then
    echo "No posterior files found at ${POSTERIOR_GLOB}" >&2
    exit 1
fi

# Ensure chosen parameter exists; otherwise pick a sensible fallback.
SELECTED_PARAMETER="$PARAMETER"
PARAM_FOUND="$(uv run --with h5py python - <<PY
import h5py
fname = r'''${FIRST_POSTERIOR}'''
param = r'''${PARAMETER}'''
with h5py.File(fname, "r") as f:
    keys = [k for k in f.keys() if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]]
    if not keys:
        print("0")
    else:
        names = f[keys[0]]["posterior_samples"].dtype.names or ()
        print("1" if param in names else "0")
PY
)"

if [[ "${PARAM_FOUND}" != "1" ]]; then
    FALLBACK_PARAM="$(uv run --with h5py python - <<PY
import h5py
fname = r'''${FIRST_POSTERIOR}'''
preferred = ("dchi_2", "dphi_2", "dchi_1", "dphi_1", "chi_eff", "chi_p")
with h5py.File(fname, "r") as f:
    keys = [k for k in f.keys() if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]]
    if not keys:
        print("")
    else:
        names = f[keys[0]]["posterior_samples"].dtype.names or ()
        for p in preferred:
            if p in names:
                print(p)
                break
        else:
            print("")
PY
)"
    if [[ -z "${FALLBACK_PARAM}" ]]; then
        echo "Parameter '${PARAMETER}' not present and no fallback found in ${FIRST_POSTERIOR}" >&2
        exit 1
    fi
    SELECTED_PARAMETER="${FALLBACK_PARAM}"
    echo "Requested parameter '${PARAMETER}' not found; using '${SELECTED_PARAMETER}' for smoke test."
fi

echo "Running end-to-end smoke test"
echo "  parameter: ${SELECTED_PARAMETER}"
echo "  model: ${MODEL}"
echo "  data root: ${DATA_ROOT}"
echo "  output: ${OUTDIR}"

if command -v uv >/dev/null 2>&1; then
    # Provide missing runtime deps without mutating project files.
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

"${RUNNER[@]}" "${REPO_DIR}/scripts/run_analysis.py" \
    "${SELECTED_PARAMETER}" \
    "${POSTERIOR_GLOB}" \
    --injection-file "${INJECTION_FILE}" \
    --model "${MODEL}" \
    --n-warmup "${N_WARMUP}" \
    --n-sample "${N_SAMPLE}" \
    --n-chains "${N_CHAINS}" \
    --n-samples-per-event "${N_SAMPLES_PER_EVENT}" \
    --no-plots \
    --force \
    --outdir "${OUTDIR}"

echo ""
echo "Smoke test completed."
echo "Results in: ${OUTDIR}"
