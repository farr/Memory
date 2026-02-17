#!/usr/bin/env bash

set -euo pipefail

# Download a small subset of remote data for quick end-to-end tests.
#
# Defaults mirror the upstream tgrpop scripts but intentionally fetch only a
# handful of posterior files.

REMOTE_USER="${REMOTE_USER:-max.isi}"
REMOTE_HOST="${REMOTE_HOST:-ldas-grid.ligo.caltech.edu}"
REMOTE_POSTERIOR_ROOT="${REMOTE_POSTERIOR_ROOT:-/home/tgr.o4/O4a/analyses/FTI/asimov/results}"
REMOTE_INJECTION_PATH="${REMOTE_INJECTION_PATH:-/home/rp.o4/offline-injections/mixtures/multirun-mixtures_20250503134659UTC/mixture-real_o4a/mixture-real_o4a-cartesian_spins_20250503134659UTC.hdf}"

NUM_EVENTS="${NUM_EVENTS:-2}"
OUTDIR="${OUTDIR:-data/test_e2e}"
EVENT_GLOB="${EVENT_GLOB:-*}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --num-events N      Number of posterior files to download (default: ${NUM_EVENTS})"
    echo "  --outdir PATH       Output data root (default: ${OUTDIR})"
    echo "  --event-glob GLOB   Filter by path substring on remote file list (default: ${EVENT_GLOB})"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment overrides:"
    echo "  REMOTE_USER, REMOTE_HOST, REMOTE_POSTERIOR_ROOT, REMOTE_INJECTION_PATH"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-events)
            NUM_EVENTS="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --event-glob)
            EVENT_GLOB="$2"
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

if ! [[ "${NUM_EVENTS}" =~ ^[0-9]+$ ]] || [[ "${NUM_EVENTS}" -lt 1 ]]; then
    echo "--num-events must be a positive integer" >&2
    exit 1
fi

for cmd in ssh scp python3; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Missing required command: ${cmd}" >&2
        exit 1
    fi
done

POST_DIR="${OUTDIR}/posteriors"
SEL_DIR="${OUTDIR}/selection"
mkdir -p "${POST_DIR}" "${SEL_DIR}"

echo "Listing remote posterior files..."
REMOTE_LIST_RAW="$(mktemp)"
REMOTE_LIST_FILTERED="$(mktemp)"
cleanup() {
    rm -f "${REMOTE_LIST_RAW}" "${REMOTE_LIST_FILTERED}"
}
trap cleanup EXIT

if ! ssh "${REMOTE_USER}@${REMOTE_HOST}" \
        "python3 - <<'PY'
import os
root = '${REMOTE_POSTERIOR_ROOT}'
for dirpath, _, filenames in os.walk(root):
    if 'posterior_samples.h5' in filenames:
        print(os.path.join(dirpath, 'posterior_samples.h5'))
PY" > "${REMOTE_LIST_RAW}"; then
    echo "Failed to query ${REMOTE_USER}@${REMOTE_HOST} via SSH." >&2
    echo "Check SSH credentials/network or set REMOTE_USER/REMOTE_HOST." >&2
    exit 1
fi

python3 -c "import fnmatch, pathlib; files=sorted(l.strip() for l in pathlib.Path('${REMOTE_LIST_RAW}').read_text().splitlines() if l.strip()); pat='${EVENT_GLOB}'; print('\n'.join(f for f in files if fnmatch.fnmatch(f, f'*{pat}*')))" > "${REMOTE_LIST_FILTERED}"

REMOTE_FILES=()
while IFS= read -r line; do
    [[ -n "${line}" ]] && REMOTE_FILES+=("${line}")
done < "${REMOTE_LIST_FILTERED}"

if [[ "${#REMOTE_FILES[@]}" -eq 0 ]]; then
    echo "No remote posterior files found matching event glob '${EVENT_GLOB}'" >&2
    exit 1
fi

if [[ "${#REMOTE_FILES[@]}" -lt "${NUM_EVENTS}" ]]; then
    echo "Only found ${#REMOTE_FILES[@]} matching files; proceeding with all of them."
    NUM_EVENTS="${#REMOTE_FILES[@]}"
fi

SELECTED_FILES=("${REMOTE_FILES[@]:0:${NUM_EVENTS}}")

echo "Downloading ${#SELECTED_FILES[@]} posterior files to ${POST_DIR}..."
for remote_file in "${SELECTED_FILES[@]}"; do
    rel="${remote_file#${REMOTE_POSTERIOR_ROOT}/}"
    local_name="${rel//\//__}"
    scp "${REMOTE_USER}@${REMOTE_HOST}:${remote_file}" "${POST_DIR}/${local_name}"
done

injection_name="$(basename "${REMOTE_INJECTION_PATH}")"
echo "Downloading injection file ${injection_name}..."
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_INJECTION_PATH}" "${SEL_DIR}/${injection_name}"

EVENT_LIST_FILE="${OUTDIR}/event_files.txt"
INJECTION_FILE="${SEL_DIR}/${injection_name}"
printf "%s\n" "${POST_DIR}"/*.h5 > "${EVENT_LIST_FILE}"
printf "%s\n" "${INJECTION_FILE}" > "${OUTDIR}/injection_file.txt"

echo ""
echo "Small test dataset ready."
echo "  Posterior files: ${#SELECTED_FILES[@]}"
echo "  Event list:      ${EVENT_LIST_FILE}"
echo "  Injection file:  ${INJECTION_FILE}"
