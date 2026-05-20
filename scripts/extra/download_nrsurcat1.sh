#!/usr/bin/env bash
set -euo pipefail

RECORD_ID="${RECORD_ID:-8115310}"
DEST_DIR="${1:-/mnt/home/ccalvk/ceph/NRSurCat-1}"
API_URL="https://zenodo.org/api/records/${RECORD_ID}"

mkdir -p "${DEST_DIR}"

echo "Downloading Zenodo record ${RECORD_ID} into ${DEST_DIR}"
echo "Fetching file manifest from ${API_URL}"

python3 - "${API_URL}" <<'PY' |
import json
import sys
from urllib.request import urlopen

api_url = sys.argv[1]
with urlopen(api_url, timeout=60) as response:
    record = json.load(response)

for entry in record["files"]:
    filename = entry["key"]
    link = entry["links"]["self"]
    checksum = entry.get("checksum", "")
    size = entry.get("size", "")
    print(f"{filename}\t{link}\t{checksum}\t{size}")
PY
while IFS=$'\t' read -r filename url checksum size; do
    target="${DEST_DIR}/${filename}"
    echo
    echo "==> ${filename} (${size} bytes)"
    if [[ -n "${checksum}" ]]; then
        echo "Expected ${checksum}"
    fi

    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 10 \
        --retry-delay 30 \
        --retry-connrefused \
        --output "${target}" \
        "${url}"
done

echo
echo "Finished downloading Zenodo record ${RECORD_ID} into ${DEST_DIR}"
