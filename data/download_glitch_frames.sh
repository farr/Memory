#!/bin/sh
# Download BayesWave glitch-subtracted GWF frames from Zenodo 16857060.
#
# These frames are needed for GWTC-4.0 events where BayesWave glitch
# subtraction was applied during PE.  Pass the download directory to
# compute_gw_memory_for_GWTC.py via --frame-dir.
#
# Usage:
#   ./download_glitch_frames.sh [OUTDIR]
#
# OUTDIR defaults to the directory containing this script (data/).
#
# After downloading, inspect the available channels in one file with:
#   python -c "
#   from gwpy.io.gwf import iter_channel_names
#   for ch in iter_channel_names('H-H1_HOFT_C00_BAYESWAVE_S00-1370046464-4096.gwf'):
#       print(ch)
#   "
# and update --glitch-channel-format in your run command if needed.

ZENODO_BASE="https://zenodo.org/records/16857060/files"
OUTDIR="${1:-$(dirname "$0")}"
mkdir -p "$OUTDIR"

FILES="
H-H1_HOFT_C00_BAYESWAVE_S00-1370046464-4096.gwf
H-H1_HOFT_C00_BAYESWAVE_S00-1372766208-4096.gwf
H-H1_HOFT_C00_BAYESWAVE_S00-1372827648-4096.gwf
H-H1_HOFT_C00_AR_BAYESWAVE_S00-1375387648-4096.gwf
L-L1_HOFT_C00_BAYESWAVE_S00-1376497664-4096.gwf
H-H1_HOFT_C00_BAYESWAVE_S00-1383968768-4096.gwf
L-L1_HOFT_C00_BAYESWAVE_S00-1383911424-4096.gwf
H-H1_HOFT_C00_BAYESWAVE_S00-1384333312-4096.gwf
H-H1_HOFT_C00_BAYESWAVE_S00-1384779776-4096.gwf
L-L1_HOFT_C00_BAYESWAVE_S00-1385279488-4096.gwf
H-H1_HOFT_C00_AR_BAYESWAVE_S00-1387200512-4096.gwf
H-H1_HOFT_C00_BAYESWAVE_S00-1387335680-4096.gwf
"

for fname in $FILES; do
    out="$OUTDIR/$fname"
    echo "Downloading $fname ..."
    curl -L -C - -z "$out" --fail \
        -o "$out" \
        "${ZENODO_BASE}/${fname}?download=1"
done

echo "Done. Frame files written to $OUTDIR"
echo ""
echo "To verify channel names in a downloaded file, run:"
echo "  python -c \""
echo "  from gwpy.io.gwf import iter_channel_names"
echo "  for ch in iter_channel_names('$OUTDIR/H-H1_HOFT_C00_BAYESWAVE_S00-1370046464-4096.gwf'):"
echo "      print(ch)"
echo "  \""
