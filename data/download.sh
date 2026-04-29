#!/bin/sh

# Download with conditional checks:
# -L: follow redirects
# -C -: resume partial downloads automatically
# -z: only download if remote file is newer than local
# -o: specify output filename
# --fail: fail silently on server errors

echo "Downloading posterior_samples_NRSur7dq4.h5..."
curl -L -C - -z posterior_samples_NRSur7dq4.h5 --fail \
  -o posterior_samples_NRSur7dq4.h5 \
  "https://zenodo.org/records/16877102/files/posterior_samples_NRSur7dq4.h5?download=1"

echo "Downloading NRSur7dq4_v1.0.h5..."
curl -L -C - -z NRSur7dq4_v1.0.h5 --fail \
  -o NRSur7dq4_v1.0.h5 \
  "https://zenodo.org/records/14999310/files/NRSur7dq4_v1.0.h5?download=1"

mkdir -p selection

echo "Downloading O3+O4a polar-spin selection file..."
curl -L -C - -z selection/mixture-real_o3_o4a-polar_spins_20250503134659UTC.hdf --fail \
  -o selection/mixture-real_o3_o4a-polar_spins_20250503134659UTC.hdf \
  "https://zenodo.org/records/16740128/files/mixture-real_o3_o4a-polar_spins_20250503134659UTC.hdf?download=1"

# Cumulative O1+O2+O3+O4a (semianalytic for O1+O2, real for O3+O4a) — needed
# to fold pre-O3 BBH events into the hierarchical analysis. The reader treats
# search-pipeline FAR recoveries and semi-analytic SNR recoveries as found.
echo "Downloading cumulative semi-O1+O2 / real-O3+O4a cartesian-spin selection file..."
curl -L -C - -z selection/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf --fail \
  -o selection/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf \
  "https://zenodo.org/records/16740128/files/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf?download=1"

echo "Downloading cumulative semi-O1+O2 / real-O3+O4a polar-spin selection file..."
curl -L -C - -z selection/mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf --fail \
  -o selection/mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf \
  "https://zenodo.org/records/16740128/files/mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf?download=1"

# Verify md5 sums for the cumulative files against the published Zenodo record.
# (The O3+O4a-only files were originally fetched without verification; we only
# guard the new cumulative downloads here so that a failed/truncated transfer
# does not silently land a corrupt selection file.)
echo "Verifying md5 of cumulative selection files..."
cumulative_md5_check() {
    local file="$1" expected="$2"
    if [ ! -f "$file" ]; then
        echo "  MISSING: $file" >&2
        return 1
    fi
    local actual
    actual=$(md5sum "$file" | awk '{print $1}')
    if [ "$actual" = "$expected" ]; then
        echo "  OK   $file"
    else
        echo "  FAIL $file (expected $expected, got $actual)" >&2
        return 1
    fi
}
cumulative_md5_check \
  selection/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf \
  6159fcfc4f8c2ca3f100c8ddb656e584
cumulative_md5_check \
  selection/mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf \
  65ebd98e8914858709603539198f940d

echo "Downloading O4a polar-spin selection file..."
curl -L -C - -z selection/samples-rpo4a_v2_20250503133839UTC-1366933504-23846400.hdf --fail \
  -o selection/samples-rpo4a_v2_20250503133839UTC-1366933504-23846400.hdf \
  "https://zenodo.org/records/16740117/files/samples-rpo4a_v2_20250503133839UTC-1366933504-23846400.hdf?download=1"

# Create symlink if it doesn't exist
if [ ! -L NRSur7dq4.h5 ]; then
  echo "Creating symlink NRSur7dq4.h5 -> NRSur7dq4_v1.0.h5"
  ln -s NRSur7dq4_v1.0.h5 NRSur7dq4.h5
fi

# SEOBNRv4ROM_v3.0.hdf5 is needed for SEOBNRv4_ROM_NRTidalv2_NSBH.
# Source: /mnt/ceph/users/misi/lscsoft/src/lalsuite-waveform-data/waveform_data/SEOBNRv4ROM_v3.0.hdf5
# Also available from: https://git.ligo.org/waveforms/software/lalsuite-waveform-data
# On this system, a symlink data/SEOBNRv4ROM_v3.0.hdf5 -> the ceph copy is used.

echo "Download complete!"
