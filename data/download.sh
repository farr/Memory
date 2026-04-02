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
