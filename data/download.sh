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
