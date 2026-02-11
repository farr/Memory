#!/bin/sh

curl -O https://zenodo.org/records/16877102/files/posterior_samples_NRSur7dq4.h5?download=1
mv posterior_samples_NRSur7dq4.h5?download=1 posterior_samples_NRSur7dq4.h5
curl -O https://zenodo.org/records/14999310/files/NRSur7dq4_v1.0.h5?download=1
mv NRSur7dq4_v1.0.h5?download=1 NRSur7dq4_v1.0.h5
ln -s NRSur7dq4_v1.0.h5 NRSur7dq4.h5
