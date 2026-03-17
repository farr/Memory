import h5py
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
import bilby

from memory.hierarchical.data import validate_posterior_prior_consistency


def _write_test_group(path, *, include_log_prior=True, redshift_shift=0.0):
    cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065, name="Planck15_LAL")
    prior = bilby.gw.prior.UniformSourceFrame(
        minimum=10.0,
        maximum=5_000.0,
        cosmology=cosmology,
        name="luminosity_distance",
    )

    z = np.array([0.05, 0.15, 0.3, 0.45])
    d_l = np.array([cosmology.luminosity_distance(zi).value for zi in z])
    m1_source = np.array([20.0, 25.0, 30.0, 35.0])
    m2_source = np.array([15.0, 18.0, 20.0, 22.0])

    dtype = [
        ("luminosity_distance", "f8"),
        ("redshift", "f8"),
        ("mass_1", "f8"),
        ("mass_1_source", "f8"),
        ("mass_2", "f8"),
        ("mass_2_source", "f8"),
    ]
    if include_log_prior:
        dtype.append(("log_prior", "f8"))

    posterior_samples = np.zeros(len(z), dtype=dtype)
    posterior_samples["luminosity_distance"] = d_l
    posterior_samples["redshift"] = z + redshift_shift
    posterior_samples["mass_1"] = m1_source * (1.0 + z)
    posterior_samples["mass_1_source"] = m1_source
    posterior_samples["mass_2"] = m2_source * (1.0 + z)
    posterior_samples["mass_2_source"] = m2_source
    if include_log_prior:
        posterior_samples["log_prior"] = np.linspace(-5.0, -3.5, len(z))

    with h5py.File(path, "w") as f:
        group = f.create_group("C00:Mixed")
        group.create_dataset("posterior_samples", data=posterior_samples)
        group.create_dataset(
            "description",
            data=np.bytes_(
                "Reweighted posterior and prior samples from Planck15 "
                "cosmology to Planck15_LAL cosmology."
            ),
        )

        meta_data = group.create_group("meta_data")
        reweighting = meta_data.create_group("reweighting")
        reweighting.create_dataset(
            "new_metafile",
            data=np.bytes_("samples/Prod0/posterior_samples_cosmo.h5"),
        )
        reweighting.create_dataset(
            "new_cosmology",
            data=np.bytes_("Planck15_LAL"),
        )

        priors = group.create_group("priors")
        analytic = priors.create_group("analytic")
        analytic.create_dataset(
            "luminosity_distance",
            data=np.bytes_(repr(prior)),
        )


def test_validate_posterior_prior_consistency_accepts_matching_cosmo_samples(tmp_path):
    path = tmp_path / "matching.hdf5"
    _write_test_group(path)

    with h5py.File(path, "r") as f:
        group = f["C00:Mixed"]
        posterior_samples = group["posterior_samples"][()]
        validate_posterior_prior_consistency(
            group,
            posterior_samples,
            filename=str(path),
            label="C00:Mixed",
        )


def test_validate_posterior_prior_consistency_rejects_missing_prior_column(tmp_path):
    path = tmp_path / "missing_prior.hdf5"
    _write_test_group(path, include_log_prior=False)

    with h5py.File(path, "r") as f:
        group = f["C00:Mixed"]
        posterior_samples = group["posterior_samples"][()]
        with pytest.raises(ValueError, match="do not contain a 'log_prior' or 'prior'"):
            validate_posterior_prior_consistency(
                group,
                posterior_samples,
                filename=str(path),
                label="C00:Mixed",
            )


def test_validate_posterior_prior_consistency_rejects_redshift_mismatch(tmp_path):
    path = tmp_path / "bad_redshift.hdf5"
    _write_test_group(path, redshift_shift=0.02)

    with h5py.File(path, "r") as f:
        group = f["C00:Mixed"]
        posterior_samples = group["posterior_samples"][()]
        with pytest.raises(ValueError, match="sample redshifts are inconsistent"):
            validate_posterior_prior_consistency(
                group,
                posterior_samples,
                filename=str(path),
                label="C00:Mixed",
            )
