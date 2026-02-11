#!/usr/bin/env python3
"""
Script to run gravitational wave population analysis with TGR parameters.
Converted from analysis_notebook.ipynb
"""

import sys
import os
import argparse
from glob import glob

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
repo_dir = os.path.dirname(script_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"

# Import required libraries
import bilby
from tqdm import tqdm
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import gelman_rubin, effective_sample_size
from astropy import cosmology as cosmo
import astropy.units as u
from numpyro.infer import MCMC, NUTS, init_to_feasible
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from corner import corner
import pandas as pd
from jax.scipy.special import logsumexp

# Import local modules
from utilties.kde_contour import kdeplot

# Configure numpyro
device_count = int(os.environ.get("TGRPOP_DEVICE_COUNT", 1))
numpyro.set_host_device_count(device_count)
numpyro.set_platform("gpu")
numpyro.enable_x64()

print(f"Using {device_count} devices")

align_spin_prior = bilby.gw.prior.AlignedSpin()

# Setting up the redshift interpolant dV_c/dz /(1+z)
Planck15_LAL = cosmo.FlatLambdaCDM(H0=67.90, Om0=0.3065, name="Planck15_LAL")
zmax = 2.5
zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 1024))
dVdzdt_interp = (
    4
    * np.pi
    * Planck15_LAL.differential_comoving_volume(zinterp)
    .to(u.Gpc**3 / u.sr)
    .value
    / (1 + zinterp)
)

def read_injection_file(
    vt_file, ifar_threshold=1000, use_tilts=False, snr_inspiral_cut=0, snr_cut=0
):
    """Read injection file and extract relevant data."""
    injections = {}

    with h5py.File(vt_file, "r") as f:
        events = f["events"]
        fars = [events[key] for key in events.dtype.names if "far" in key]
        min_fars = np.min(fars, axis=0)
        found = min_fars < 1 / ifar_threshold

        snrs = events["estimated_optimal_snr_net"]

        if snr_cut > 0:
            found = found | (snrs > snr_cut)

        if snr_inspiral_cut > 0:
            snrs_inspiral = (
                1.1
                - 0.9
                * (events["mass1_source"] + events["mass2_source"])
                * (1 + events["redshift"])
                / 100
            ) * snrs
            found = found | (snrs_inspiral > snr_inspiral_cut)

        events = events[found]

        injections["mass_1_source"] = events["mass1_source"]
        injections["mass_ratio"] = (
            events["mass2_source"] / injections["mass_1_source"]
        )
        injections["redshift"] = events["redshift"]
        injections["a_1"] = (
            events["spin1x"] ** 2
            + events["spin1y"] ** 2
            + events["spin1z"] ** 2
        ) ** 0.5
        injections["a_2"] = (
            events["spin2x"] ** 2
            + events["spin2y"] ** 2
            + events["spin2z"] ** 2
        ) ** 0.5
        injections["cos_tilt_1"] = events["spin1z"] / injections["a_1"]
        injections["cos_tilt_2"] = events["spin2z"] / injections["a_2"]

        injections["spin1z"] = events["spin1z"]
        injections["spin2z"] = events["spin2z"]

        ln_prior = events[
            "lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
        ]

        if use_tilts:
            prior = np.exp(ln_prior)
        else:
            prior = np.exp(
                ln_prior
                + np.log(align_spin_prior.prob(events["spin1z"]))
                + np.log(align_spin_prior.prob(events["spin2z"]))
            )

        injections["prior"] = prior / events["weights"]

        q = injections["mass_ratio"]
        a1 = injections["a_1"]
        a2 = injections["a_2"]
        c1 = injections["cos_tilt_1"]
        c2 = injections["cos_tilt_2"]
        s1 = np.sin(np.arccos(c1))
        s2 = np.sin(np.arccos(c2))
        injections["chi_eff"] = (a1 * c1 + q * a2 * c2) / (1 + q)
        injections["chi_p"] = np.max(
            [a1 * s1, a2 * s2 * q * (4 * q + 3) / (4 + 3 * q)],
            axis=0,
        )
        injections["prior_effective_spin"] = injections["prior"]

        injections["found"] = found.sum()
        injections["total_generated"] = f.attrs["total_generated"]

        for key in "analysis_time", "total_analysis_time", "analysis_time_s":
            if key in f.attrs:
                injections["analysis_time"] = f.attrs[key]
        if "analysis_time" not in injections:
            print("analysis_time not found")
        else:
            injections["analysis_time"] /= 60 * 60 * 24 * 365.25

    for key in injections:
        injections[key] = np.asarray(injections[key])

    for key in injections:
        injections[key] = np.array(injections[key])

    return injections


def generate_data(
    event_posteriors,
    injection_file,
    parameter_name,
    use_tgr=True,
    use_tilts=False,
    ifar_threshold=1000,
    N_samples=2000,
    snr_cut=0,
    snr_inspiral_cut=0,
    prng=None,
    scale_tgr=False,
):
    """Generate data arrays for analysis."""
    Nobs = len(event_posteriors)

    print(f"Using {Nobs} events!")

    # Construct the event posterior arrays
    m1s = []
    qs = []
    cost1s = []
    cost2s = []
    a1s = []
    a2s = []
    zs = []
    log_pdraw = []
    dphis = []
    kde_weights = []

    BW_matrices = []
    BW_matrices_sel = []

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

    if scale_tgr:
        # find std(phi) pooled over all events
        pooled_phi = np.concatenate([
            np.asarray(e[parameter_name]).ravel() for e in event_posteriors
        ])
        dphi_scale = np.std(pooled_phi)
    else:
        dphi_scale = 1

    for event_posterior in tqdm(event_posteriors):
        # instead of picking the first N_samples, pick N_samples randomly
        # use this already to apply the weights (should be more efficient
        # than applying the weights after the fact, after trimming the
        # samples)
        if "weights" in event_posterior.dtype.names:
            w = event_posterior["weights"]
        else:
            w = np.ones(len(event_posterior))
        idxs = prng.choice(len(event_posterior), size=N_samples,
                           replace=True, p=w/w.sum())

        m1s.append(event_posterior["mass_1_source"][idxs])
        qs.append(event_posterior["mass_ratio"][idxs])

        a1s.append(event_posterior["a_1"][idxs])
        a2s.append(event_posterior["a_2"][idxs])
        dphis.append(event_posterior[parameter_name][idxs] / dphi_scale)

        cost1s.append(event_posterior["cos_tilt_1"][idxs])
        cost2s.append(event_posterior["cos_tilt_2"][idxs])
        zs.append(event_posterior["redshift"][idxs])
        log_pdraw.append(event_posterior["log_prior"][idxs])

        if use_tgr:
            d = 3
            if use_tilts:
                data_array = np.array(
                    [
                        a1s[-1],
                        a2s[-1],
                        dphis[-1],
                        m1s[-1],
                        qs[-1],
                        zs[-1],
                        cost1s[-1],
                        cost2s[-1],
                    ]
                )
            else:
                data_array = np.array(
                    [a1s[-1], a2s[-1], dphis[-1], m1s[-1], qs[-1], zs[-1]]
                )
        else:
            d = 2
            if use_tilts:
                data_array = np.array(
                    [
                        a1s[-1],
                        a2s[-1],
                        m1s[-1],
                        qs[-1],
                        zs[-1],
                        cost1s[-1],
                        cost2s[-1],
                    ]
                )
            else:
                data_array = np.array(
                    [a1s[-1], a2s[-1], m1s[-1], qs[-1], zs[-1]]
                )

        # could have applied the weights here instead
        weights_i = np.ones(N_samples)

        kde_weights.append(weights_i)

        N_eff = np.sum(weights_i) ** 2 / np.sum(weights_i**2)

        full_cov_i = np.cov(data_array, aweights=weights_i)
        prec_i = np.linalg.inv(full_cov_i)[:d, :d]
        cov_i = np.linalg.inv(prec_i)

        BW_matrices.append(cov_i * N_eff ** (-2.0 / (4 + d)))
        BW_matrices_sel.append(cov_i[:2, :2] * N_eff ** (-2.0 / (6)))

    BW_matrices = np.array(BW_matrices)
    BW_matrices_sel = np.array(BW_matrices_sel)

    event_data_array = np.array(
        [m1s, qs, cost1s, cost2s, a1s, a2s, dphis, zs, log_pdraw, kde_weights]
    )

    injection_data = read_injection_file(
        injection_file,
        ifar_threshold=ifar_threshold,
        snr_cut=snr_cut,
        snr_inspiral_cut=snr_inspiral_cut,
        use_tilts=use_tilts,
    )
    Ndraw = int(injection_data["total_generated"])

    # Construct the injection arrays
    if use_tilts:
        injection_data_array = np.array(
            [
                injection_data["mass_1_source"],
                injection_data["mass_ratio"],
                injection_data["cos_tilt_1"],
                injection_data["cos_tilt_2"],
                injection_data["a_1"],
                injection_data["a_2"],
                injection_data["redshift"],
                injection_data["prior"],
            ]
        )
    else:
        injection_data_array = np.array(
            [
                injection_data["mass_1_source"],
                injection_data["mass_ratio"],
                injection_data["cos_tilt_1"],
                injection_data["cos_tilt_2"],
                injection_data["spin1z"],
                injection_data["spin2z"],
                injection_data["redshift"],
                injection_data["prior"],
            ]
        )

    return (
        event_data_array,
        injection_data_array,
        BW_matrices,
        BW_matrices_sel,
        Nobs,
        Ndraw,
        dphi_scale,
    )


def generate_tgr_only_data(event_posteriors, parameter_name,
                           N_samples=2000, prng=None, scale_tgr=False):
    """Generate TGR-only data arrays."""
    Nobs = len(event_posteriors)

    print(f"Using {Nobs} events!")

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

    if scale_tgr:
        # find std(phi) pooled over all events
        pooled_phi = np.concatenate([
            np.asarray(e[parameter_name]).ravel() for e in event_posteriors
        ])
        dphi_scale = np.std(pooled_phi)
    else:
        dphi_scale = 1

    # Construct the event posterior arrays
    dphis = []
    bws_tgr = []
    for event_posterior in event_posteriors:
        idxs = prng.choice(len(event_posterior), size=N_samples,
                           replace=True)
        dphis.append(event_posterior[parameter_name][idxs] / dphi_scale)

        bws_tgr.append(
            np.std(event_posterior[parameter_name][idxs])
            * N_samples ** (-1.0 / 5)
        )

    bws_tgr = np.array(bws_tgr)
    dphis = np.array(dphis)

    return dphis, bws_tgr, Nobs, dphi_scale


def make_tgr_only_model(dphis, bws_tgr, Nobs, mu_tgr_scale=None,
                        sigma_tgr_scale=None):
    """Define TGR-only model."""
    dphi_scale = jnp.maximum(1e-6, jnp.max(jnp.abs(dphis)))
    if mu_tgr_scale is None:
        mu_tgr_scale = 1
    mu_tgr_scale *= dphi_scale
    if sigma_tgr_scale is None:
        sigma_tgr_scale = 1
    sigma_tgr_scale *= dphi_scale

    mu_tgr = numpyro.sample("mu_tgr", dist.Uniform(-mu_tgr_scale, mu_tgr_scale))
    sigma_tgr = numpyro.sample("sigma_tgr", dist.Uniform(0, sigma_tgr_scale))

    sigma_tgr_i = jnp.sqrt(jnp.square(sigma_tgr) + bws_tgr**2)
    log_wts = dist.Normal(mu_tgr, sigma_tgr_i).log_prob(dphis.T).T

    log_like = logsumexp(log_wts, axis=1)
    log_like = jnp.nan_to_num(log_like, neginf=-1e20, posinf=1e20)
    numpyro.factor("log_likelihood", jnp.sum(log_like))


def make_joint_model(
    event_data_array,
    injection_data_array,
    BW_matrices,
    BW_matrices_sel,
    Nobs,
    Ndraw,
    use_tilts,
    use_tgr,
    mu_tgr_scale=None,
    sigma_tgr_scale=None,
):
    """Define joint model for population analysis."""
    m1s = event_data_array[0]
    qs = event_data_array[1]
    cost1s = event_data_array[2]
    cost2s = event_data_array[3]
    a1_a2s = event_data_array[4:6]
    a1_a2_dphis = event_data_array[4:7]
    zs = event_data_array[7]
    pdraw = event_data_array[8]

    m1s_sel = injection_data_array[0]
    qs_sel = injection_data_array[1]
    cost1s_sel = injection_data_array[2]
    cost2s_sel = injection_data_array[3]
    a1_a2_sel = injection_data_array[4:6]
    zs_sel = injection_data_array[6]
    pdraw_sel = injection_data_array[7]

    # Model parameters
    alpha = numpyro.sample("alpha", dist.Uniform(-4, 12))
    beta = numpyro.sample("beta", dist.Uniform(-4, 12))
    mmin = 5

    frac_bump = numpyro.sample("frac_bump", dist.Uniform(0, 1))
    mu_bump = numpyro.sample("mu_bump", dist.Uniform(20, 50))
    sigma_bump = numpyro.sample("sigma_bump", dist.Uniform(1, 20))

    # Spin magnitude distribution parameters
    mu_spin = numpyro.sample("mu_spin", dist.Uniform(0, 0.7))
    sigma_spin = numpyro.sample("sigma_spin", dist.Uniform(0.05, 10))

    # Redshift parameters
    lamb = numpyro.sample("lamb", dist.Uniform(-30, 30))

    # Defining the models
    def log_m1_powerlaw_density(primary_masses):
        huge_neg = -1.0e12
        indicator = jnp.where(jnp.greater_equal(primary_masses, mmin), 0.0, huge_neg)
        log_powerlaw_comp = -alpha * jnp.log(primary_masses) + indicator
        log_norm = jax.lax.select(
            jnp.isclose(alpha, 1),
            jnp.log(1 / jnp.log(100 / mmin)),
            jnp.log((1 - alpha) / (100 ** (1 - alpha) - mmin ** (1 - alpha))),
        )

        normal_log_prob = dist.Normal(mu_bump, sigma_bump).log_prob(
            primary_masses.T
        ).T

        comp_a = jnp.log1p(-frac_bump) + log_powerlaw_comp + log_norm
        comp_b = jnp.log(frac_bump) + normal_log_prob
        return logsumexp(jnp.stack([comp_a, comp_b], axis=0), axis=0)

    def log_q_powerlaw_density(mass_ratios, primary_masses):
        low = mmin / primary_masses

        log_norm = jax.lax.select(
            jnp.isclose(beta, -1),
            jnp.log(-1 / jnp.log(low)),
            jnp.log((1 + beta) / (1 - low ** (1 + beta))),
        )

        log_norm = jax.lax.select(
            jnp.isnan(log_norm), -jnp.inf * jnp.ones(jnp.shape(log_norm)), log_norm
        )

        indicator = jnp.where(jnp.greater_equal(mass_ratios, low), 0.0, -1.0e12)
        return beta * jnp.log(mass_ratios) + log_norm + indicator

    def log_redshift_powerlaw(redshifts):
        interp_val = jnp.interp(redshifts, zinterp, dVdzdt_interp)
        interp_val = jnp.clip(interp_val, a_min=1e-300)
        return lamb * jnp.log1p(redshifts) + jnp.log(interp_val)

    # Evaluate the per event probabilities
    def safe_log(x):
        return jnp.log(jnp.clip(x, a_min=1e-300))

    log_wts = (
        log_m1_powerlaw_density(m1s)
        + log_q_powerlaw_density(qs, m1s)
        + log_redshift_powerlaw(zs)
        - safe_log(pdraw)
    )

    # Evaluate the selection term
    log_sel_wts = (
        log_m1_powerlaw_density(m1s_sel)
        + log_q_powerlaw_density(qs_sel, m1s_sel)
        + log_redshift_powerlaw(zs_sel)
        - safe_log(pdraw_sel)
    )

    # Adding the tilt
    if use_tilts:
        f_iso = numpyro.sample("f_iso", dist.Uniform(0, 1))
        sigma_tilt = numpyro.sample("sigma_tilt", dist.Uniform(0.05, 10))

        def log_tilt_density(cost1, cost2):
            quad = ((cost1 - 1) ** 2 + (cost2 - 1) ** 2)
            log_gauss = -quad / (2 * jnp.square(sigma_tilt)) - jnp.log(
                2 * jnp.pi * jnp.square(sigma_tilt)
            )
            term_a = jnp.log(f_iso) - jnp.log(4.0)
            term_b = jnp.log1p(-f_iso) + log_gauss
            return logsumexp(jnp.stack([term_a, term_b], axis=0), axis=0)

        log_wts += log_tilt_density(cost1s, cost2s)
        log_sel_wts += log_tilt_density(cost1s_sel, cost2s_sel)

    # Handling the KDE with and without the TGR parameters
    if use_tgr:
        dphi_scale = jnp.maximum(1e-6, jnp.max(jnp.abs(a1_a2_dphis[2])))
        if mu_tgr_scale is None:
            mu_tgr_scale = 1
        mu_tgr_scale *= dphi_scale
        if sigma_tgr_scale is None:
            sigma_tgr_scale = 1
        sigma_tgr_scale *= dphi_scale

        mu_tgr = numpyro.sample("mu_tgr", dist.Uniform(-mu_tgr_scale, mu_tgr_scale))
        sigma_tgr = numpyro.sample("sigma_tgr", dist.Uniform(0, sigma_tgr_scale))

        sigma_evts = BW_matrices + jnp.diag(
            jnp.array(
                [
                    jnp.square(sigma_spin),
                    jnp.square(sigma_spin),
                    jnp.square(sigma_tgr),
                ]
            )
        )
        # Add tiny jitter for numerical stability
        sigma_evts = sigma_evts + jnp.eye(3) * 1e-15
        sigma_sel = jnp.diag(
            jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])
        )
        sigma_sel = sigma_sel + jnp.eye(2) * 1e-15
        mu_evts = jnp.array([mu_spin, mu_spin, mu_tgr])

        logp_normal_sel = dist.MultivariateNormal(
            jnp.array([mu_spin, mu_spin]), sigma_sel
        ).log_prob(a1_a2_sel.T)
        log_sel_wts += logp_normal_sel

        logp_normal = (
            dist.MultivariateNormal(mu_evts, sigma_evts)
            .log_prob(
                jnp.array(
                    [
                        a1_a2_dphis.T,
                    ]
                )
            )
            .T[:, :, 0]
        )
        log_wts += logp_normal
    else:
        sigma_evts = BW_matrices + jnp.diag(
            jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])
        )
        sigma_evts = sigma_evts + jnp.eye(2) * 1e-15
        sigma_sel = jnp.diag(
            jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])
        )
        sigma_sel = sigma_sel + jnp.eye(2) * 1e-15
        mu_evts = jnp.array([mu_spin, mu_spin])

        logp_normal_sel = dist.MultivariateNormal(
            jnp.array([mu_spin, mu_spin]), sigma_sel
        ).log_prob(a1_a2_sel.T)
        log_sel_wts += logp_normal_sel

        logp_normal = (
            dist.MultivariateNormal(mu_evts, sigma_evts)
            .log_prob(
                jnp.array(
                    [
                        a1_a2s.T,
                    ]
                )
            )
            .T[:, :, 0]
        )
        log_wts += logp_normal

    # Adding the per event likelihood term (stable log-sum-exp)
    log_like = logsumexp(log_wts, axis=1)
    log_like = jnp.nan_to_num(log_like, neginf=-1e20, posinf=1e20)
    numpyro.factor("log_likelihood", jnp.sum(log_like))

    # Selection effect term
    log_sel = logsumexp(log_sel_wts) - jnp.log(Ndraw)
    log_sel = jnp.nan_to_num(log_sel, neginf=-1e20, posinf=1e20)
    numpyro.factor("selection", -Nobs * log_sel)

    # N eff cuts
    def log_smooth_neff_boundary(values, criteria):
        scaled_x = (values - criteria) / (0.05 * criteria)
        return jax.lax.select(
            jnp.greater_equal(scaled_x, 0.0), 0.0, -jnp.power(scaled_x, 10)
        )

    neff = jnp.exp(2 * logsumexp(log_wts, axis=1) - logsumexp(2 * log_wts, axis=1))
    min_neff = jnp.min(neff)
    numpyro.deterministic("neff", neff)
    numpyro.factor(
        "neff_criteria",
        jnp.nan_to_num(log_smooth_neff_boundary(min_neff, Nobs), neginf=-1e20, posinf=1e20),
    )

    log_mu2 = logsumexp(2 * log_sel_wts) - 2 * jnp.log(Ndraw)
    ratio = jnp.exp(2 * log_sel - jnp.log(Ndraw) - log_mu2)
    ratio = jnp.clip(ratio, a_min=0.0, a_max=1.0 - 1e-12)
    log_s2 = log_mu2 + jnp.log1p(-ratio)
    neff_sel = jnp.exp(2 * log_sel - log_s2)
    numpyro.deterministic("neff_sel", neff_sel)
    numpyro.factor(
        "neff_sel_criteria",
        jnp.nan_to_num(
            log_smooth_neff_boundary(neff_sel, 4 * Nobs), neginf=-1e20, posinf=1e20
        ),
    )


def get_samples_df(fit):
    """Convert fit results to DataFrame."""
    stacked_samples = fit.posterior.stack(sample=("chain", "draw"))
    samples_df = pd.DataFrame()
    for k, v in stacked_samples.data_vars.items():
        if "neff" not in k:
            samples_df[k] = v.values
    return samples_df


def create_plots(fit_joint, fit_tgr, parameter, outdir):
    """Create and save plots comparing joint and TGR-only models.

    If either fit is None, plotting will proceed using only the available fit(s).
    """
    # Collect available fits
    fits = [("joint", fit_joint), ("tgr", fit_tgr)]
    fits = [(name, fit) for name, fit in fits if fit is not None]
    if len(fits) == 0:
        print("No fits provided; skipping plots.")
        return

    # Get sample dataframes and save sample data
    df_dict = {}
    for key, fit in fits:
        df = get_samples_df(fit)
        df["draw_tgr"] = np.random.normal(df["mu_tgr"], df["sigma_tgr"])
        df_dict[key] = df
        # Save sample data per available fit
        df.to_csv(
            f"{outdir}/fit_{key}_samples.dat", index=False, sep=" "
        )

    # Create population distribution plot
    plt.figure(figsize=(10, 6))
    x = {k: df["draw_tgr"] for k, df in df_dict.items()}
    sns.kdeplot(x, common_norm=False)
    plt.xlabel(parameter)
    plt.title(f"Population Distribution for {parameter}")
    plt.savefig(
        f"{outdir}/population_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create comparison plots
    dfs = []
    labels = []
    for key, df in df_dict.items():
        df["run"] = key
        dfs.append(df)
        labels.append(key)
    df = pd.concat(dfs, ignore_index=True)

    g = sns.PairGrid(
        df,
        x_vars=["mu_tgr", "sigma_tgr"],
        y_vars=["mu_tgr", "sigma_tgr"],
        hue="run",
        diag_sharey=False,
        corner=True,
    )

    g.map_diag(kdeplot, auto_bound=True)
    g.map_offdiag(kdeplot, y_min=0)

    g.axes[1, 0].set_ylim(0)
    g.axes[1, 1].set_xlim(0)

    for i, label in enumerate(labels):
        g.axes[1, 1].plot([], [], color=f"C{i}", label=label)
    g.axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4))

    plt.savefig(f"{outdir}/hyperparameters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # TGR parameters corner plot (overlay if both fits; otherwise single)
    fig = None
    for i, (key, fit) in enumerate(fits):
        if fit is not None:
            fig = corner(
                fit,
                var_names=["mu_tgr", "sigma_tgr"],
                figsize=(12, 12),
                plot_density=False,
                plot_contours=True,
                color=f"C{i}",
                truths=[0, 0],
                truth_color="k",
                fig=fig
            )
    if fig is not None:
        plt.savefig(
            f"{outdir}/tgr_comparison_corner.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Full joint model corner plot (only if joint available)
    if fit_joint is not None:
        corner(
            fit_joint,
            var_names=[
                "alpha",
                "beta",
                "mu_bump",
                "sigma_bump",
                "frac_bump",
                "mu_spin",
                "sigma_spin",
                "lamb",
                "mu_tgr",
                "sigma_tgr",
            ],
            figsize=(12, 12),
            color="C0",
        )
        plt.savefig(
            f"{outdir}/joint_model_corner.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Plots saved to {outdir}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Run gravitational wave population analysis with TGR parameters"
    )
    parser.add_argument(
        "parameter", type=str, help="Parameter name to analyze (e.g., dchi_2)"
    )
    parser.add_argument(
        "data_paths",
        type=str,
        nargs="+",
        help=(
            "Path template(s) to posterior files containing, e.g., "
            "'/abs/path/to/posteriors_*.h5'. Can provide multiple paths."
        ),
    )
    parser.add_argument(
        "--param-key",
        type=str,
        help=(
            "Parameter key for file paths "
            "(default: guessed by searching for 'posterior_samples' in HDF5)"
        ),
    )
    parser.add_argument(
        "--n-warmup", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--n-sample", type=int, default=1000, help="Number of samples"
    )
    parser.add_argument(
        "--n-chains", type=int, default=4, help="Number of chains"
    )
    parser.add_argument(
        "--n-samples-per-event",
        type=int,
        default=2000,
        help="Number of samples per event",
    )
    parser.add_argument(
        "--use-tilts", action="store_true", help="Use tilt angles"
    )
    parser.add_argument(
        "--ifar-threshold", type=float, default=1000, help="IFAR threshold"
    )
    parser.add_argument("--snr-cut", type=float, default=0, help="SNR cut")
    parser.add_argument(
        "--snr-inspiral-cut", type=float, default=6, help="Inspiral SNR cut"
    )
    parser.add_argument(
        '--injection-runs',
        default='o4a',
        choices=['o4a', 'o3+o4a'],
        help="Runs to use for injection selection ('o4a', 'o3+o4a')"
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        help="Path to selection-injection file (supersedes --injection-runs)",
    )
    parser.add_argument(
        "--model",
        choices=["joint", "tgr", "both"],
        default="both",
        help="Which model(s) to run: 'joint' (population+TGR), 'tgr' (TGR-only), or 'both'",
    )
    parser.add_argument(
        "-o", "--outdir", type=str,
        help="Output directory (default: results_{parameter})"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip creating plots"
    )
    parser.add_argument(
        "--seed", type=int, default=150914, help="Random seed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if output files exist",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="List of events to exclude from analysis (e.g., GW15 GW17)",
    )
    parser.add_argument(
        "--mu-tgr-scale",
        type=float,
        default=None,
        help="Scale for mu_tgr parameter (default: auto-calculated from data)",
    )
    parser.add_argument(
        "--sigma-tgr-scale",
        type=float,
        default=None,
        help="Scale for sigma_tgr parameter (default: auto-calculated from data)",
    )
    parser.add_argument(
        "--scale-tgr",
        action="store_true",
        help="Scale TGR parameters by the standard deviation of the data",
    )
    args = parser.parse_args()

    if args.seed == 0:
        seed = np.random.randint(1 << 32)
        print(f"No PRNG key provided, using random seed! {seed}")
    else:
        seed = args.seed
    prng = jax.random.PRNGKey(seed)

    if args.injection_file is None:
        if args.injection_runs == 'o4a':
            injection_file = os.path.join(repo_dir,
                                      "data/selection/mixture-real_o4a-cartesian_spins_20250503134659UTC.hdf")
        elif args.injection_runs == 'o3+o4a':
            injection_file = os.path.join(repo_dir,
                                      "data/selection/mixture-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf")
        else:
            raise ValueError(f"Unrecognizedinjection runs: {args.injection_runs}")
    else:
        injection_file = args.injection_file
    print(f"Using injection file: {injection_file}")

    # Generate the output directory
    outdir = args.outdir or f'results_{args.parameter}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def check_output_files_exist():
        """Check if all expected output files exist."""
        required_files = []

        if args.model in ("joint", "both"):
            required_files.extend(
                [
                    os.path.join(outdir, "result_joint.nc"),
                    os.path.join(outdir, "fit_joint_samples.dat"),
                ]
            )

        if args.model in ("tgr", "both"):
            required_files.extend(
                [
                    os.path.join(outdir, "result_tgr.nc"),
                    os.path.join(outdir, "fit_tgr_samples.dat"),
                ]
            )

        # Add plot files if plots are enabled
        if not args.no_plots:
            plot_files = [
                os.path.join(outdir, "population_distribution.png"),
                os.path.join(outdir, "hyperparameters.png"),
                os.path.join(outdir, "tgr_comparison_corner.png"),
            ]
            if args.model in ("joint", "both"):
                plot_files.append(os.path.join(outdir, "joint_model_corner.png"))
            required_files.extend(plot_files)

        existing_files = [f for f in required_files if os.path.exists(f)]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        return (
            len(existing_files) == len(required_files),
            existing_files,
            missing_files,
        )

    # Check if output files already exist
    all_exist, existing_files, missing_files = check_output_files_exist()

    if all_exist and not args.force:
        print(f"All output files already exist in {outdir}")
        print("Existing files:")
        for f in existing_files:
            print(f"  - {os.path.basename(f)}")
        print("Use --force to re-run the analysis")
        sys.exit(0)
    elif all_exist and args.force:
        print(
            f"Output files exist but --force flag provided. Re-running analysis..."
        )
    elif not all_exist:
        print(f"Missing output files: {len(missing_files)}")
        for f in missing_files:
            print(f"  - {os.path.basename(f)}")
        print("Proceeding with analysis...")

    print(f"Running in output directory: {outdir}")

    # Define file paths

    exclude = args.exclude + ['GW15', 'GW17']
    event_files = []
    discarded_files = []
    for data_path in args.data_paths:
        # exclude pre-O3 events
        paths = glob(data_path)
        for path in paths:
            if any(e in path for e in exclude):
                discarded_files.append(path)
            else:
                event_files.append(path)

    print(f"Discarded {len(discarded_files)} files")
    for f in discarded_files:
        print(f"  - {os.path.basename(f)}")

    if not event_files:
        raise FileNotFoundError(
            f"No event files found: {args.data_paths}"
        )

    if not os.path.exists(injection_file):
        raise FileNotFoundError(
            f"Injection file not found: {injection_file}"
        )

    print(f"Found {len(event_files)} event files")

    # Save injection file path to output directory
    injection_file_path = os.path.join(outdir, "injection_file.txt")
    with open(injection_file_path, "w") as f:
        f.write(f"{injection_file}\n")
    print(f"Saved injection file path to: {injection_file_path}")

    # Save list of event files to output directory
    event_files_list_path = os.path.join(outdir, "event_files.txt")
    with open(event_files_list_path, "w") as f:
        for event_file in event_files:
            f.write(f"{event_file}\n")
    print(f"Saved event files list to: {event_files_list_path}")

    # Save exact command line to output directory
    command_file_path = os.path.join(outdir, "command.txt")
    with open(command_file_path, "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    print(f"Saved command to: {command_file_path}")

    # Loading in the event posteriors
    event_posteriors = []
    for filename in event_files:
        with h5py.File(filename, "r") as f:
            if args.param_key:
                # Search for keys containing param_key
                keys = [k for k in f.keys() if args.param_key in k]
            else:
                # Search for keys containing "posterior_samples"
                keys = [k for k in f.keys() if "posterior_samples" in f[k]]
            if len(keys) != 1:
                raise KeyError(
                    f"Expected 1 key with 'posterior_samples' in {filename}, "
                    f"found {len(keys)}: {keys}"
                )
            posterior = f[keys[0]]["posterior_samples"][()]
        event_posteriors.append(posterior)

    # Generate data arrays
    (
        event_data_array,
        injection_data_array,
        BW_matrices,
        BW_matrices_sel,
        Nobs,
        Ndraw,
        dphi_scale,
    ) = generate_data(
        event_posteriors,
        injection_file,
        args.parameter,
        ifar_threshold=args.ifar_threshold,
        use_tgr=True,
        snr_inspiral_cut=args.snr_inspiral_cut,
        N_samples=args.n_samples_per_event,
        snr_cut=args.snr_cut,
        use_tilts=args.use_tilts,
        prng=seed,
        scale_tgr=args.scale_tgr,
    )

    # store dphi_scale in output directory
    path = os.path.join(outdir, "dphi_scale_joint.txt")
    with open(path, "w") as f:
        f.write(f"{dphi_scale}\n")
    print(f"Saved dphi_scale to: {path}")

    if args.model in ("tgr", "both"):
        event_data_tgr, bws_tgr, _, dphi_scale = generate_tgr_only_data(
            event_posteriors, args.parameter,
            N_samples=args.n_samples_per_event, prng=seed,
            scale_tgr=args.scale_tgr
        )
    else:
        event_data_tgr, bws_tgr, _, dphi_scale = None, None, None, 1

    # store dphi_scale in output directory
    path = os.path.join(outdir, "dphi_scale_tgr.txt")
    with open(path, "w") as f:
        f.write(f"{dphi_scale}\n")
    print(f"Saved dphi_scale to: {path}")

    # Run joint model
    prng0, prng1 = jax.random.split(prng, 2)

    fit_joint = None
    if args.model in ("joint", "both"):
        print("Running joint model...")
        kernel = NUTS(make_joint_model, init_strategy=init_to_feasible())
        mcmc = MCMC(
            kernel,
            num_warmup=args.n_warmup,
            num_samples=args.n_sample,
            num_chains=args.n_chains,
        )
        mcmc.run(
            prng0,
            event_data_array,
            injection_data_array,
            BW_matrices,
            BW_matrices_sel,
            Nobs,
            Ndraw,
            args.use_tilts,
            True,
            args.mu_tgr_scale,
            args.sigma_tgr_scale,
        )

        fit_joint = az.from_numpyro(mcmc)

        fname = f"{outdir}/result_joint.nc"
        fit_joint.to_netcdf(fname)
        print(f"Saved joint results: {fname}")

    # Run TGR-only model
    fit_tgr = None
    if args.model in ("tgr", "both"):
        print("Running TGR-only model...")
        kernel = NUTS(make_tgr_only_model, init_strategy=init_to_feasible())
        mcmc = MCMC(
            kernel,
            num_warmup=args.n_warmup,
            num_samples=args.n_sample,
            num_chains=args.n_chains,
        )
        mcmc.run(
            prng1,
            event_data_tgr,
            bws_tgr,
            Nobs,
            args.mu_tgr_scale,
            args.sigma_tgr_scale,
        )

        fit_tgr = az.from_numpyro(mcmc)

        fname = f"{outdir}/result_tgr.nc"
        fit_tgr.to_netcdf(fname)
        print(f"Saved TGR-only results: {fname}")

    # Create plots and save sample data
    if not args.no_plots:
        print("Creating plots...")
        create_plots(fit_joint, fit_tgr, args.parameter, outdir)
    else:
        # Save sample data for the models that were run
        if fit_joint is not None:
            get_samples_df(fit_joint).to_csv(
                f"{outdir}/fit_joint_samples.dat", index=False, sep=" "
            )
        if fit_tgr is not None:
            get_samples_df(fit_tgr).to_csv(
                f"{outdir}/fit_tgr_samples.dat", index=False, sep=" "
            )

    # Print summary statistics
    if fit_joint is not None:
        print("\nJoint model results:")
        num_chains_joint = fit_joint.posterior.sizes.get("chain", 1)
        for var in fit_joint.posterior:
            if "neff" not in var:
                mean_val = fit_joint.posterior[var].mean().values
                std_val = fit_joint.posterior[var].std().values
                print(f"{var}: {mean_val:.3f} +/- {std_val:.3f}")
                if num_chains_joint >= 2:
                    print(f"Rhat: {gelman_rubin(fit_joint.posterior[var].values):.3f}")
                    print(
                        f"Effective sample size: {effective_sample_size(fit_joint.posterior[var].values):.1f}"
                    )
                print()

    if fit_tgr is not None:
        print("\nTGR-only model results:")
        num_chains_tgr = fit_tgr.posterior.sizes.get("chain", 1)
        for var in fit_tgr.posterior:
            mean_val = fit_tgr.posterior[var].mean().values
            std_val = fit_tgr.posterior[var].std().values
            print(f"{var}: {mean_val:.3f} +/- {std_val:.3f}")
            if num_chains_tgr >= 2:
                print(f"Rhat: {gelman_rubin(fit_tgr.posterior[var].values):.3f}")
                print(
                    f"Effective sample size: {effective_sample_size(fit_tgr.posterior[var].values):.1f}"
                )
            print()

    print(f"Analysis complete! Results saved to {outdir}")


if __name__ == "__main__":
    main()
