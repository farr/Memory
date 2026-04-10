"""Population Predictive Distribution (PPD) computation and plotting.

Provides :func:`generate_ppd` for programmatic use from
``run_hierarchical_analysis.py`` and the command-line ``plot_ppd.py`` wrapper.

For each input NetCDF file, draws the PPD (median + 90% CI shaded band) for:
  - Primary mass  m1          on semi-log-y axes
  - Mass ratio    q = m2/m1   marginalised over m1
  - Spin magnitude a           (Gaussian model, truncated to [0, 1])

Multiple NetCDF files can be overlaid on the same axes for comparison.

Rate mode (injection_file)
---------------------------
Without an injection file the m1 panel shows the normalised population PDF
p(m1).  Supplying an injection file switches the m1 panel to the differential
merger rate dR/dm1 [Gpc^-3 yr^-1 M_sun^-1]:

    dR/dm1(m1 | Lambda) = R(Lambda) * p(m1 | Lambda)

The total volumetric merger rate is drawn from the conditional posterior
given each hyperposterior sample:

    p(R | Lambda, data) = Gamma(k=N_obs, rate=T_obs * beta(Lambda))     (1)

where:
  - N_obs   is the number of observed events used in the analysis
  - T_obs   is the live observing time in years (from injection file metadata)
  - beta(Lambda) is the effective surveyed comoving volume [Gpc^3]:

        beta(Lambda) = (1/N_draw) * sum_{found} p_pop(theta | Lambda) / p_draw(theta)  (2)
"""

import multiprocessing as mp
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm as scipy_norm
import arviz as az

from memory.hierarchical.data import (
    _get_injection_log_draw_prior,
    _get_injection_redshift,
    _get_injection_spin_data,
)
from memory.hierarchical.models import MMIN, MMAX, zinterp, dVdzdt_interp, JOINT_SPIN_PHI_LOGPDF


LOG_2PI = np.log(2.0 * np.pi)
RATE_PARAM_KEYS = (
    "alpha_1",
    "alpha_2",
    "b",
    "frac_bpl",
    "frac_peak_1",
    "frac_peak_2",
    "mu_peak_1",
    "sigma_peak_1",
    "mu_peak_2",
    "sigma_peak_2",
    "beta",
    "lamb",
    "mu_spin",
    "sigma_spin",
    "f_iso",
    "mu_tilt",
    "sigma_tilt",
)
_RATE_WORKER_INJ_DATA = None


def _default_rate_workers():
    """Choose a conservative default number of CPU workers."""
    cpu_count = mp.cpu_count() or 1
    return min(4, cpu_count)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_params(nc_file, n_ppd, seed):
    """Return a dict of 1-D posterior sample arrays from an ArviZ NetCDF file."""
    fit = az.from_netcdf(nc_file)
    stacked = fit.posterior.stack(sample=("chain", "draw"))
    params = {
        k: v.values
        for k, v in stacked.data_vars.items()
        if "neff" not in k and v.values.ndim == 1
    }
    N = len(next(iter(params.values())))
    # Backward compatibility: some archived runs fixed mu_tilt = 1 and did not
    # write it to the NetCDF posterior. Reconstruct that constant here so the
    # tilt selection term can still be evaluated during post-processing.
    if "mu_tilt" not in params and ("f_iso" in params or "sigma_tilt" in params):
        params["mu_tilt"] = np.ones(N)
    if n_ppd is not None and n_ppd < N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_ppd, replace=False)
        params = {k: v[idx] for k, v in params.items()}
    return params


def _slice_rate_params(params, start, end):
    """Return the subset of posterior hyperparameters needed for rate mode."""
    return {k: np.asarray(params[k][start:end]) for k in RATE_PARAM_KEYS}


def _load_injection_data(inj_file, ifar_threshold=1000, snr_inspiral_cut=6):
    """Load found injections from an HDF5 sensitivity-estimate file.

    Applies the same IFAR + inspiral-SNR selection as data.py and mirrors the
    same ``log_p_draw`` convention used by the inference model for both the new
    polar-spin releases and the legacy Cartesian mixture files. This keeps the
    post-processed beta and rate posterior consistent with archived runs.

    Parameters
    ----------
    inj_file : str
        Path to the HDF5 sensitivity-estimate file.
    ifar_threshold : float
        Minimum IFAR [yr] for an injection to be considered "found".
    snr_inspiral_cut : float
        Alternative "found" criterion: inspiral-SNR proxy above this value.

    Returns
    -------
    dict with keys:
        m1, q, z, a1, a2, cos_tilt_1, cos_tilt_2
                           : parameter arrays for found injections
        log_p_draw         : log draw prior in
                             (m1, q, z, a1, a2, cos_tilt_1, cos_tilt_2) space
        Ndraw              : total injections attempted (found + missed)
        T_obs_yr           : live observing time in years
    """
    with h5py.File(inj_file, "r") as f:
        ev = f["events"]
        m1_all = ev["mass1_source"][:]
        m2_all = ev["mass2_source"][:]
        z_all = np.asarray(_get_injection_redshift(ev))
        spin_data = _get_injection_spin_data(ev)
        lnp_all, draw_coordinates = _get_injection_log_draw_prior(ev)
        wgt_all = ev["weights"][:]
        if "estimated_optimal_snr_net" in ev.dtype.names:
            snr_all = ev["estimated_optimal_snr_net"][:]
        else:
            snr_all = ev["snr_net"][:]

        # IFAR cut
        far_keys = [k for k in ev.dtype.names if "far" in k]
        min_far  = np.min([ev[k][:] for k in far_keys], axis=0)
        found    = min_far < 1.0 / ifar_threshold

        # Inspiral-SNR cut (matches data.py)
        if snr_inspiral_cut > 0:
            snr_insp = (
                1.1 - 0.9 * (m1_all + m2_all) * (1.0 + z_all) / 100.0
            ) * snr_all
            found = found | (snr_insp > snr_inspiral_cut)

        Ndraw = int(f.attrs["total_generated"])
        T_obs = float(f.attrs["total_analysis_time"]) / (3600.0 * 24.0 * 365.25)

    a1_all = np.asarray(spin_data["a_1"])
    a2_all = np.asarray(spin_data["a_2"])
    cost1_all = np.asarray(spin_data["cos_tilt_1"])
    cost2_all = np.asarray(spin_data["cos_tilt_2"])

    sin_tilt_1 = np.asarray(spin_data["sin_tilt_1"])
    sin_tilt_2 = np.asarray(spin_data["sin_tilt_2"])

    # Match memory.hierarchical.data.read_injection_file exactly so the
    # post-processing beta uses the same draw-density convention as the
    # inference model for archived runs.
    log_p_draw = (
        lnp_all
        + np.log(np.maximum(m1_all, 1e-300))
        - np.log(np.maximum(wgt_all, 1e-300))
    )
    if draw_coordinates == "cartesian":
        log_p_draw += (
            2.0 * np.log(np.maximum(a1_all, 1e-300))
            + 2.0 * np.log(np.maximum(a2_all, 1e-300))
        )
    else:
        log_p_draw -= (
            np.log(np.maximum(sin_tilt_1, 1e-300))
            + np.log(np.maximum(sin_tilt_2, 1e-300))
        )

    return {
        "m1":       m1_all[found],
        "q":        m2_all[found] / m1_all[found],
        "z":        z_all[found],
        "a1":       a1_all[found],
        "a2":       a2_all[found],
        "cos_tilt_1": cost1_all[found],
        "cos_tilt_2": cost2_all[found],
        "log_p_draw": log_p_draw[found],
        "Ndraw":    Ndraw,
        "T_obs_yr": T_obs,
    }


# ---------------------------------------------------------------------------
# Population model helpers (numpy replication of models.py)
# ---------------------------------------------------------------------------

def _log_norm_pl(m, alpha, lo, hi):
    """Vectorized log-density of a normalized power law on [lo, hi].

    Parameters
    ----------
    m     : array (..., M)
    alpha : array (..., 1)  — power-law index (model convention: rho ~ m^{-alpha})
    lo    : scalar or array (..., 1)
    hi    : scalar or array (..., 1)

    Returns array of the same broadcast shape as the inputs.
    """
    log_unnorm = -alpha * np.log(m)
    not_one = ~np.isclose(alpha, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_norm_not1 = np.log(np.abs(1.0 - alpha)) - np.log(
            np.maximum(1e-300, np.abs(hi ** (1.0 - alpha) - lo ** (1.0 - alpha)))
        )
        log_norm_1 = -np.log(np.log(hi / lo)) * np.ones_like(alpha)
    return log_unnorm + np.where(not_one, log_norm_not1, log_norm_1)


def _log_normal_pdf(x, mu, sigma):
    """Fast vectorized Normal log-PDF."""
    sigma2 = np.square(sigma)
    return -0.5 * (np.square(x - mu) / sigma2 + np.log(sigma2) + LOG_2PI)


def _log_tilt_density(cost1, cost2, f_iso, mu_tilt, sigma_tilt):
    """Replicate the tilt mixture used in ``make_joint_model``."""
    quad = (cost1 - mu_tilt) ** 2 + (cost2 - mu_tilt) ** 2
    log_gauss = -0.5 * quad / np.square(sigma_tilt) - np.log(
        2.0 * np.pi * np.square(sigma_tilt)
    )
    trunc_norm_1d = scipy_norm.cdf((1.0 - mu_tilt) / sigma_tilt) - scipy_norm.cdf(
        (-1.0 - mu_tilt) / sigma_tilt
    )
    log_gauss -= 2.0 * np.log(np.maximum(trunc_norm_1d, 1e-300))
    term_a = np.log(np.maximum(f_iso, 1e-300)) - np.log(4.0)
    term_b = np.log(np.maximum(1.0 - f_iso, 1e-300)) + log_gauss
    return np.logaddexp(term_a, term_b)


def _init_rate_worker(inj_data):
    """Store the shared injection data in a forked worker process."""
    global _RATE_WORKER_INJ_DATA
    _RATE_WORKER_INJ_DATA = inj_data


def _compute_rate_chunk(params_chunk, inj_data):
    """Compute selection-factor samples beta(Lambda) for one posterior chunk."""
    m1_inj = inj_data["m1"][np.newaxis, :]
    q_inj = inj_data["q"][np.newaxis, :]
    z_inj = inj_data["z"][np.newaxis, :]
    a1_inj = inj_data["a1"][np.newaxis, :]
    a2_inj = inj_data["a2"][np.newaxis, :]
    cost1_inj = inj_data["cos_tilt_1"][np.newaxis, :]
    cost2_inj = inj_data["cos_tilt_2"][np.newaxis, :]
    lp_inj = inj_data["log_p_draw"][np.newaxis, :]
    Ndraw = inj_data["Ndraw"]
    dVdzdt_inj = np.interp(z_inj, zinterp, dVdzdt_interp)
    log_q_inj = np.log(np.maximum(q_inj, 1e-300))
    log_m1_over_mmin = np.log(m1_inj / MMIN)

    alpha_1 = params_chunk["alpha_1"][:, np.newaxis]
    alpha_2 = params_chunk["alpha_2"][:, np.newaxis]
    b = params_chunk["b"][:, np.newaxis]
    frac_bpl = params_chunk["frac_bpl"][:, np.newaxis]
    frac_peak_1 = params_chunk["frac_peak_1"][:, np.newaxis]
    frac_peak_2 = params_chunk["frac_peak_2"][:, np.newaxis]
    mu_peak_1 = params_chunk["mu_peak_1"][:, np.newaxis]
    sigma_peak_1 = params_chunk["sigma_peak_1"][:, np.newaxis]
    mu_peak_2 = params_chunk["mu_peak_2"][:, np.newaxis]
    sigma_peak_2 = params_chunk["sigma_peak_2"][:, np.newaxis]
    beta_q = params_chunk["beta"][:, np.newaxis]
    lamb = params_chunk["lamb"][:, np.newaxis]
    mu_spin = params_chunk["mu_spin"][:, np.newaxis]
    sigma_spin = params_chunk["sigma_spin"][:, np.newaxis]
    f_iso = params_chunk["f_iso"][:, np.newaxis]
    mu_tilt = params_chunk["mu_tilt"][:, np.newaxis]
    sigma_tilt = params_chunk["sigma_tilt"][:, np.newaxis]

    m_break = MMIN + b * (MMAX - MMIN)
    log_pl_lo = _log_norm_pl(m1_inj, alpha_1, MMIN, m_break)
    log_pl_hi = _log_norm_pl(m1_inj, alpha_2, m_break, MMAX)
    log_C = (
        _log_norm_pl(m_break, alpha_2, m_break, MMAX)
        - _log_norm_pl(m_break, alpha_1, MMIN, m_break)
    )
    log_bpl = (
        np.where(m1_inj < m_break, log_pl_lo + log_C, log_pl_hi)
        - np.logaddexp(0.0, log_C)
    )
    trunc_g1 = scipy_norm.cdf((MMAX - mu_peak_1) / sigma_peak_1) - scipy_norm.cdf((MMIN - mu_peak_1) / sigma_peak_1)
    log_g1 = _log_normal_pdf(m1_inj, mu_peak_1, sigma_peak_1) - np.log(np.maximum(trunc_g1, 1e-300))
    trunc_g2 = scipy_norm.cdf((MMAX - mu_peak_2) / sigma_peak_2) - scipy_norm.cdf((MMIN - mu_peak_2) / sigma_peak_2)
    log_g2 = _log_normal_pdf(m1_inj, mu_peak_2, sigma_peak_2) - np.log(np.maximum(trunc_g2, 1e-300))
    log_m1d = np.logaddexp(
        np.logaddexp(
            np.log(np.maximum(frac_bpl, 1e-30)) + log_bpl,
            np.log(np.maximum(frac_peak_1, 1e-30)) + log_g1,
        ),
        np.log(np.maximum(frac_peak_2, 1e-30)) + log_g2,
    )
    log_m1d = np.where((m1_inj >= MMIN) & (m1_inj <= MMAX), log_m1d, -1e12)

    low = MMIN / m1_inj
    not_neg1 = ~np.isclose(beta_q, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_n_ne1 = np.log(np.abs(1.0 + beta_q)) - np.log(
            np.maximum(1e-300, np.abs(1.0 - low ** (1.0 + beta_q)))
        )
        log_n_e1 = -np.log(log_m1_over_mmin)
    log_qd = beta_q * log_q_inj + np.where(not_neg1, log_n_ne1, log_n_e1)
    log_qd = np.where(q_inj >= low, log_qd, -1e12)

    log_zd = lamb * np.log1p(z_inj) + np.log(np.maximum(dVdzdt_inj, 1e-300))
    log_sd = _log_normal_pdf(a1_inj, mu_spin, sigma_spin) + _log_normal_pdf(
        a2_inj, mu_spin, sigma_spin
    )
    log_td = _log_tilt_density(cost1_inj, cost2_inj, f_iso, mu_tilt, sigma_tilt)

    log_wts = log_m1d + log_qd + log_zd + log_sd + log_td - lp_inj + JOINT_SPIN_PHI_LOGPDF
    log_sel = logsumexp(log_wts, axis=1) - np.log(Ndraw)
    return np.exp(log_sel)


def _compute_rate_chunk_worker(task):
    """Worker wrapper for multiprocessing."""
    params_chunk = task
    return _compute_rate_chunk(params_chunk, _RATE_WORKER_INJ_DATA)


# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------

def compute_R_samples(params, inj_data, N_obs, chunk=50, workers=1, seed=None):
    """Draw merger-rate samples R for each posterior sample.

    The expected number of detections given population hyperparameters Λ is

        mu(Λ) = R(Λ) * T_obs * beta(Λ)

    With the usual log-uniform prior pi(R) propto 1/R, the conditional
    posterior for the local merger rate is

        R | Λ, data ~ Gamma(shape=N_obs, rate=T_obs * beta(Λ))         (1)

    The effective surveyed volume beta(Λ) [Gpc^3] is estimated by importance
    sampling over the found injections:

        beta(Λ) = (1/N_draw) * sum_{found}  p_pop(theta | Λ) / p_draw(theta)   (2)

    Parameters
    ----------
    params : dict
        Posterior sample arrays (1-D, each of length N).
    inj_data : dict
        Output of _load_injection_data.
    N_obs : int
        Number of observed events used in the analysis.
    chunk : int
        Number of posterior samples processed at once (memory control).
    workers : int
        Number of CPU worker processes for chunked rate evaluation.
    seed : int or None
        Seed for reproducible conditional rate draws.

    Returns
    -------
    R_samples : ndarray, shape (N,)
        Conditional merger-rate draws in Gpc^-3 yr^-1, one per posterior sample.
    """
    N = len(params["alpha_1"])
    workers = max(1, int(workers))
    T_obs = float(inj_data["T_obs_yr"])
    rng = np.random.default_rng(seed)
    tasks = [
        _slice_rate_params(params, i0, min(i0 + chunk, N))
        for i0 in range(0, N, chunk)
    ]

    if workers == 1 or len(tasks) == 1:
        parts = [
            _compute_rate_chunk(params_chunk, inj_data)
            for params_chunk in tasks
        ]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=min(workers, len(tasks)),
            initializer=_init_rate_worker,
            initargs=(inj_data,),
        ) as pool:
            parts = pool.map(_compute_rate_chunk_worker, tasks)

    beta_samples = np.concatenate(parts)
    rate_scale = 1.0 / np.maximum(T_obs * beta_samples, 1e-300)
    return rng.gamma(shape=N_obs, scale=rate_scale)


# ---------------------------------------------------------------------------
# PPD computations
# ---------------------------------------------------------------------------

def compute_ppd_m1(m1_grid, params):
    """Evaluate p(m1 | theta_i) on *m1_grid* for each posterior sample.

    Returns array of shape (N, M).
    """
    m  = m1_grid[np.newaxis, :]                                   # (1, M)
    a1 = params["alpha_1"][:, np.newaxis]                         # (N, 1)
    a2 = params["alpha_2"][:, np.newaxis]
    mb = (MMIN + params["b"] * (MMAX - MMIN))[:, np.newaxis]     # (N, 1)
    fb = params["frac_bpl"][:, np.newaxis]
    f1 = params["frac_peak_1"][:, np.newaxis]
    f2 = params["frac_peak_2"][:, np.newaxis]
    m1 = params["mu_peak_1"][:, np.newaxis]
    s1 = params["sigma_peak_1"][:, np.newaxis]
    m2 = params["mu_peak_2"][:, np.newaxis]
    s2 = params["sigma_peak_2"][:, np.newaxis]

    # Broken power law
    log_pl_lo = _log_norm_pl(m, a1, MMIN, mb)                    # (N, M)
    log_pl_hi = _log_norm_pl(m, a2, mb, MMAX)
    log_C     = (_log_norm_pl(mb, a2, mb, MMAX)                   # (N, 1)
                 - _log_norm_pl(mb, a1, MMIN, mb))

    log_bpl = (
        np.where(m < mb, log_pl_lo + log_C, log_pl_hi)
        - np.logaddexp(0.0, log_C)
    )

    # Gaussian peaks (truncated to [MMIN, MMAX])
    trunc1 = scipy_norm.cdf((MMAX - m1) / s1) - scipy_norm.cdf((MMIN - m1) / s1)
    log_g1 = scipy_norm.logpdf(m, m1, s1) - np.log(np.maximum(trunc1, 1e-300))
    trunc2 = scipy_norm.cdf((MMAX - m2) / s2) - scipy_norm.cdf((MMIN - m2) / s2)
    log_g2 = scipy_norm.logpdf(m, m2, s2) - np.log(np.maximum(trunc2, 1e-300))

    # Three-component mixture
    log_p = np.logaddexp(
        np.logaddexp(
            np.log(np.maximum(fb, 1e-30)) + log_bpl,
            np.log(np.maximum(f1, 1e-30)) + log_g1,
        ),
        np.log(np.maximum(f2, 1e-30)) + log_g2,
    )
    # Apply support indicator [mmin, mmax]
    in_support = (m1_grid >= MMIN) & (m1_grid <= MMAX)
    log_p[:, ~in_support] = -np.inf
    return np.exp(log_p)


def compute_ppd_q(q_grid, m1_grid, p_m1_normed, params, chunk=400):
    """Compute marginal p(q) = ∫ p(q|m1) p(m1) dm1.

    Processes posterior samples in chunks of *chunk* to limit peak memory.
    Returns array of shape (N, Q).
    """
    beta  = params["beta"]
    N     = len(beta)
    Q     = len(q_grid)
    dm1   = np.diff(m1_grid)
    p_q   = np.zeros((N, Q))

    # q and m1 are fixed across all sample chunks
    q_2d  = q_grid[np.newaxis, :, np.newaxis]       # (1, Q, 1)
    m_2d  = m1_grid[np.newaxis, np.newaxis, :]       # (1, 1, M)
    low   = MMIN / m_2d                              # (1, 1, M)  q_min for this m1
    valid = (q_2d >= low) & (q_2d <= 1.0)           # (1, Q, M)

    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        b    = beta[start:end, np.newaxis, np.newaxis]          # (nc, 1, 1)
        pm   = p_m1_normed[start:end, np.newaxis, :]            # (nc, 1, M)

        not_neg1 = ~np.isclose(b, -1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_norm_gen = np.log(np.abs(1.0 + b)) - np.log(
                np.maximum(1e-300, np.abs(1.0 - low ** (1.0 + b)))
            )
            log_norm_m1  = -np.log(np.log(m_2d / MMIN))
        log_norm = np.where(not_neg1, log_norm_gen, log_norm_m1)  # (nc, 1, M)

        log_pq_m1 = b * np.log(np.maximum(q_2d, 1e-300)) + log_norm  # (nc, Q, M)
        log_pq_m1 = np.where(valid, log_pq_m1, -np.inf)

        pqm1 = np.exp(log_pq_m1) * pm                          # (nc, Q, M)
        # Trapezoidal integration over m1
        p_q[start:end] = 0.5 * np.sum(
            (pqm1[:, :, :-1] + pqm1[:, :, 1:]) * dm1[np.newaxis, np.newaxis, :],
            axis=2,
        )

    return p_q


def compute_ppd_spin(a_grid, params):
    """Evaluate the Gaussian spin-magnitude PPD truncated to [0, 1].

    Returns array of shape (N, A).
    """
    a  = a_grid[np.newaxis, :]
    mu = params["mu_spin"][:, np.newaxis]
    s  = params["sigma_spin"][:, np.newaxis]

    p_a = scipy_norm.pdf(a, mu, s)
    p_a[:, (a_grid < 0) | (a_grid > 1)] = 0.0

    # Renormalize over [0, 1]
    da   = np.diff(a_grid)
    norm = 0.5 * np.sum((p_a[:, :-1] + p_a[:, 1:]) * da, axis=1, keepdims=True)
    return p_a / np.maximum(norm, 1e-300)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_band(ax, grid, ppd_arr, color, label, ci=(5, 95)):
    """Shade the CI and draw the median of a (N, G) PPD array."""
    lo, med, hi = np.percentile(ppd_arr, [ci[0], 50, ci[1]], axis=0)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.25)
    ax.plot(grid, med, color=color, label=label, lw=1.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ppd(
    nc_files,
    outdir,
    *,
    labels=None,
    injection_file=None,
    n_obs=None,
    n_ppd=None,
    seed=42,
    rate_chunk=50,
    rate_workers=None,
    n_m1_grid=1200,
    n_q_grid=1200,
    n_m1_q_grid=None,
    n_a_grid=200,
    out_filename="ppd.png",
):
    """Compute and save PPD plots for one or more hierarchical analysis results.

    Parameters
    ----------
    nc_files : list of str
        ArviZ NetCDF result file paths.
    outdir : str
        Directory where ``out_filename`` will be saved.
    labels : list of str, optional
        Legend labels for each file (default: stem of each filename).
    injection_file : str, optional
        HDF5 sensitivity-estimate file.  When provided the m1 panel shows
        dR/dm1 [Gpc⁻³ yr⁻¹ M☉⁻¹] instead of p(m1).
    n_obs : int, optional
        Number of observed events used in the analysis (required when
        ``injection_file`` is given).
    n_ppd : int, optional
        Maximum posterior samples to use (default: all; 500 in rate mode).
    seed : int
        RNG seed.
    rate_chunk : int
        Posterior chunk size for the rate calculation.
    rate_workers : int, optional
        Number of worker processes for the rate calculation (default: min(4, cpu_count)).
    n_m1_grid : int
        Number of primary-mass grid points for the plotted m1 panel.
    n_q_grid : int
        Number of mass-ratio grid points.
    n_m1_q_grid : int, optional
        Number of primary-mass grid points for the internal q marginalisation
        (default: max(n_m1_grid, 4000)).
    n_a_grid : int
        Number of spin-magnitude grid points.
    out_filename : str
        Output filename relative to *outdir* (default: ``ppd.png``).
    """
    if rate_workers is None:
        rate_workers = _default_rate_workers()

    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in nc_files]

    os.makedirs(outdir, exist_ok=True)

    effective_n_ppd = n_ppd
    if effective_n_ppd is None and injection_file is not None:
        effective_n_ppd = 500
        print(
            "Rate mode: defaulting to 500 posterior samples for faster "
            "post-processing. Override with --n-ppd."
        )

    # Load injection data once (shared across all NC files)
    inj_data = None
    if injection_file is not None:
        print(f"Loading injection file: {injection_file} ...")
        inj_data = _load_injection_data(injection_file)
        print(
            f"  {len(inj_data['m1'])} found injections, "
            f"Ndraw={inj_data['Ndraw']}, "
            f"T_obs={inj_data['T_obs_yr']:.4f} yr"
        )

    # Evaluation grids.
    # m1_grid excludes MMIN exactly: at m1=MMIN the q-distribution is a Dirac delta
    # at q=1 and its normalisation diverges, causing float64 overflow in the integral.
    # Dropping that single measure-zero point has no effect on the integral value.
    #
    # Use a denser *internal* m1 grid for the q marginalisation than for the plotted
    # m1 panel. This smooths the visible stair-step artifacts from the hard
    # q >= MMIN / m1 support boundary without forcing an equally dense display grid.
    _n_m1_q_grid = n_m1_q_grid or max(n_m1_grid, 4000)
    m1_grid = np.linspace(MMIN, MMAX, n_m1_grid + 1)[1:]
    m1_q_grid = np.linspace(MMIN, MMAX, _n_m1_q_grid + 1)[1:]
    q_grid = np.linspace(0.01, 1.0, n_q_grid)
    a_grid = np.linspace(0.0, 1.0, n_a_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_m1, ax_q, ax_a = axes

    rate_title_lines = []

    for i, (nc_file, label) in enumerate(zip(nc_files, labels)):
        color = f"C{i}"
        print(f"[{label}] Loading {nc_file} ...")
        params = _load_params(nc_file, effective_n_ppd, seed + i)
        N = len(params["alpha_1"])
        print(f"[{label}] {N} samples — computing PPDs ...")

        # Primary mass
        pm1 = compute_ppd_m1(m1_grid, params)                    # (N, M)
        dm1 = np.diff(m1_grid)
        norm_m1 = 0.5 * np.sum(
            (pm1[:, :-1] + pm1[:, 1:]) * dm1, axis=1, keepdims=True
        )
        pm1_normed = pm1 / np.maximum(norm_m1, 1e-300)

        # Re-evaluate p(m1) on a denser internal grid for the q marginalisation.
        pm1_q = compute_ppd_m1(m1_q_grid, params)
        dm1_q = np.diff(m1_q_grid)
        norm_m1_q = 0.5 * np.sum(
            (pm1_q[:, :-1] + pm1_q[:, 1:]) * dm1_q, axis=1, keepdims=True
        )
        pm1_q_normed = pm1_q / np.maximum(norm_m1_q, 1e-300)

        if inj_data is not None:
            print(f"[{label}] Drawing merger-rate posterior R(Λ) ...")
            R_samples = compute_R_samples(
                params,
                inj_data,
                n_obs,
                chunk=rate_chunk,
                workers=rate_workers,
                seed=seed + i,
            )
            R_med = np.median(R_samples)
            R_lo, R_hi = np.percentile(R_samples, [5, 95])
            print(
                f"[{label}] R(z=0) = {R_med:.2f} [{R_lo:.2f}, {R_hi:.2f}] "
                f"Gpc⁻³ yr⁻¹  (median, 90% CI)"
            )
            if len(nc_files) > 1:
                rate_title_lines.append(
                    f"{label}: R = {R_med:.2f} [{R_lo:.2f}, {R_hi:.2f}]"
                )
            else:
                rate_title_lines.append(
                    f"R = {R_med:.2f} [{R_lo:.2f}, {R_hi:.2f}] Gpc^-3 yr^-1"
                )
            # Evaluate rate at z = 0.2 (matches LVK populations paper convention).
            z_eval = 0.2
            R_at_z = R_samples * (1.0 + z_eval) ** params["lamb"]
            R_z_med = np.median(R_at_z)
            R_z_lo, R_z_hi = np.percentile(R_at_z, [5, 95])
            print(
                f"[{label}] R(z={z_eval}) = {R_z_med:.2f} [{R_z_lo:.2f}, {R_z_hi:.2f}] "
                f"Gpc⁻³ yr⁻¹  (median, 90% CI)"
            )
            # dR/dm1(z=0.2) = R(z=0.2) × p(m1), shape (N, M)
            dRdm1 = R_at_z[:, np.newaxis] * pm1_normed
            _plot_band(ax_m1, m1_grid, dRdm1, color, label)
        else:
            _plot_band(ax_m1, m1_grid, pm1_normed, color, label)

        # Mass ratio (marginalised over m1)
        pq = compute_ppd_q(q_grid, m1_q_grid, pm1_q_normed, params)  # (N, Q)
        _plot_band(ax_q, q_grid, pq, color, label)

        # Spin magnitude
        pa = compute_ppd_spin(a_grid, params)                     # (N, A)
        _plot_band(ax_a, a_grid, pa, color, label)

    # Formatting
    ax_m1.set_yscale("log")
    ax_m1.set_ylim(bottom=1e-3)
    ax_m1.set_xlabel(r"$m_1\ [M_\odot]$")
    if inj_data is not None:
        ax_m1.set_ylabel(
            r"$\mathrm{d}R/\mathrm{d}m_1\ (z{=}0.2)$"
            r"$\ [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,M_\odot^{-1}]$"
        )
        title = "Primary mass merger rate"
        if rate_title_lines:
            title = title + "\n" + "\n".join(rate_title_lines)
        ax_m1.set_title(title)
    else:
        ax_m1.set_ylabel(r"$p(m_1)\ [M_\odot^{-1}]$")
        ax_m1.set_title("Primary mass")
    ax_m1.set_xlim(MMIN, 100)  # display mass panel up to 100 Msun

    ax_q.set_xlabel(r"$q = m_2/m_1$")
    ax_q.set_ylabel(r"$p(q)$")
    ax_q.set_title("Mass ratio")
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(bottom=0)

    ax_a.set_xlabel(r"$a$")
    ax_a.set_ylabel(r"$p(a)$")
    ax_a.set_title("Spin magnitude")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(bottom=0)

    if len(nc_files) > 1:
        for ax in axes:
            ax.legend()

    plt.tight_layout()
    out_path = os.path.join(outdir, out_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
