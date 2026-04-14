"""Population Predictive Distribution (PPD) computation and plotting.

Provides :func:`generate_ppd` for programmatic use from
``run_hierarchical_analysis.py`` and the command-line ``plot_ppd.py`` wrapper.

For each input NetCDF file, draws the PPD (median + 90% CI shaded band) for:
  - Primary mass  m1          on semi-log-y axes
  - Mass ratio    q = m2/m1   marginalised over m1
  - Spin magnitude a           (Gaussian model, truncated to [0, 1])

When the posterior trace contains ``R`` (the local merger-rate density
R(z=0) in Gpc^-3 yr^-1, drawn in the model via the Gamma auxiliary
variable), the m1 panel automatically shows the differential merger rate

    dR/dm1(m1, z=0.2) = R_0 * (1+0.2)^lambda * p(m1 | Lambda)

in units of Gpc^-3 yr^-1 M_sun^-1, matching the LVK populations paper
convention.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
import arviz as az

from memory.hierarchical.models import MMIN, MMAX


LOG_2PI = np.log(2.0 * np.pi)
Z_RATE = 0.2


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
    if "mu_tilt" not in params and ("f_iso" in params or "sigma_tilt" in params):
        params["mu_tilt"] = np.ones(N)
    if n_ppd is not None and n_ppd < N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_ppd, replace=False)
        params = {k: v[idx] for k, v in params.items()}
    return params


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
    """Compute marginal p(q) = integral p(q|m1) p(m1) dm1.

    Processes posterior samples in chunks of *chunk* to limit peak memory.
    Returns array of shape (N, Q).
    """
    beta  = params["beta"]
    N     = len(beta)
    Q     = len(q_grid)
    dm1   = np.diff(m1_grid)
    p_q   = np.zeros((N, Q))

    q_2d  = q_grid[np.newaxis, :, np.newaxis]       # (1, Q, 1)
    m_2d  = m1_grid[np.newaxis, np.newaxis, :]       # (1, 1, M)
    low   = MMIN / m_2d                              # (1, 1, M)
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
    """Evaluate the truncated Gaussian spin-magnitude PPD on [0, 1].

    Uses the analytic normalization constant
        Z = Φ((1 − μ)/σ) − Φ((0 − μ)/σ)
    to match the model exactly.

    Returns array of shape (N, A).
    """
    a  = a_grid[np.newaxis, :]
    mu = params["mu_spin"][:, np.newaxis]
    s  = params["sigma_spin"][:, np.newaxis]

    trunc_norm = (
        scipy_norm.cdf((1.0 - mu) / s) - scipy_norm.cdf((0.0 - mu) / s)
    )
    p_a = scipy_norm.pdf(a, mu, s) / np.maximum(trunc_norm, 1e-300)
    p_a[:, (a_grid < 0) | (a_grid > 1)] = 0.0
    return p_a


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
    n_ppd=None,
    seed=42,
    n_m1_grid=1200,
    n_q_grid=1200,
    n_m1_q_grid=None,
    n_a_grid=200,
    out_filename="ppd.png",
):
    """Compute and save PPD plots for one or more hierarchical analysis results.

    When the posterior trace contains ``R`` (the local merger-rate density
    sampled via the Gamma auxiliary variable in ``make_joint_model``), the
    m1 panel automatically shows

        dR/dm1(m1, z=0.2) = R_0 * (1+0.2)^lambda * p(m1 | Lambda)

    in units of Gpc^-3 yr^-1 M_sun^-1.

    Parameters
    ----------
    nc_files : list of str
        ArviZ NetCDF result file paths.
    outdir : str
        Directory where ``out_filename`` will be saved.
    labels : list of str, optional
        Legend labels for each file (default: stem of each filename).
    n_ppd : int, optional
        Maximum posterior samples to use (default: all).
    seed : int
        RNG seed.
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
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in nc_files]

    os.makedirs(outdir, exist_ok=True)

    _n_m1_q_grid = n_m1_q_grid or max(n_m1_grid, 4000)
    m1_grid = np.linspace(MMIN, MMAX, n_m1_grid + 1)[1:]
    m1_q_grid = np.linspace(MMIN, MMAX, _n_m1_q_grid + 1)[1:]
    q_grid = np.linspace(0.01, 1.0, n_q_grid)
    a_grid = np.linspace(0.0, 1.0, n_a_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_m1, ax_q, ax_a = axes

    rate_title_lines = []
    any_has_rate = False

    for i, (nc_file, label) in enumerate(zip(nc_files, labels)):
        color = f"C{i}"
        print(f"[{label}] Loading {nc_file} ...")
        params = _load_params(nc_file, n_ppd, seed + i)
        N = len(params["alpha_1"])
        print(f"[{label}] {N} samples — computing PPDs ...")

        # Primary mass p(m1 | Lambda), shape (N, M)
        pm1 = compute_ppd_m1(m1_grid, params)
        dm1 = np.diff(m1_grid)
        norm_m1 = 0.5 * np.sum(
            (pm1[:, :-1] + pm1[:, 1:]) * dm1, axis=1, keepdims=True
        )
        pm1_normed = pm1 / np.maximum(norm_m1, 1e-300)

        # Denser internal grid for q marginalisation
        pm1_q = compute_ppd_m1(m1_q_grid, params)
        dm1_q = np.diff(m1_q_grid)
        norm_m1_q = 0.5 * np.sum(
            (pm1_q[:, :-1] + pm1_q[:, 1:]) * dm1_q, axis=1, keepdims=True
        )
        pm1_q_normed = pm1_q / np.maximum(norm_m1_q, 1e-300)

        has_rate = "R" in params
        if has_rate:
            any_has_rate = True
            R_samples = params["R"]
            R_med = np.median(R_samples)
            R_lo, R_hi = np.percentile(R_samples, [5, 95])
            print(
                f"[{label}] R(z=0) = {R_med:.2f} [{R_lo:.2f}, {R_hi:.2f}] "
                f"Gpc^-3 yr^-1  (median, 90% CI)"
            )

            R_at_z = R_samples * (1.0 + Z_RATE) ** params["lamb"]
            R_z_med = np.median(R_at_z)
            R_z_lo, R_z_hi = np.percentile(R_at_z, [5, 95])
            print(
                f"[{label}] R(z={Z_RATE}) = {R_z_med:.2f} "
                f"[{R_z_lo:.2f}, {R_z_hi:.2f}] Gpc^-3 yr^-1  (median, 90% CI)"
            )

            if len(nc_files) > 1:
                rate_title_lines.append(
                    f"{label}: R = {R_med:.2f} [{R_lo:.2f}, {R_hi:.2f}]"
                )
            else:
                rate_title_lines.append(
                    f"R = {R_med:.2f} [{R_lo:.2f}, {R_hi:.2f}] Gpc^-3 yr^-1"
                )

            # dR/dm1(z=0.2) = R(z=0.2) * p(m1 | Lambda)
            dRdm1 = R_at_z[:, np.newaxis] * pm1_normed
            _plot_band(ax_m1, m1_grid, dRdm1, color, label)
        else:
            _plot_band(ax_m1, m1_grid, pm1_normed, color, label)

        # Mass ratio (marginalised over m1)
        pq = compute_ppd_q(q_grid, m1_q_grid, pm1_q_normed, params)
        _plot_band(ax_q, q_grid, pq, color, label)

        # Spin magnitude
        pa = compute_ppd_spin(a_grid, params)
        _plot_band(ax_a, a_grid, pa, color, label)

    # Formatting
    ax_m1.set_yscale("log")
    ax_m1.set_ylim(bottom=1e-3)
    ax_m1.set_xlabel(r"$m_1\ [M_\odot]$")
    if any_has_rate:
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
    ax_m1.set_xlim(MMIN, 100)

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
