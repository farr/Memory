#!/usr/bin/env python3
"""Plot 1D Population Predictive Distributions (PPDs) from hierarchical analysis.

For each input NetCDF file, computes and overlays the PPD (median + 90% CI) for:
  - Primary mass  p(m1)  on semi-log-y axes
  - Mass ratio    p(q)   marginalised over m1
  - Spin magnitude p(a)

Usage:
    python scripts/plot_ppd.py result_astro.nc
    python scripts/plot_ppd.py result_astro.nc result_joint.nc --labels astro joint
    python scripts/plot_ppd.py result_astro.nc --outdir /path/to/dir --n-ppd 1000
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
import arviz as az

MMIN = 5.0
MMAX = 100.0


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

    # Gaussian peaks
    log_g1 = scipy_norm.logpdf(m, m1, s1)
    log_g2 = scipy_norm.logpdf(m, m2, s2)

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
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D PPDs from hierarchical population analysis"
    )
    parser.add_argument(
        "nc_files", nargs="+",
        help="ArviZ NetCDF result file(s) (e.g. result_astro.nc)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Legend labels for each file (default: stem of filename)",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory (default: directory of first input file)",
    )
    parser.add_argument(
        "--n-ppd", type=int, default=None,
        help="Max posterior samples to use (default: all); reduces memory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.nc_files):
        parser.error("--labels must match the number of nc_files")

    labels = args.labels or [
        os.path.splitext(os.path.basename(f))[0] for f in args.nc_files
    ]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.nc_files[0]))
    os.makedirs(outdir, exist_ok=True)

    # Evaluation grids.
    # m1_grid excludes MMIN exactly: at m1=MMIN the q-distribution is a Dirac delta
    # at q=1 and its normalisation diverges, causing float64 overflow in the integral.
    # Dropping that single measure-zero point has no effect on the integral value.
    m1_grid = np.linspace(MMIN, MMAX, 201)[1:]   # 200 pts, first at ~5.48 M☉
    q_grid  = np.linspace(0.01, 1.0, 150)
    a_grid  = np.linspace(0.0,  1.0, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_m1, ax_q, ax_a = axes

    for i, (nc_file, label) in enumerate(zip(args.nc_files, labels)):
        color = f"C{i}"
        print(f"[{label}] Loading {nc_file} ...")
        params = _load_params(nc_file, args.n_ppd, args.seed + i)
        N = len(params["alpha_1"])
        print(f"[{label}] {N} samples — computing PPDs ...")

        # Primary mass
        pm1 = compute_ppd_m1(m1_grid, params)                    # (N, M)
        dm1 = np.diff(m1_grid)
        norm_m1 = 0.5 * np.sum(
            (pm1[:, :-1] + pm1[:, 1:]) * dm1, axis=1, keepdims=True
        )
        pm1_normed = pm1 / np.maximum(norm_m1, 1e-300)
        _plot_band(ax_m1, m1_grid, pm1_normed, color, label)

        # Mass ratio (marginalised over m1)
        pq = compute_ppd_q(q_grid, m1_grid, pm1_normed, params)  # (N, Q)
        _plot_band(ax_q, q_grid, pq, color, label)

        # Spin magnitude
        pa = compute_ppd_spin(a_grid, params)                     # (N, A)
        _plot_band(ax_a, a_grid, pa, color, label)

    # Formatting
    ax_m1.set_yscale("log")
    ax_m1.set_xlabel(r"$m_1\ [M_\odot]$")
    ax_m1.set_ylabel(r"$p(m_1)\ [M_\odot^{-1}]$")
    ax_m1.set_title("Primary mass")
    ax_m1.set_xlim(MMIN, MMAX)  # show full range even though grid starts at 5.48

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

    if len(args.nc_files) > 1:
        for ax in axes:
            ax.legend()

    plt.tight_layout()
    out_path = os.path.join(outdir, "ppd.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
