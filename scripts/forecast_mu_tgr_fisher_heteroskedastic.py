#!/usr/bin/env python3
"""Forecast number of events to reach \Delta\mu_{\Lambda} < 1
using a heteroskadistic Fisher matrix analysis.

    uv run python scripts/forecast_mu_tgr_fisher_heteroskedastic.py \
        <PE file globs> \
        --memory-dir analysis \
        --posterior-nc <result_memory.nc> \
        --include-events-file <analyzed_events.txt> \
        --truth-mode conditional-sigma \
        --current-bns-range 266 \
        --bns-range-scenario O5a=375 \
        --outdir <outdir>
"""

from __future__ import annotations

import argparse
import math
import os
import re
from glob import glob
from pathlib import Path

import arviz as az
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MU_TRUE = 1.0
TARGET_HALFWIDTH = 1.0
MAX_POSTERIOR_DRAWS = 20_000
N_TRUTH_DRAWS = 4_000
MAX_SAMPLE_SNR = 10.0
MAX_EVENT_MEDIAN_SNR = 5.0
EXCLUDED_EVENTS = {
    "GW200105_162426",
    "GW200115_042309",
    "GW230518_125908",
    "GW230529_181500",
}


def parse_args() -> argparse.Namespace:
    """Parse only the options needed by the user's current command."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_paths", nargs="+", help="PE posterior-file glob pattern(s).")
    p.add_argument("--memory-dir", required=True, help="Directory containing {event}/memory_results.h5")
    p.add_argument("--posterior-nc", required=True, help="Current hierarchical posterior NetCDF")
    p.add_argument("--include-events-file", required=True, help="One event name per line")
    p.add_argument(
        "--truth-mode",
        choices=["conditional-sigma"],
        default="conditional-sigma",
        help="Kept only so your current command works unchanged.",
    )
    p.add_argument("--current-bns-range", type=float, required=True, help="Current BNS range in Mpc")
    p.add_argument(
        "--bns-range-scenario",
        action="append",
        default=[],
        required=True,
        help="Future scenario(s) as LABEL=RANGE_MPC, e.g. O5a=375",
    )
    p.add_argument("--outdir", required=True, help="Directory for CSV outputs and the forecast plot")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Event compression
# -----------------------------------------------------------------------------


def event_name_from_path(path: str) -> str | None:
    """Extract a GW event name like GW190521_074359 from a release filename."""
    m = re.search(r"(GW\d{6}_\d{6})", os.path.basename(path))
    return m.group(1) if m else None


def waveform_sort_key(label: str) -> tuple[int, str]:
    """Prefer larger calibration version numbers, then sort lexicographically."""
    m = re.match(r"C(\d+):", label)
    calib = int(m.group(1)) if m else -1
    return (-calib, label)


def pick_waveform_label(labels: list[str]) -> str:
    """Use the same automatic waveform preference as the longer script."""
    labels = sorted(labels, key=waveform_sort_key)
    for token in ["NRSur", "SEOB", "IMRPhenomXO4a", "IMRPhenomXPHM-SpinTaylor", "IMRPhenomXPHM"]:
        for label in labels:
            if token in label:
                return label
    return labels[0]


def normalize_logweights(logw: np.ndarray) -> np.ndarray:
    """Return normalized log-weights that sum to one in linear space."""
    return np.asarray(logw, float) - logsumexp(logw)



def event_loglike_grad_hess(
    A: float,
    a_hat: np.ndarray,
    a_sigma: np.ndarray,
    lw_norm: np.ndarray,
) -> tuple[float, float, float]:
    """Log-likelihood, gradient, and Hessian for the event-level Gaussian mixture.

    The per-event memory posterior approximation is

        p(A | event) ~ sum_s w_s N(A | A_hat_s, A_sigma_s).

    We need the local curvature of ``log p(A | event)`` at its mode in order to
    define one effective Gaussian width for the event.
    """
    z = (a_hat - A) / a_sigma
    log_comp = lw_norm - np.log(a_sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * z * z
    logL = float(logsumexp(log_comp))
    resp = np.exp(log_comp - logL)
    d1 = (a_hat - A) / (a_sigma**2)
    d2 = -1.0 / (a_sigma**2)
    grad = float(np.sum(resp * d1))
    hess = float(np.sum(resp * (d1**2 + d2)) - grad**2)
    return logL, grad, hess



def event_mode_and_sigma(a_hat: np.ndarray, a_sigma: np.ndarray, log_weight: np.ndarray) -> tuple[float, float]:
    """Compress one event to a mode and a local Gaussian width.

    The returned width is

        sigma_local = [- d^2/dA^2 log p(A|event) at A_mode]^{-1/2}.
    """
    lw_norm = normalize_logweights(log_weight)
    w = np.exp(lw_norm)
    mean = float(np.sum(w * a_hat))
    second_moment = float(np.sum(w * (a_sigma**2 + a_hat**2)))
    sigma_moment = math.sqrt(max(second_moment - mean**2, 1e-12))

    A = mean
    step_cap = max(1.0, sigma_moment)
    converged = False
    # Newton-Raphson optimizer to find mode of log p(A)
    for _ in range(30):
        _, grad, hess = event_loglike_grad_hess(A, a_hat, a_sigma, lw_norm)
        if not (np.isfinite(grad) and np.isfinite(hess)):
            break
        if abs(grad) < 1e-10:
            converged = True
            break
        if hess >= 0:
            break
        A_new = A - np.clip(grad / hess, -step_cap, step_cap)
        if abs(A_new - A) < 1e-8:
            A = float(A_new)
            converged = True
            break
        A = float(A_new)

    # brute force
    if not converged:
        span = max(5.0 * sigma_moment, 3.0 * float(np.median(a_sigma)), 1.0)
        grid = np.linspace(mean - span, mean + span, 401)
        vals = np.array([event_loglike_grad_hess(x, a_hat, a_sigma, lw_norm)[0] for x in grid])
        A = float(grid[int(np.argmax(vals))])

    _, _, hess = event_loglike_grad_hess(A, a_hat, a_sigma, lw_norm)
    info = float(np.sum(w / (a_sigma**2))) if (not np.isfinite(hess) or hess >= 0) else float(max(-hess, 1e-12))
    return A, 1.0 / math.sqrt(max(info, 1e-12))



def read_allowlist(path: str) -> set[str]:
    """Read the list of analyzed events provided to the command."""
    keep: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                keep.add(line)
    return keep



def load_event_table(data_paths: list[str], memory_dir: str, allowlist: set[str]) -> pd.DataFrame:
    """Load and compress all usable events into one table.

    Each output row contains:
    - ``event_name``: GW event name
    - ``waveform_label``: selected waveform group in the memory HDF5 file
    - ``center_mode``: mode of the event-level memory posterior mixture
    - ``sigma_local``: local Gaussian width used by the Fisher forecast
    - ``n_raw`` / ``n_kept``: sample counts before/after filtering
    """
    rows: list[dict[str, object]] = []
    files = sorted({path for pattern in data_paths for path in glob(pattern)})
    if not files:
        raise RuntimeError("No PE files matched the provided glob patterns.")

    for path in files:
        event = event_name_from_path(path)
        if event is None or event not in allowlist or event in EXCLUDED_EVENTS:
            continue

        mem_path = os.path.join(memory_dir, event, "memory_results.h5")
        if not os.path.exists(mem_path):
            continue

        with h5py.File(mem_path, "r") as f:
            labels = list(f.keys())
            if not labels:
                continue
            label = pick_waveform_label(labels)
            grp = f[label]
            a_hat = np.asarray(grp["A_hat"][()].real, float)
            a_sigma = np.asarray(grp["A_sigma"][()].real, float)
            log_weight = np.asarray(grp["log_weight"][()].real, float)

        if not (len(a_hat) == len(a_sigma) == len(log_weight)):
            continue
        n_raw = len(a_hat)

        # Preserve the same filtering logic as the longer script.
        snr = np.abs(a_hat / a_sigma)
        bad = snr > MAX_SAMPLE_SNR
        if np.any(bad):
            a_hat = a_hat.copy()
            a_sigma = a_sigma.copy()
            log_weight = log_weight.copy()
            a_hat[bad] = 0.0
            a_sigma[bad] = 1.0
            log_weight[bad] = -np.inf
            snr[bad] = 0.0
        if np.median(snr[np.isfinite(snr)]) > MAX_EVENT_MEDIAN_SNR:
            continue

        good = np.isfinite(a_hat) & np.isfinite(a_sigma) & np.isfinite(log_weight) & (a_sigma > 0)
        if not np.any(good):
            continue

        center_mode, sigma_local = event_mode_and_sigma(a_hat[good], a_sigma[good], log_weight[good])
        rows.append(
            {
                "event_name": event,
                "waveform_label": label,
                "center_mode": float(center_mode),
                "sigma_local": float(sigma_local),
                "n_raw": int(n_raw),
                "n_kept": int(np.sum(good)),
            }
        )

    if not rows:
        raise RuntimeError("No usable memory events were found.")
    return pd.DataFrame(rows).sort_values("event_name").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Posterior handling and calibration
# -----------------------------------------------------------------------------


def load_posterior(nc_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened posterior draws of (mu_tgr, sigma_tgr) from NetCDF."""
    fit = az.from_netcdf(nc_path)
    mu = np.asarray(fit.posterior["mu_tgr"].values, float).reshape(-1)
    sigma = np.asarray(fit.posterior["sigma_tgr"].values, float).reshape(-1)
    good = np.isfinite(mu) & np.isfinite(sigma) & (sigma >= 0)
    mu, sigma = mu[good], sigma[good]
    if len(mu) == 0:
        raise RuntimeError("No finite mu_tgr/sigma_tgr samples found in the NetCDF posterior.")
    if len(mu) > MAX_POSTERIOR_DRAWS:
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(mu), size=MAX_POSTERIOR_DRAWS, replace=False)
        mu, sigma = mu[idx], sigma[idx]
    return mu, sigma



def posterior_halfwidth68(values: np.ndarray) -> float:
    """Return the central 68% half-width: (q84 - q16) / 2."""
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return 0.5 * float(q84 - q16)



def fisher_mu_std(sigmas: np.ndarray, sigma_pop: float) -> float:
    """Heteroskedastic Fisher width on the population mean.

    For event-level widths ``sigma_i`` and population width ``sigma_pop``,

        std(mu_tgr) ~= [sum_i 1 / (sigma_i^2 + sigma_pop^2)]^(-1/2).
    """
    tau2 = np.asarray(sigmas, float) ** 2 + float(sigma_pop) ** 2
    return 1.0 / math.sqrt(float(np.sum(1.0 / np.maximum(tau2, 1e-300))))



def solve_scale(sigmas_raw: np.ndarray, target_width: float, sigma_ref: float) -> float:
    """Solve for the calibration factor alpha by bisection.

    We choose ``alpha`` so that the calibrated event widths ``alpha * sigma_i``
    reproduce the current posterior half-width on ``mu_tgr`` under the
    heteroskedastic Fisher formula.
    """
    def width(alpha: float) -> float:
        return fisher_mu_std(alpha * sigmas_raw, sigma_ref)

    lo, hi = 0.0, 1.0
    if target_width < width(0.0) * (1.0 - 1e-10):
        raise RuntimeError("Requested current width is below the sigma_ref-imposed minimum.")
    while width(hi) < target_width:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if width(mid) < target_width:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def conditional_sigma_draws(mu: np.ndarray, sigma: np.ndarray, mu_true: float, n: int) -> np.ndarray:
    """Sample sigma_tgr from an approximate posterior conditioned on mu_tgr = mu_true.

    Conditioning is implemented with Gaussian kernel reweighting in ``mu_tgr``:

        w_k ∝ exp[-(mu_k - mu_true)^2 / (2 bw^2)],

    with bandwidth ``bw = max(0.05, 0.25 * std(mu))``.
    """
    bw = max(0.05, 0.25 * float(np.std(mu)))
    z = (mu - mu_true) / bw
    logw = -0.5 * z * z
    logw -= logsumexp(logw)
    rng = np.random.default_rng(12345)
    return sigma[rng.choice(len(sigma), size=n, replace=True, p=np.exp(logw))]


# -----------------------------------------------------------------------------
# Forecast helpers
# -----------------------------------------------------------------------------


def default_n_grid(n_current: int, current_width: float, target_width: float) -> np.ndarray:
    """Construct a log-spaced total-N grid."""
    guess = max(float(n_current), n_current * (current_width / target_width) ** 2)
    hi = max(float(n_current + 100), 2.0 * guess)
    grid = np.unique(np.round(np.geomspace(max(n_current, 1), hi, 14)).astype(int))
    grid = grid[grid >= n_current]
    return grid if len(grid) and grid[0] == n_current else np.concatenate([[n_current], grid])



def parse_bns_scenarios(current_bns_range: float, specs: list[str]) -> dict[str, float]:
    """Convert LABEL=RANGE_MPC into detector-sensitivity gains.

    The baseline scenario is always the current detector sensitivity, i.e. gain 1.
    Each future scenario gain is ``future_range / current_range``.
    """
    scenarios = {"current": 1.0}
    for spec in specs:
        label, raw = spec.split("=", 1)
        scenarios[label.strip()] = float(raw) / float(current_bns_range)
    return scenarios



def summarize(values: np.ndarray) -> tuple[float, float, float]:
    """Return the 16th, 50th, and 84th percentiles of a finite-valued array."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return tuple(np.quantile(values, [0.16, 0.50, 0.84])) if len(values) else (np.nan, np.nan, np.nan)


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    allowlist = read_allowlist(args.include_events_file)
    event_df = load_event_table(args.data_paths, args.memory_dir, allowlist)
    mu_draws, sigma_draws = load_posterior(args.posterior_nc)

    current_halfwidth = posterior_halfwidth68(mu_draws)
    sigma_ref = float(np.median(sigma_draws))
    raw_sigmas = event_df["sigma_local"].to_numpy(float)
    scale = solve_scale(raw_sigmas, current_halfwidth, sigma_ref)
    calibrated_sigmas = scale * raw_sigmas
    event_df["sigma_local_calibrated"] = calibrated_sigmas
    event_df.to_csv(outdir / "event_quality_summaries.csv", index=False)

    sigma_truth = conditional_sigma_draws(mu_draws, sigma_draws, MU_TRUE, N_TRUTH_DRAWS)
    n_current = len(calibrated_sigmas)
    n_grid = default_n_grid(n_current, current_halfwidth, TARGET_HALFWIDTH)
    scenarios = parse_bns_scenarios(args.current_bns_range, args.bns_range_scenario)

    summary_rows: list[dict[str, float | str | int]] = []
    crossing_rows: list[dict[str, float | str]] = []

    for label, gain in scenarios.items():
        future_sigmas = calibrated_sigmas / float(gain)

        # Anchor the curve at the actual current posterior width, not at the
        # Fisher approximation. This mirrors the longer script.
        summary_rows.append(
            {
                "scenario": label,
                "gain": float(gain),
                "n_total": int(n_current),
                "mu_halfwidth68_p16": float(current_halfwidth),
                "mu_halfwidth68_p50": float(current_halfwidth),
                "mu_halfwidth68_p84": float(current_halfwidth),
                "sigma_true_p16": float(np.quantile(sigma_truth, 0.16)),
                "sigma_true_p50": float(np.quantile(sigma_truth, 0.50)),
                "sigma_true_p84": float(np.quantile(sigma_truth, 0.84)),
                "meets_target_fraction": float(current_halfwidth <= TARGET_HALFWIDTH),
                "row_type": "current_posterior",
            }
        )

        # Loop over possible sigma draws from \sigma_{\Lambda}
        # not actually used for anything
        crossing_per_draw = []
        for sigma_true in sigma_truth:
            info_current = np.sum(1.0 / (calibrated_sigmas**2 + sigma_true**2))
            info_future_per_event = float(np.mean(1.0 / (future_sigmas**2 + sigma_true**2)))
            target_info = 1.0 / TARGET_HALFWIDTH**2
            if info_current >= target_info:
                crossing_per_draw.append(float(n_current))
            else:
                crossing_per_draw.append(float(n_current + math.ceil((target_info - info_current) / info_future_per_event)))

        # Compute \Delta\mu_{\Lambda} at each catalog size
        for n_total in n_grid[1:]:
            n_new = int(n_total - n_current)
            widths = np.array(
                [1.0 / math.sqrt(np.sum(1.0 / (calibrated_sigmas**2 + s**2)) + n_new * np.mean(1.0 / (future_sigmas**2 + s**2))) for s in sigma_truth]
            )
            w16, w50, w84 = summarize(widths)
            s16, s50, s84 = summarize(sigma_truth)
            summary_rows.append(
                {
                    "scenario": label,
                    "gain": float(gain),
                    "n_total": int(n_total),
                    "mu_halfwidth68_p16": float(w16),
                    "mu_halfwidth68_p50": float(w50),
                    "mu_halfwidth68_p84": float(w84),
                    "sigma_true_p16": float(s16),
                    "sigma_true_p50": float(s50),
                    "sigma_true_p84": float(s84),
                    "meets_target_fraction": float(np.mean(widths <= TARGET_HALFWIDTH)),
                    "row_type": "fisher_forecast",
                }
            )

        c16, c50, c84 = summarize(np.array(crossing_per_draw))
        crossing_rows.append({"scenario": label, "gain": float(gain), "crossing_total_n_p16": c16, "crossing_total_n_p50": c50, "crossing_total_n_p84": c84})

    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "n_total", "row_type"])
    crossings_df = pd.DataFrame(crossing_rows).sort_values("scenario")
    assumptions_df = pd.DataFrame(
        {
            "key": [
                "mu_true",
                "target_halfwidth",
                "truth_mode",
                "waveform_selection",
                "current_catalog_size",
                "current_mu_halfwidth68",
                "calibration_sigma_ref_median",
                "calibration_scale",
                "max_posterior_draws",
                "n_truth_draws",
            ],
            "value": [
                MU_TRUE,
                TARGET_HALFWIDTH,
                args.truth_mode,
                "auto",
                n_current,
                current_halfwidth,
                sigma_ref,
                scale,
                MAX_POSTERIOR_DRAWS,
                N_TRUTH_DRAWS,
            ],
        }
    )

    summary_df.to_csv(outdir / "fisher_forecast_summary.csv", index=False)
    crossings_df.to_csv(outdir / "fisher_forecast_crossings.csv", index=False)
    assumptions_df.to_csv(outdir / "fisher_forecast_assumptions.csv", index=False)

    # Plot median forecast curves with p16-p84 bands.
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, group in summary_df.groupby("scenario", sort=False):
        group = group.sort_values("n_total")
        x = group["n_total"].to_numpy(float)
        y16 = group["mu_halfwidth68_p16"].to_numpy(float)
        y50 = group["mu_halfwidth68_p50"].to_numpy(float)
        y84 = group["mu_halfwidth68_p84"].to_numpy(float)
        ax.fill_between(x, y16, y84, alpha=0.12)
        ax.plot(x, y50, marker="o", label=f"{label} (gain {group['gain'].iloc[0]:.2f})")

    ax.axhline(TARGET_HALFWIDTH, linestyle="--", linewidth=1.5, label=f"target = {TARGET_HALFWIDTH:g}")
    ax.axvline(n_current, linestyle=":", linewidth=1.5, label=f"current N = {n_current}")
    ax.set_xlabel("Total number of memory-analyzed events")
    ax.set_ylabel(r"68% half-width on $\mu_A$")
    ax.set_title("Heteroskedastic Fisher forecast for the mean memory-enhancement factor")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "forecast_mu_tgr_fisher_heteroskedastic.png", dpi=200)
    plt.close(fig)

    print(f"Current catalog size: {n_current}")
    print(f"Current mu_tgr half-width68: {current_halfwidth:.3f}")
    print(f"Calibration sigma_ref (posterior median sigma_tgr): {sigma_ref:.3f}")
    print(f"Calibration scale alpha: {scale:.3f}")
    for row in crossing_rows:
        print(
            f"{row['scenario']}: crossing total N (p16, p50, p84) = "
            f"({row['crossing_total_n_p16']:.1f}, {row['crossing_total_n_p50']:.1f}, {row['crossing_total_n_p84']:.1f})"
        )
    print(f"Wrote forecast products to {outdir}")


if __name__ == "__main__":
    main()
