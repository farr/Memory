#!/usr/bin/env python3
r"""Forecast future constraints on mu_tgr with multiple detector-sensitivity scenarios.

This is an extension of ``forecast_mu_tgr_sequential.py`` that can draw several
curves on the same plot, e.g. a GWTC-4-like curve and one or more O5 proxy
curves.

Core idea
---------
Detector sensitivity should enter through the *event-level measurement widths*,
not through the population width ``sigma_tgr``. For a detector-sensitivity gain
``g > 1``, this script approximates

    sigma_event,new = sigma_event,current / g

for future events, while keeping the current posterior ``p(mu_tgr, sigma_tgr |
current data)`` fixed as the starting point.

This is the right place to encode an O5 sensitivity jump if your x-axis is the
*number of analyzed events*. If instead you want a forecast in *observing time*,
you should also fold in the higher event rate expected from the larger sensitive
volume.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Sequence

import arviz as az
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

LOG = logging.getLogger("forecast_mu_tgr_sensitivity_curves")

EXCLUDED_EVENTS = {
    "GW200105_162426",
    "GW200115_042309",
    "GW230518_125908",
    "GW230529_181500",
}
MAX_SAMPLE_SNR = 10.0
MAX_EVENT_MEDIAN_SNR = 5.0


@dataclass(frozen=True)
class EventQuality:
    event_name: str
    waveform_label: str
    center_mode: float
    sigma_local: float
    sigma_moment: float
    n_raw: int
    n_kept: int


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Forecast the total number of memory-analyzed events needed to constrain "
            "mu_tgr to a target 68% half-width, for one or more detector-sensitivity scenarios."
        )
    )
    p.add_argument("data_paths", nargs="+", help="PE posterior-file glob pattern(s).")
    p.add_argument("--memory-dir", required=True, help="Directory containing {event_name}/memory_results.h5")
    p.add_argument("--posterior-nc", required=True, help="Current hierarchical posterior NetCDF.")
    p.add_argument("--waveform", default="auto", help="Waveform selection rule; default 'auto'.")
    p.add_argument("--include-events-file", help="Optional file with one event name per line.")
    p.add_argument("--outdir", default="forecast_mu_tgr_sensitivity_curves", help="Output directory.")
    p.add_argument("--seed", type=int, default=12345, help="Random seed.")
    p.add_argument("--n-mocks", type=int, default=300, help="Mock catalogs per N value.")
    p.add_argument(
        "--n-total",
        type=int,
        nargs="*",
        help=(
            "Total catalog sizes to evaluate. If omitted, a log-spaced grid is constructed "
            "around the 1/sqrt(N) scaling estimate."
        ),
    )
    p.add_argument("--mu-true", type=float, default=1.0, help="Injected true population mean.")
    p.add_argument(
        "--truth-mode",
        choices=["strict-gr", "fixed-sigma", "posterior-sigma", "conditional-sigma"],
        default="conditional-sigma",
        help="How to choose the injected true population width sigma_tgr.",
    )
    p.add_argument("--fixed-sigma", type=float, default=0.0, help="Used only with --truth-mode fixed-sigma.")
    p.add_argument(
        "--condition-bandwidth",
        type=float,
        default=None,
        help="Kernel width in mu for approximate conditioning when using conditional-sigma.",
    )
    p.add_argument("--target-halfwidth", type=float, default=1.0, help="Target 68% half-width on mu_tgr.")
    p.add_argument("--metric", choices=["halfwidth68", "std"], default="halfwidth68")
    p.add_argument("--no-calibrate-widths", action="store_true")
    p.add_argument("--max-posterior-draws", type=int, default=20000)
    p.add_argument("--plot-title", default=None)
    p.add_argument("--verbose", action="store_true")

    # New: multiple sensitivity scenarios.
    p.add_argument(
        "--gain-scenario",
        action="append",
        default=[],
        help=(
            "Additional scenario in the form LABEL=GAIN, where GAIN>1 means future event-level "
            "measurement widths are divided by GAIN. Example: --gain-scenario O5proxy=1.7"
        ),
    )
    p.add_argument(
        "--current-label",
        default="GWTC-4-like",
        help="Legend label for the baseline current-sensitivity curve.",
    )
    p.add_argument(
        "--current-bns-range",
        type=float,
        default=None,
        help=(
            "Optional current single-detector BNS range in Mpc. If supplied together with "
            "--bns-range-scenario, gains are computed as range/current_range."
        ),
    )
    p.add_argument(
        "--bns-range-scenario",
        action="append",
        default=[],
        help=(
            "Additional scenario in the form LABEL=RANGE_MPC. Requires --current-bns-range. "
            "Example: --current-bns-range 160 --bns-range-scenario O5a=225"
        ),
    )
    return p


def _event_name_from_path(path: str) -> str | None:
    m = re.search(r"(GW\d{6}_\d{6})", os.path.basename(path))
    return m.group(1) if m else None


def _waveform_sort_key(label: str) -> tuple[int, str]:
    m = re.match(r"C(\d+):", label)
    calib = int(m.group(1)) if m else -1
    return (-calib, label)


def _pick_waveform_label(keys: Sequence[str]) -> str:
    keys_sorted = sorted(keys, key=_waveform_sort_key)
    for key in keys_sorted:
        if "NRSur" in key:
            return key
    for key in keys_sorted:
        if "SEOB" in key:
            return key
    return keys_sorted[0]


def _resolve_waveform_label(keys: Sequence[str], waveform: str | None) -> str:
    if not keys:
        raise KeyError("No waveform labels available")
    if waveform is None or waveform == "auto":
        return _pick_waveform_label(keys)
    if waveform in keys:
        return waveform
    requested = waveform
    if re.match(r"C\d+:", requested):
        requested = requested.split(":", 1)[1]
    matches = [key for key in keys if key.split(":", 1)[-1] == requested]
    if matches:
        return sorted(matches, key=_waveform_sort_key)[0]
    raise KeyError(f"Waveform '{waveform}' not found; available labels: {list(keys)}")


def _collect_event_files(patterns: Sequence[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob(pattern))
    seen: set[str] = set()
    unique: list[str] = []
    for path in sorted(files):
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _read_event_allowlist(path: str | None) -> set[str] | None:
    if path is None:
        return None
    keep: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keep.add(line)
    return keep


def _normalize_logweights(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=float)
    return logw - logsumexp(logw)


def _event_loglike_grad_hess(A: float, a_hat: np.ndarray, a_sigma: np.ndarray, lw_norm: np.ndarray) -> tuple[float, float, float]:
    z = (a_hat - A) / a_sigma
    log_comp = lw_norm - np.log(a_sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * z * z
    logL = float(logsumexp(log_comp))
    resp = np.exp(log_comp - logL)
    d1 = (a_hat - A) / (a_sigma**2)
    d2 = -1.0 / (a_sigma**2)
    grad = float(np.sum(resp * d1))
    hess = float(np.sum(resp * (d1**2 + d2)) - grad**2)
    return logL, grad, hess


def _event_mode_and_local_sigma(a_hat: np.ndarray, a_sigma: np.ndarray, log_weight: np.ndarray) -> tuple[float, float, float]:
    lw_norm = _normalize_logweights(log_weight)
    w = np.exp(lw_norm)

    mean = float(np.sum(w * a_hat))
    second_moment = float(np.sum(w * (a_sigma**2 + a_hat**2)))
    var_moment = max(second_moment - mean**2, 1e-12)
    sigma_moment = math.sqrt(var_moment)

    A = mean
    step_cap = max(1.0, sigma_moment)
    converged = False
    for _ in range(30):
        _, grad, hess = _event_loglike_grad_hess(A, a_hat, a_sigma, lw_norm)
        if not (np.isfinite(grad) and np.isfinite(hess)):
            break
        if abs(grad) < 1e-10:
            converged = True
            break
        if hess >= 0:
            break
        step = -grad / hess
        step = float(np.clip(step, -step_cap, step_cap))
        A_new = A + step
        if abs(A_new - A) < 1e-8:
            A = A_new
            converged = True
            break
        A = A_new

    if not converged:
        span = max(5.0 * sigma_moment, 3.0 * float(np.median(a_sigma)), 1.0)
        grid = np.linspace(mean - span, mean + span, 401)
        vals = np.array([_event_loglike_grad_hess(x, a_hat, a_sigma, lw_norm)[0] for x in grid])
        A = float(grid[int(np.argmax(vals))])

    _, _, hess = _event_loglike_grad_hess(A, a_hat, a_sigma, lw_norm)
    if not np.isfinite(hess) or hess >= 0:
        info = float(np.sum(w / (a_sigma**2)))
    else:
        info = float(max(-hess, 1e-12))
    sigma_local = 1.0 / math.sqrt(max(info, 1e-12))
    return A, sigma_local, sigma_moment


def _load_event_qualities(
    event_files: Sequence[str],
    memory_dir: str,
    waveform: str | None = "auto",
    allowlist: set[str] | None = None,
) -> list[EventQuality]:
    qualities: list[EventQuality] = []
    n_missing = 0

    for path in event_files:
        event_name = _event_name_from_path(path)
        if event_name is None:
            LOG.warning("Skipping %s: could not parse GW event name", path)
            continue
        if allowlist is not None and event_name not in allowlist:
            continue
        if event_name in EXCLUDED_EVENTS:
            LOG.info("Skipping %s: explicitly excluded", event_name)
            continue

        mem_path = os.path.join(memory_dir, event_name, "memory_results.h5")
        if not os.path.exists(mem_path):
            n_missing += 1
            continue

        with h5py.File(mem_path, "r") as f:
            keys = list(f.keys())
            if not keys:
                continue
            try:
                label = _resolve_waveform_label(keys, waveform)
            except KeyError as exc:
                LOG.warning("Skipping %s: %s", event_name, exc)
                continue
            grp = f[label]
            a_hat = np.asarray(grp["A_hat"][()].real, dtype=float)
            a_sigma = np.asarray(grp["A_sigma"][()].real, dtype=float)
            log_weight = np.asarray(grp["log_weight"][()].real, dtype=float)

        n_raw = len(a_hat)
        if not (len(a_hat) == len(a_sigma) == len(log_weight)):
            LOG.warning("Skipping %s: inconsistent array lengths", event_name)
            continue

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

        finite_snr = snr[np.isfinite(snr)]
        median_snr = float(np.median(finite_snr)) if len(finite_snr) else 0.0
        if median_snr > MAX_EVENT_MEDIAN_SNR:
            LOG.info(
                "Skipping %s: median |A_hat/A_sigma|=%.3f exceeds %.1f",
                event_name,
                median_snr,
                MAX_EVENT_MEDIAN_SNR,
            )
            continue

        good = np.isfinite(a_hat) & np.isfinite(a_sigma) & np.isfinite(log_weight) & (a_sigma > 0)
        if not np.any(good):
            LOG.warning("Skipping %s: no finite samples after filtering", event_name)
            continue

        a_hat = a_hat[good]
        a_sigma = a_sigma[good]
        log_weight = log_weight[good]
        center_mode, sigma_local, sigma_moment = _event_mode_and_local_sigma(a_hat, a_sigma, log_weight)
        qualities.append(
            EventQuality(
                event_name=event_name,
                waveform_label=label,
                center_mode=float(center_mode),
                sigma_local=float(sigma_local),
                sigma_moment=float(sigma_moment),
                n_raw=int(n_raw),
                n_kept=int(np.sum(good)),
            )
        )

    if n_missing:
        LOG.info("Skipped %d event files without matching memory_results.h5", n_missing)
    if not qualities:
        raise RuntimeError("No usable memory events were found.")
    return qualities


def _flatten_posterior_vars(nc_path: str) -> tuple[np.ndarray, np.ndarray]:
    fit = az.from_netcdf(nc_path)
    if "mu_tgr" not in fit.posterior or "sigma_tgr" not in fit.posterior:
        raise ValueError(f"{nc_path} does not contain mu_tgr and sigma_tgr")
    mu = np.asarray(fit.posterior["mu_tgr"].values, dtype=float).reshape(-1)
    sigma = np.asarray(fit.posterior["sigma_tgr"].values, dtype=float).reshape(-1)
    good = np.isfinite(mu) & np.isfinite(sigma) & (sigma >= 0)
    if not np.any(good):
        raise ValueError(f"No finite mu_tgr/sigma_tgr samples found in {nc_path}")
    return mu[good], sigma[good]


def _downsample_draws(rng: np.random.Generator, mu: np.ndarray, sigma: np.ndarray, max_draws: int) -> tuple[np.ndarray, np.ndarray]:
    if len(mu) <= max_draws:
        return mu, sigma
    idx = rng.choice(len(mu), size=max_draws, replace=False)
    return mu[idx], sigma[idx]


def _choose_sigma_true(
    rng: np.random.Generator,
    mode: str,
    mu_true: float,
    fixed_sigma: float,
    posterior_mu: np.ndarray,
    posterior_sigma: np.ndarray,
    condition_bandwidth: float | None,
) -> float:
    if mode == "strict-gr":
        return 0.0
    if mode == "fixed-sigma":
        return max(0.0, float(fixed_sigma))
    if mode == "posterior-sigma":
        idx = int(rng.integers(0, len(posterior_sigma)))
        return float(posterior_sigma[idx])
    if mode != "conditional-sigma":
        raise ValueError(f"Unknown truth mode: {mode}")

    bw = condition_bandwidth
    if bw is None:
        bw = max(0.05, 0.25 * float(np.std(posterior_mu)))
    if bw <= 0:
        bw = 0.05

    z = (posterior_mu - mu_true) / bw
    logw = -0.5 * z * z
    logw -= logsumexp(logw)
    w = np.exp(logw)
    idx = int(rng.choice(len(posterior_sigma), p=w))
    return float(posterior_sigma[idx])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probs = np.asarray(probs, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[good]
    weights = weights[good]
    if len(values) == 0 or np.sum(weights) <= 0:
        return np.full(len(probs), np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cum /= cum[-1]
    cum_mid = cum - 0.5 * weights / np.sum(weights)
    cum_mid[0] = max(cum_mid[0], 0.0)
    cum_mid[-1] = min(cum_mid[-1], 1.0)
    return np.interp(probs, cum_mid, values)


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[good]
    weights = weights[good]
    if len(values) == 0 or np.sum(weights) <= 0:
        return {k: np.nan for k in ["mu_mean", "mu_std", "mu_q16", "mu_q50", "mu_q84", "mu_halfwidth68"]}
    weights = weights / np.sum(weights)
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    q16, q50, q84 = _weighted_quantile(values, weights, [0.16, 0.50, 0.84])
    return {
        "mu_mean": mean,
        "mu_std": math.sqrt(max(var, 0.0)),
        "mu_q16": float(q16),
        "mu_q50": float(q50),
        "mu_q84": float(q84),
        "mu_halfwidth68": 0.5 * float(q84 - q16),
    }


def _unweighted_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {k: np.nan for k in ["mu_mean", "mu_std", "mu_q16", "mu_q50", "mu_q84", "mu_halfwidth68"]}
    q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "mu_mean": float(np.mean(values)),
        "mu_std": float(np.std(values, ddof=0)),
        "mu_q16": float(q16),
        "mu_q50": float(q50),
        "mu_q84": float(q84),
        "mu_halfwidth68": 0.5 * float(q84 - q16),
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    if len(w) == 0 or np.sum(w) <= 0:
        return 0.0
    w = w / np.sum(w)
    return float(1.0 / np.sum(w * w))


def _simulate_mock_catalog_summary(
    rng: np.random.Generator,
    base_sigmas: np.ndarray,
    n_new: int,
    mu_true: float,
    sigma_true: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_base = len(base_sigmas)
    counts = np.zeros(n_base, dtype=int)
    sum_m = np.zeros(n_base, dtype=float)
    sum_m2 = np.zeros(n_base, dtype=float)
    if n_new <= 0:
        return counts, sum_m, sum_m2

    idx = rng.integers(0, n_base, size=n_new)
    counts = np.bincount(idx, minlength=n_base)
    for i, n_i in enumerate(counts):
        if n_i == 0:
            continue
        var_true = float(base_sigmas[i] ** 2 + sigma_true ** 2)
        m = rng.normal(loc=mu_true, scale=math.sqrt(var_true), size=n_i)
        sum_m[i] = float(np.sum(m))
        sum_m2[i] = float(np.sum(m * m))
    return counts, sum_m, sum_m2


def _update_current_posterior_with_mock(
    mu_draws: np.ndarray,
    sigma_draws: np.ndarray,
    base_sigmas: np.ndarray,
    counts: np.ndarray,
    sum_m: np.ndarray,
    sum_m2: np.ndarray,
) -> tuple[np.ndarray, float]:
    logw = np.zeros(len(mu_draws), dtype=float)
    active = np.where(counts > 0)[0]
    if len(active) == 0:
        w = np.full(len(mu_draws), 1.0 / len(mu_draws), dtype=float)
        return w, float(len(mu_draws))

    mu = mu_draws
    sigma = sigma_draws
    for i in active:
        n_i = float(counts[i])
        s2 = float(base_sigmas[i] ** 2)
        v = s2 + sigma * sigma
        logw += -0.5 * n_i * np.log(2.0 * np.pi * v)
        quad = sum_m2[i] - 2.0 * mu * sum_m[i] + n_i * mu * mu
        logw += -0.5 * quad / v

    logw -= logsumexp(logw)
    w = np.exp(logw)
    ess = _effective_sample_size(w)
    return w, ess


def _default_n_grid(n_current: int, current_width: float, target: float) -> list[int]:
    if not (np.isfinite(current_width) and current_width > 0 and target > 0):
        return [n_current, n_current + 100, n_current + 300, n_current + 1000]
    guess = max(float(n_current), n_current * (current_width / target) ** 2)
    hi = max(float(n_current + 100), 2.0 * guess)
    vals = np.unique(np.round(np.geomspace(max(n_current, 1), hi, num=14)).astype(int))
    vals = vals[vals >= n_current]
    if vals[0] != n_current:
        vals = np.concatenate([[n_current], vals])
    return [int(x) for x in vals]


def _parse_gain_scenarios(args: argparse.Namespace) -> OrderedDict[str, float]:
    scenarios: OrderedDict[str, float] = OrderedDict()
    scenarios[args.current_label] = 1.0

    for spec in args.gain_scenario:
        if "=" not in spec:
            raise ValueError(f"Invalid --gain-scenario '{spec}'; expected LABEL=GAIN")
        label, raw = spec.split("=", 1)
        label = label.strip()
        gain = float(raw)
        if gain <= 0:
            raise ValueError(f"Scenario gain must be positive: {spec}")
        scenarios[label] = gain

    if args.bns_range_scenario:
        if args.current_bns_range is None or args.current_bns_range <= 0:
            raise ValueError("--current-bns-range must be supplied and positive when using --bns-range-scenario")
        for spec in args.bns_range_scenario:
            if "=" not in spec:
                raise ValueError(f"Invalid --bns-range-scenario '{spec}'; expected LABEL=RANGE_MPC")
            label, raw = spec.split("=", 1)
            label = label.strip()
            rng_mpc = float(raw)
            if rng_mpc <= 0:
                raise ValueError(f"Scenario BNS range must be positive: {spec}")
            scenarios[label] = rng_mpc / float(args.current_bns_range)

    return scenarios


def _seed_for(base_seed: int, n_total: int, mock_id: int) -> int:
    return int((np.uint64(base_seed) + np.uint64(1000003) * np.uint64(n_total) + np.uint64(9176) * np.uint64(mock_id)) % np.uint64(2**63 - 1))


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    event_files = _collect_event_files(args.data_paths)
    if not event_files:
        raise RuntimeError("No event files matched the provided patterns.")

    allowlist = _read_event_allowlist(args.include_events_file)
    qualities = _load_event_qualities(event_files=event_files, memory_dir=args.memory_dir, waveform=args.waveform, allowlist=allowlist)
    quality_df = pd.DataFrame([q.__dict__ for q in qualities]).sort_values("event_name")
    quality_df["moment_over_local"] = quality_df["sigma_moment"] / quality_df["sigma_local"]
    quality_df.to_csv(outdir / "event_quality_summaries.csv", index=False)

    mu_draws_full, sigma_draws_full = _flatten_posterior_vars(args.posterior_nc)
    mu_draws, sigma_draws = _downsample_draws(rng, mu_draws_full, sigma_draws_full, args.max_posterior_draws)

    current_stats = _unweighted_stats(mu_draws)
    current_metric_value = current_stats["mu_halfwidth68"] if args.metric == "halfwidth68" else current_stats["mu_std"]

    base_sigmas_raw = quality_df["sigma_local"].to_numpy(dtype=float)
    sigma_moment = quality_df["sigma_moment"].to_numpy(dtype=float)
    n_current = len(base_sigmas_raw)

    raw_gaussian_hw = 1.0 / math.sqrt(np.sum(1.0 / np.maximum(base_sigmas_raw, 1e-12) ** 2))
    width_scale = 1.0 if (args.no_calibrate_widths or not np.isfinite(current_metric_value) or current_metric_value <= 0) else float(current_metric_value / max(raw_gaussian_hw, 1e-12))
    base_sigmas_current = base_sigmas_raw * width_scale

    quality_df["sigma_local_calibrated"] = base_sigmas_current
    quality_df.to_csv(outdir / "event_quality_summaries.csv", index=False)

    median_ratio = float(np.median(sigma_moment / np.maximum(base_sigmas_raw, 1e-12)))
    quick_guess = n_current * (current_metric_value / args.target_halfwidth) ** 2
    n_grid = sorted(set(args.n_total or _default_n_grid(n_current, current_metric_value, args.target_halfwidth)))
    n_grid = [n for n in n_grid if n >= n_current]
    if not n_grid:
        raise ValueError("All requested --n-total values are smaller than the current catalog size.")

    scenarios = _parse_gain_scenarios(args)
    metric_col = "mu_halfwidth68" if args.metric == "halfwidth68" else "mu_std"

    all_records: list[dict[str, float | int | str]] = []
    all_summary_rows: list[dict[str, float | int | str]] = []
    crossing_rows: list[dict[str, float | int | str]] = []

    for scenario_label, gain in scenarios.items():
        base_sigmas = base_sigmas_current / float(gain)
        scenario_records: list[dict[str, float | int | str]] = [
            {
                "scenario": scenario_label,
                "sensitivity_gain": float(gain),
                "n_total": int(n_current),
                "n_new": 0,
                "mock_id": -1,
                "sigma_true": np.nan,
                "ess": float(len(mu_draws)),
                **current_stats,
            }
        ]

        for n_total in n_grid:
            if n_total == n_current:
                continue
            n_new = n_total - n_current
            for mock_id in range(args.n_mocks):
                rng_mock = np.random.default_rng(_seed_for(args.seed, n_total, mock_id))
                sigma_true = _choose_sigma_true(
                    rng=rng_mock,
                    mode=args.truth_mode,
                    mu_true=args.mu_true,
                    fixed_sigma=args.fixed_sigma,
                    posterior_mu=mu_draws_full,
                    posterior_sigma=sigma_draws_full,
                    condition_bandwidth=args.condition_bandwidth,
                )
                counts, sum_m, sum_m2 = _simulate_mock_catalog_summary(
                    rng=rng_mock,
                    base_sigmas=base_sigmas,
                    n_new=n_new,
                    mu_true=args.mu_true,
                    sigma_true=sigma_true,
                )
                w_post, ess = _update_current_posterior_with_mock(
                    mu_draws=mu_draws,
                    sigma_draws=sigma_draws,
                    base_sigmas=base_sigmas,
                    counts=counts,
                    sum_m=sum_m,
                    sum_m2=sum_m2,
                )
                stats = _weighted_stats(mu_draws, w_post)
                scenario_records.append(
                    {
                        "scenario": scenario_label,
                        "sensitivity_gain": float(gain),
                        "n_total": int(n_total),
                        "n_new": int(n_new),
                        "mock_id": int(mock_id),
                        "sigma_true": float(sigma_true),
                        "ess": float(ess),
                        **stats,
                    }
                )

        scenario_df = pd.DataFrame.from_records(scenario_records)
        all_records.extend(scenario_records)

        rows: list[dict[str, float | int | str]] = []
        for n_total, group in scenario_df.groupby("n_total", sort=True):
            row: dict[str, float | int | str] = {
                "scenario": scenario_label,
                "sensitivity_gain": float(gain),
                "n_total": int(n_total),
            }
            for col in ["mu_halfwidth68", "mu_std", "mu_mean", "mu_q16", "mu_q50", "mu_q84", "sigma_true", "ess"]:
                vals = np.asarray(group[col], dtype=float)
                finite = vals[np.isfinite(vals)]
                if len(finite) == 0:
                    row[f"{col}_p16"] = np.nan
                    row[f"{col}_p50"] = np.nan
                    row[f"{col}_p84"] = np.nan
                else:
                    row[f"{col}_p16"] = float(np.quantile(finite, 0.16))
                    row[f"{col}_p50"] = float(np.quantile(finite, 0.50))
                    row[f"{col}_p84"] = float(np.quantile(finite, 0.84))
            metric_vals = np.asarray(group[metric_col], dtype=float)
            metric_vals = metric_vals[np.isfinite(metric_vals)]
            row["meets_target_fraction"] = float(np.mean(metric_vals <= args.target_halfwidth)) if len(metric_vals) else np.nan
            rows.append(row)

        scenario_summary_df = pd.DataFrame(rows).sort_values("n_total")
        all_summary_rows.extend(rows)
        crossing = scenario_summary_df.loc[scenario_summary_df[f"{metric_col}_p50"] <= args.target_halfwidth]
        n_cross = int(crossing["n_total"].iloc[0]) if len(crossing) else None
        crossing_rows.append({"scenario": scenario_label, "sensitivity_gain": float(gain), "median_crossing_total_n": n_cross})

    mock_df = pd.DataFrame.from_records(all_records)
    mock_df.to_csv(outdir / "forecast_mock_catalogs.csv", index=False)
    summary_df = pd.DataFrame(all_summary_rows).sort_values(["scenario", "n_total"])
    summary_df.to_csv(outdir / "forecast_summary.csv", index=False)
    pd.DataFrame(crossing_rows).to_csv(outdir / "forecast_crossings.csv", index=False)

    assumptions = pd.DataFrame(
        {
            "key": [
                "current_catalog_size",
                "current_catalog_mu_halfwidth68",
                "current_catalog_mu_std",
                "target_halfwidth",
                "metric",
                "truth_mode",
                "mu_true",
                "fixed_sigma",
                "condition_bandwidth",
                "quick_guess_total_n",
                "n_mocks",
                "posterior_draws_used",
                "local_width_calibration_factor",
                "raw_local_width_implied_halfwidth",
                "median_moment_over_local_width",
                "waveform",
            ],
            "value": [
                int(n_current),
                float(current_stats["mu_halfwidth68"]),
                float(current_stats["mu_std"]),
                float(args.target_halfwidth),
                args.metric,
                args.truth_mode,
                float(args.mu_true),
                float(args.fixed_sigma),
                args.condition_bandwidth,
                float(quick_guess),
                int(args.n_mocks),
                int(len(mu_draws)),
                float(width_scale),
                float(raw_gaussian_hw),
                float(median_ratio),
                args.waveform,
            ],
        }
    )
    assumptions.to_csv(outdir / "forecast_assumptions.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for scenario_label, group in summary_df.groupby("scenario", sort=False):
        x = group["n_total"].to_numpy(dtype=float)
        y50 = group[f"{metric_col}_p50"].to_numpy(dtype=float)
        y16 = group[f"{metric_col}_p16"].to_numpy(dtype=float)
        y84 = group[f"{metric_col}_p84"].to_numpy(dtype=float)
        ax.fill_between(x, y16, y84, alpha=0.12)
        gain = float(group["sensitivity_gain"].iloc[0])
        ax.plot(x, y50, marker="o", label=f"{scenario_label} (gain {gain:.2f})")

    ax.axhline(args.target_halfwidth, linestyle="--", linewidth=1.5, label=f"target = {args.target_halfwidth:g}")
    ax.axvline(n_current, linestyle=":", linewidth=1.5, label=f"current N = {n_current}")
    if np.isfinite(quick_guess) and quick_guess >= n_current and quick_guess <= max(summary_df["n_total"]) * 1.05:
        ax.axvline(quick_guess, linestyle="-.", linewidth=1.2, label=f"1/sqrt(N) guess ≈ {quick_guess:.0f}")

    ax.set_xlabel("Total number of memory-analyzed events")
    ax.set_ylabel(r"68% half-width on $\mu_A$" if args.metric == "halfwidth68" else r"Posterior std of $\mu_A$")
    ax.set_title(args.plot_title or "Sensitivity-scenario forecast for the mean memory-enhancement factor")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "forecast_mu_tgr_sensitivity_curves.png", dpi=200)
    plt.close(fig)

    print(f"Current catalog size: {n_current}")
    print(f"Current mu half-width68 from posterior_nc: {current_stats['mu_halfwidth68']:.3f}")
    print(f"Current mu std from posterior_nc: {current_stats['mu_std']:.3f}")
    print(f"Raw local-width implied half-width: {raw_gaussian_hw:.3f}")
    print(f"Local-width calibration factor: {width_scale:.3f}")
    print(f"Median(moment-width / local-width): {median_ratio:.3f}")
    print(f"1/sqrt(N) quick guess for total N at target: {quick_guess:.1f}")
    for row in crossing_rows:
        print(f"{row['scenario']}: median crossing total N = {row['median_crossing_total_n']}")
    print(f"Wrote forecast products to {outdir}")


if __name__ == "__main__":
    main()
