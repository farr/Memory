#!/usr/bin/env python3
"""Rank which events drive the memory-only (mu_tgr, sigma_tgr) constraint.

This script is designed for the `farr/Memory` repository. It reconstructs the
memory-only likelihood inputs used by `run_hierarchical_analysis.py`, then uses
fast leave-one-event-out importance reweighting of `result_memory.nc` to ask
which events matter most for:

- tightening the `mu_tgr` constraint,
- tightening the `sigma_tgr` constraint,
- shifting the posterior medians, and
- shrinking the joint `(mu_tgr, sigma_tgr)` contour area.

Outputs
-------
- event_influence.csv
    One row per event with leave-one-out influence metrics and PE-derived
    properties.
- property_correlations.csv
    Spearman correlations between influence metrics and event properties.
- top_events_report.txt
    Human-readable summary of the most important events and their properties.
- top_mu_precision_events.csv
    Top events ranked by how much they tighten `mu_tgr`.
- top_sigma_precision_events.csv
    Top events ranked by how much they tighten `sigma_tgr`.
- top_area_events.csv
    Top events ranked by how much they shrink the joint `(mu_tgr, sigma_tgr)`
    contour area.
- top_mu_precision_drivers.png
- top_sigma_precision_drivers.png
- top_area_drivers.png

Typical usage
-------------
python rank_memory_events.py \
    --results-dir results \
    --outdir results/ranking
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn
from scipy.special import logsumexp

from memory.hierarchical import generate_tgr_only_data, load_memory_data
from memory.hierarchical.data import (
    _pick_waveform_label,
    _resolve_waveform_label,
    compute_log_prior_from_config,
    load_event_ifars,
    validate_posterior_prior_consistency,
)

LOG = logging.getLogger("rank_memory_events")
EVENT_RE = re.compile(r"(GW\d{6}_\d{6})")
DEFAULT_SNR_FIELDS = (
    "network_matched_filter_snr",
    "network_optimal_snr",
    "optimal_snr_net",
    "matched_filter_snr",
    "snr",
)


@dataclass
class RunConfig:
    memory_dir: Optional[str]
    waveform: Optional[str]
    n_samples_per_event: int
    seed: int
    ifar_threshold: float
    scale_tgr: bool
    ignore_memory_weights: bool


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _event_name_from_path(path: str) -> str:
    m = EVENT_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Could not extract event name from: {path}")
    return m.group(1)


def _read_lines(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def _read_first_line(path: Path) -> str:
    lines = _read_lines(path)
    if not lines:
        raise ValueError(f"File is empty: {path}")
    return lines[0]


def _fmt(x: float, ndigits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{ndigits}g}"


def _resolve_path_from_command(raw: str, *, results_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [
        p,
        Path.cwd() / p,
        results_dir / p,
        results_dir.parent / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return p


def _resolve_memory_dir(
    memory_dir_override: Optional[Path],
    command_memory_dir: Optional[str],
    *,
    results_dir: Path,
) -> Path:
    if memory_dir_override is not None:
        path = memory_dir_override
    elif command_memory_dir:
        path = _resolve_path_from_command(command_memory_dir, results_dir=results_dir)
    else:
        raise ValueError(
            "Could not determine the memory-results directory. Pass --memory-dir, "
            "or ensure command.txt contains --memory-dir."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Memory directory not found: {path}. Pass --memory-dir explicitly if command.txt "
            "contains a stale or relative path that no longer resolves."
        )
    return path


def _parse_original_command(command_path: Path) -> RunConfig:
    text = _read_first_line(command_path)
    argv = shlex.split(text)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory-dir", type=str, default=None)
    parser.add_argument("--waveform", type=str, default="auto")
    parser.add_argument("--n-samples-per-event", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=150914)
    parser.add_argument("--ifar-threshold", type=float, default=1.0)
    parser.add_argument("--scale-tgr", action="store_true")
    parser.add_argument("--ignore-memory-weights", action="store_true")
    ns, _ = parser.parse_known_args(argv[1:])
    waveform = None if ns.waveform is None or ns.waveform.lower() == "auto" else ns.waveform
    return RunConfig(
        memory_dir=ns.memory_dir,
        waveform=waveform,
        n_samples_per_event=ns.n_samples_per_event,
        seed=ns.seed,
        ifar_threshold=ns.ifar_threshold,
        scale_tgr=bool(ns.scale_tgr),
        ignore_memory_weights=bool(ns.ignore_memory_weights),
    )


def _effective_sample_size_from_logw(logw: np.ndarray) -> float:
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probs = np.asarray(probs, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.full_like(probs, np.nan, dtype=float)
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = (cdf - 0.5 * weights) / np.sum(weights)
    return np.interp(probs, cdf, values)


def _weighted_cov_2d(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.full((2, 2), np.nan)
    mx = np.sum(w * x) / s
    my = np.sum(w * y) / s
    dx = x - mx
    dy = y - my
    cov_xx = np.sum(w * dx * dx) / s
    cov_yy = np.sum(w * dy * dy) / s
    cov_xy = np.sum(w * dx * dy) / s
    return np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)


def _summary_stats(mu: np.ndarray, sigma: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if weights is None:
        weights = np.ones_like(mu)
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if np.sum(weights) == 0:
        weights = np.ones_like(mu)
    weights = weights / np.sum(weights)
    mu_q16, mu_q50, mu_q84 = _weighted_quantile(mu, weights, [0.16, 0.50, 0.84])
    sg_q16, sg_q50, sg_q84 = _weighted_quantile(sigma, weights, [0.16, 0.50, 0.84])
    mu_mean = float(np.sum(weights * mu))
    sg_mean = float(np.sum(weights * sigma))
    mu_std = float(np.sqrt(np.sum(weights * (mu - mu_mean) ** 2)))
    sg_std = float(np.sqrt(np.sum(weights * (sigma - sg_mean) ** 2)))
    cov = _weighted_cov_2d(mu, sigma, weights)
    det_cov = float(np.linalg.det(cov)) if np.all(np.isfinite(cov)) else np.nan
    logdet_cov = float(np.log(det_cov)) if det_cov > 0 else np.nan
    logarea = 0.5 * logdet_cov if np.isfinite(logdet_cov) else np.nan
    return {
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "mu_q16": float(mu_q16),
        "mu_q50": float(mu_q50),
        "mu_q84": float(mu_q84),
        "mu_halfwidth68": 0.5 * float(mu_q84 - mu_q16),
        "sigma_mean": sg_mean,
        "sigma_std": sg_std,
        "sigma_q16": float(sg_q16),
        "sigma_q50": float(sg_q50),
        "sigma_q84": float(sg_q84),
        "sigma_halfwidth68": 0.5 * float(sg_q84 - sg_q16),
        "logdet_cov": logdet_cov,
        "logarea_approx": logarea,
    }


def _percentile_of_value(sample: np.ndarray, value: float) -> float:
    sample = np.asarray(sample, dtype=float)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0 or not np.isfinite(value):
        return np.nan
    return 100.0 * float(np.mean(finite <= value))


# ---------------------------------------------------------------------------
# Reconstruct the exact memory-model inputs used in the original run
# ---------------------------------------------------------------------------

def _load_event_posteriors(
    event_files: Sequence[str],
    waveform: Optional[str],
    per_event_labels: Optional[Mapping[str, str]] = None,
) -> Tuple[List[np.ndarray], List[str], Dict[str, str]]:
    per_event_labels = dict(per_event_labels or {})
    posteriors: List[np.ndarray] = []
    kept_files: List[str] = []
    used_labels: Dict[str, str] = {}

    for filename in event_files:
        event_name = _event_name_from_path(filename)
        with h5py.File(filename, "r") as f:
            all_ps_keys = [
                k
                for k in f.keys()
                if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]
            ]
            if not all_ps_keys:
                LOG.warning("Skipping %s: no posterior_samples groups", os.path.basename(filename))
                continue

            chosen = None
            if event_name in per_event_labels and per_event_labels[event_name] in all_ps_keys:
                chosen = per_event_labels[event_name]
            if chosen is None:
                try:
                    chosen = _resolve_waveform_label(all_ps_keys, waveform)
                except KeyError:
                    LOG.warning("Skipping %s: no matching waveform=%r", os.path.basename(filename), waveform)
                    continue

            ps_fields = f[chosen]["posterior_samples"].dtype.names
            computed_log_prior = None
            tmp_ps = None
            if "log_prior" not in ps_fields and "prior" not in ps_fields:
                tmp_ps = f[chosen]["posterior_samples"][()]
                computed_log_prior, _ = compute_log_prior_from_config(f[chosen], tmp_ps)
                if computed_log_prior is None:
                    fallback_keys = [
                        k
                        for k in all_ps_keys
                        if k != chosen and (
                            "log_prior" in f[k]["posterior_samples"].dtype.names
                            or "prior" in f[k]["posterior_samples"].dtype.names
                        )
                    ]
                    if not fallback_keys:
                        LOG.warning("Skipping %s: no group with log_prior/prior", os.path.basename(filename))
                        continue
                    try:
                        chosen = _resolve_waveform_label(fallback_keys, waveform)
                    except KeyError:
                        chosen = _pick_waveform_label(fallback_keys)
                    tmp_ps = None
                    computed_log_prior = None

            if computed_log_prior is not None:
                posterior_samples = rfn.append_fields(
                    tmp_ps,
                    "log_prior",
                    computed_log_prior,
                    dtypes=float,
                    usemask=False,
                )
            else:
                posterior_samples = f[chosen]["posterior_samples"][()]

            validate_posterior_prior_consistency(
                f[chosen], posterior_samples, filename=filename, label=chosen
            )
            posteriors.append(posterior_samples)
            kept_files.append(filename)
            used_labels[event_name] = chosen

    return posteriors, kept_files, used_labels


def _reconstruct_final_event_names(
    ordered_names: Sequence[str],
    memory_data: Sequence[dict],
    *,
    n_samples_per_event: int,
    seed: int,
    ifar_threshold: float,
    ifar_cache_path: Optional[str],
    analyzed_events_path: Path,
    nobs_expected: int,
) -> List[str]:
    if analyzed_events_path.exists():
        final_names = _read_lines(analyzed_events_path)
        if len(final_names) == nobs_expected:
            return final_names
        LOG.warning(
            "Ignoring %s because it contains %d names but Nobs=%d.",
            analyzed_events_path,
            len(final_names),
            nobs_expected,
        )

    ifars = load_event_ifars(list(ordered_names), ifar_cache_path)
    pre_names = [n for n in ordered_names if ifars.get(n, float("inf")) >= ifar_threshold]
    name_to_md = {md["event_name"]: md for md in memory_data}
    rng = np.random.default_rng(seed)
    final_names: List[str] = []
    for name in pre_names:
        md = name_to_md[name]
        idxs = rng.choice(len(md["A_hat"]), size=n_samples_per_event, replace=True)
        if np.all(np.isnan(md["A_hat"][idxs])):
            continue
        final_names.append(name)

    if len(final_names) != nobs_expected:
        raise RuntimeError(
            "Could not reconstruct the final event ordering used by generate_tgr_only_data: "
            f"reconstructed {len(final_names)} names, but Nobs={nobs_expected}."
        )
    return final_names


def _load_run_inputs(
    results_dir: Path,
    posterior_nc: Optional[Path],
    event_files_path: Optional[Path],
    memory_dir: Optional[Path],
    seed_override: Optional[int],
    n_samples_override: Optional[int],
    ifar_override: Optional[float],
    waveform_override: Optional[str],
    scale_tgr_override: Optional[bool],
    ignore_memory_weights_override: Optional[bool],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, float, RunConfig]:
    posterior_nc = posterior_nc or results_dir / "result_memory.nc"
    event_files_path = event_files_path or results_dir / "event_files.txt"
    command_path = results_dir / "command.txt"
    ifar_cache_path = results_dir / "event_ifars.txt"
    analyzed_events_path = results_dir / "analyzed_events.txt"

    if not posterior_nc.exists():
        raise FileNotFoundError(f"Posterior file not found: {posterior_nc}")
    if not event_files_path.exists():
        raise FileNotFoundError(f"Event-file list not found: {event_files_path}")
    if not command_path.exists():
        raise FileNotFoundError(f"Command file not found: {command_path}")

    cfg = _parse_original_command(command_path)
    memory_dir = _resolve_memory_dir(memory_dir, cfg.memory_dir, results_dir=results_dir)

    if seed_override is not None:
        cfg.seed = seed_override
    if n_samples_override is not None:
        cfg.n_samples_per_event = n_samples_override
    if ifar_override is not None:
        cfg.ifar_threshold = ifar_override
    if waveform_override is not None:
        cfg.waveform = None if waveform_override.lower() == "auto" else waveform_override
    if scale_tgr_override is not None:
        cfg.scale_tgr = scale_tgr_override
    if ignore_memory_weights_override is not None:
        cfg.ignore_memory_weights = ignore_memory_weights_override

    if cfg.seed == 0:
        raise ValueError(
            "The original run used --seed 0, so the resampling is not reproducible from command.txt alone. "
            "Please rerun with --seed-override set to the actual seed used in that run."
        )

    event_files = _read_lines(event_files_path)
    if not event_files:
        raise ValueError(f"No event files listed in {event_files_path}")

    memory_data = load_memory_data(event_files, str(memory_dir), cfg.waveform)
    mem_names_loaded = {md["event_name"] for md in memory_data}
    mem_files = [ef for ef in event_files if _event_name_from_path(ef) in mem_names_loaded]

    per_event_labels = {md["event_name"]: md["waveform_label"] for md in memory_data}
    all_posteriors, all_event_files, used_labels = _load_event_posteriors(mem_files, cfg.waveform, per_event_labels)

    name_to_memory = {md["event_name"]: md for md in memory_data}
    name_to_post = {_event_name_from_path(f): p for f, p in zip(all_event_files, all_posteriors)}

    ordered_names = [
        _event_name_from_path(f)
        for f in mem_files
        if _event_name_from_path(f) in name_to_post
    ]
    memory_data = [name_to_memory[n] for n in ordered_names]
    mem_posteriors = [name_to_post[n] for n in ordered_names]

    fixed_memory: List[dict] = []
    fixed_posteriors: List[np.ndarray] = []
    fixed_names: List[str] = []
    for name, md, post in zip(ordered_names, memory_data, mem_posteriors):
        used = used_labels.get(name)
        if used is not None and used != md["waveform_label"]:
            mem_path = memory_dir / name / "memory_results.h5"
            with h5py.File(mem_path, "r") as mf:
                if used not in mf:
                    LOG.warning("Skipping %s: fallback memory waveform '%s' not found", name, used)
                    continue
                grp = mf[used]
                md = {
                    "A_sample": np.asarray(grp["A_sample"][()].real),
                    "A_hat": np.asarray(grp["A_hat"][()].real),
                    "A_sigma": np.asarray(grp["A_sigma"][()].real),
                    "log_weight": np.asarray(grp["log_weight"][()].real),
                    "event_name": name,
                    "waveform_label": used,
                }
        fixed_memory.append(md)
        fixed_posteriors.append(post)
        fixed_names.append(name)

    memory_data = fixed_memory
    mem_posteriors = fixed_posteriors
    ordered_names = fixed_names

    A_hats, A_sigmas, log_weights, nobs, A_scale = generate_tgr_only_data(
        mem_posteriors,
        memory_data,
        N_samples=cfg.n_samples_per_event,
        prng=cfg.seed,
        scale_tgr=cfg.scale_tgr,
        ignore_memory_weights=cfg.ignore_memory_weights,
        ifar_threshold=cfg.ifar_threshold,
        ifar_cache_file=str(ifar_cache_path) if ifar_cache_path.exists() else None,
    )

    final_names = _reconstruct_final_event_names(
        ordered_names,
        memory_data,
        n_samples_per_event=cfg.n_samples_per_event,
        seed=cfg.seed,
        ifar_threshold=cfg.ifar_threshold,
        ifar_cache_path=str(ifar_cache_path) if ifar_cache_path.exists() else None,
        analyzed_events_path=analyzed_events_path,
        nobs_expected=nobs,
    )
    name_to_post_final = {n: p for n, p in zip(ordered_names, mem_posteriors)}
    final_posteriors = [name_to_post_final[n] for n in final_names]

    fit = az.from_netcdf(posterior_nc)
    mu = np.asarray(fit.posterior["mu_tgr"].values, dtype=float).reshape(-1)
    sigma = np.asarray(fit.posterior["sigma_tgr"].values, dtype=float).reshape(-1)
    good = np.isfinite(mu) & np.isfinite(sigma)
    mu = mu[good]
    sigma = sigma[good]

    return mu, sigma, final_names, final_posteriors, A_hats, A_sigmas, log_weights, A_scale, cfg


# ---------------------------------------------------------------------------
# Event properties
# ---------------------------------------------------------------------------

def _field(ps: np.ndarray, *names: str) -> Optional[np.ndarray]:
    dtype_names = set(ps.dtype.names or [])
    for name in names:
        if name in dtype_names:
            return np.asarray(ps[name], dtype=float)
    return None


def _q05_q50_q95(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return tuple(float(v) for v in np.quantile(x, [0.05, 0.50, 0.95]))


def _event_properties(name: str, ps: np.ndarray, A_hat: np.ndarray, A_sigma: np.ndarray) -> Dict[str, float]:
    m1 = _field(ps, "mass_1_source")
    q = _field(ps, "mass_ratio")
    if q is None:
        m2_src = _field(ps, "mass_2_source")
        if m1 is not None and m2_src is not None:
            q = np.clip(m2_src / np.maximum(m1, 1e-12), 0.0, 1.0)

    redshift = _field(ps, "redshift")
    dl = _field(ps, "luminosity_distance")
    a1 = _field(ps, "a_1", "spin1_a")
    a2 = _field(ps, "a_2", "spin2_a")
    ct1 = _field(ps, "cos_tilt_1")
    ct2 = _field(ps, "cos_tilt_2")
    snr = _field(ps, *DEFAULT_SNR_FIELDS)

    out: Dict[str, float] = {
        "event_name": name,
        "n_posterior_samples": float(len(ps)),
        "memory_sigma_median": float(np.nanmedian(A_sigma)),
        "memory_absA_over_sigma_median": float(np.nanmedian(np.abs(A_hat / A_sigma))),
    }

    if snr is not None:
        q05, q50, q95 = _q05_q50_q95(snr)
        out.update({"snr_q05": q05, "snr_median": q50, "snr_q95": q95})
    else:
        out.update({"snr_q05": np.nan, "snr_median": np.nan, "snr_q95": np.nan})

    if m1 is not None and q is not None:
        m2 = m1 * q
        mtot = m1 + m2
        mchirp = (m1 * m2) ** (3.0 / 5.0) / np.maximum(mtot, 1e-12) ** (1.0 / 5.0)
        q05, q50, q95 = _q05_q50_q95(mchirp)
        out.update({"mchirp_source_q05": q05, "mchirp_source_median": q50, "mchirp_source_q95": q95})
        q05, q50, q95 = _q05_q50_q95(mtot)
        out.update({"mtot_source_q05": q05, "mtot_source_median": q50, "mtot_source_q95": q95})
        q05, q50, q95 = _q05_q50_q95(q)
        out.update({"mass_ratio_q05": q05, "mass_ratio_median": q50, "mass_ratio_q95": q95})
    else:
        for prefix in ["mchirp_source", "mtot_source", "mass_ratio"]:
            out.update({f"{prefix}_q05": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q95": np.nan})

    if dl is not None:
        q05, q50, q95 = _q05_q50_q95(dl)
        out.update({"luminosity_distance_q05": q05, "luminosity_distance_median": q50, "luminosity_distance_q95": q95})
    else:
        out.update({"luminosity_distance_q05": np.nan, "luminosity_distance_median": np.nan, "luminosity_distance_q95": np.nan})

    if redshift is not None:
        q05, q50, q95 = _q05_q50_q95(redshift)
        out.update({"redshift_q05": q05, "redshift_median": q50, "redshift_q95": q95})
    else:
        out.update({"redshift_q05": np.nan, "redshift_median": np.nan, "redshift_q95": np.nan})

    if a1 is not None:
        q05, q50, q95 = _q05_q50_q95(a1)
        out.update({"a1_q05": q05, "a1_median": q50, "a1_q95": q95})
    else:
        out.update({"a1_q05": np.nan, "a1_median": np.nan, "a1_q95": np.nan})

    if a2 is not None:
        q05, q50, q95 = _q05_q50_q95(a2)
        out.update({"a2_q05": q05, "a2_median": q50, "a2_q95": q95})
    else:
        out.update({"a2_q05": np.nan, "a2_median": np.nan, "a2_q95": np.nan})

    if ct1 is not None:
        q05, q50, q95 = _q05_q50_q95(ct1)
        out.update({"cos_tilt_1_q05": q05, "cos_tilt_1_median": q50, "cos_tilt_1_q95": q95})
    else:
        out.update({"cos_tilt_1_q05": np.nan, "cos_tilt_1_median": np.nan, "cos_tilt_1_q95": np.nan})

    if ct2 is not None:
        q05, q50, q95 = _q05_q50_q95(ct2)
        out.update({"cos_tilt_2_q05": q05, "cos_tilt_2_median": q50, "cos_tilt_2_q95": q95})
    else:
        out.update({"cos_tilt_2_q05": np.nan, "cos_tilt_2_median": np.nan, "cos_tilt_2_q95": np.nan})

    if a1 is not None and a2 is not None and q is not None and ct1 is not None and ct2 is not None:
        chi_eff = (a1 * ct1 + q * a2 * ct2) / np.maximum(1.0 + q, 1e-12)
        spin_mean = 0.5 * (a1 + a2)
        q05, q50, q95 = _q05_q50_q95(chi_eff)
        out.update({"chi_eff_q05": q05, "chi_eff_median": q50, "chi_eff_q95": q95})
        q05, q50, q95 = _q05_q50_q95(np.abs(chi_eff))
        out.update({"abs_chi_eff_q05": q05, "abs_chi_eff_median": q50, "abs_chi_eff_q95": q95})
        q05, q50, q95 = _q05_q50_q95(spin_mean)
        out.update({"spin_mean_q05": q05, "spin_mean_median": q50, "spin_mean_q95": q95})
    else:
        for prefix in ["chi_eff", "abs_chi_eff", "spin_mean"]:
            out.update({f"{prefix}_q05": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q95": np.nan})

    return out


# ---------------------------------------------------------------------------
# Influence computation
# ---------------------------------------------------------------------------

def _event_loglike_matrix(
    mu_model: np.ndarray,
    sigma_model: np.ndarray,
    A_hats: np.ndarray,
    A_sigmas: np.ndarray,
    log_weights: np.ndarray,
    draw_chunk: int = 2000,
) -> np.ndarray:
    n_events, _ = A_hats.shape
    n_draws = len(mu_model)
    loglikes = np.empty((n_events, n_draws), dtype=float)
    const = -0.5 * np.log(2.0 * np.pi)

    for j in range(n_events):
        ah = A_hats[j][None, :]
        asig = A_sigmas[j][None, :]
        lw = log_weights[j][None, :]
        for start in range(0, n_draws, draw_chunk):
            stop = min(start + draw_chunk, n_draws)
            sl = slice(start, stop)
            sigma_eff = np.sqrt(asig * asig + sigma_model[sl, None] ** 2)
            z = (ah - mu_model[sl, None]) / sigma_eff
            lp = const - np.log(sigma_eff) - 0.5 * z * z + lw
            loglikes[j, sl] = logsumexp(lp, axis=1)
    return loglikes


def _rank_events(
    names: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    loglikes: np.ndarray,
    properties_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    full = _summary_stats(mu, sigma)
    rows: List[Dict[str, float]] = []
    for j, name in enumerate(names):
        logw = -loglikes[j]
        logw -= logsumexp(logw)
        w = np.exp(logw)
        loo = _summary_stats(mu, sigma, weights=w)
        delta_logdet = loo["logdet_cov"] - full["logdet_cov"]
        row = {
            "event_name": name,
            "loo_weight_ess": _effective_sample_size_from_logw(logw),
            "full_mu_q50": full["mu_q50"],
            "full_mu_halfwidth68": full["mu_halfwidth68"],
            "full_sigma_q50": full["sigma_q50"],
            "full_sigma_halfwidth68": full["sigma_halfwidth68"],
            "full_logdet_cov": full["logdet_cov"],
            "full_logarea_approx": full["logarea_approx"],
            "loo_mu_q50": loo["mu_q50"],
            "loo_mu_halfwidth68": loo["mu_halfwidth68"],
            "loo_sigma_q50": loo["sigma_q50"],
            "loo_sigma_halfwidth68": loo["sigma_halfwidth68"],
            "loo_logdet_cov": loo["logdet_cov"],
            "loo_logarea_approx": loo["logarea_approx"],
            "delta_mu_q50": loo["mu_q50"] - full["mu_q50"],
            "abs_delta_mu_q50": abs(loo["mu_q50"] - full["mu_q50"]),
            "delta_mu_halfwidth68": loo["mu_halfwidth68"] - full["mu_halfwidth68"],
            "delta_sigma_q50": loo["sigma_q50"] - full["sigma_q50"],
            "abs_delta_sigma_q50": abs(loo["sigma_q50"] - full["sigma_q50"]),
            "delta_sigma_halfwidth68": loo["sigma_halfwidth68"] - full["sigma_halfwidth68"],
            "delta_logdet_cov": delta_logdet,
            "delta_logarea_approx": 0.5 * delta_logdet if np.isfinite(delta_logdet) else np.nan,
            "area_ratio_approx": float(np.exp(0.5 * delta_logdet)) if np.isfinite(delta_logdet) else np.nan,
            "event_loglike_mean": float(np.mean(loglikes[j])),
        }
        rows.append(row)

    influence = pd.DataFrame(rows)
    influence = influence.merge(properties_df, on="event_name", how="left")
    influence["rank_mu_precision"] = influence["delta_mu_halfwidth68"].rank(ascending=False, method="dense").astype(int)
    influence["rank_sigma_precision"] = influence["delta_sigma_halfwidth68"].rank(ascending=False, method="dense").astype(int)
    influence["rank_mu_shift"] = influence["abs_delta_mu_q50"].rank(ascending=False, method="dense").astype(int)
    influence["rank_sigma_shift"] = influence["abs_delta_sigma_q50"].rank(ascending=False, method="dense").astype(int)
    influence["rank_area"] = influence["delta_logdet_cov"].rank(ascending=False, method="dense").astype(int)
    influence = influence.sort_values(["rank_sigma_precision", "rank_mu_precision", "rank_area", "event_name"]).reset_index(drop=True)
    return influence, full


# ---------------------------------------------------------------------------
# Summary products
# ---------------------------------------------------------------------------

def _property_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_mu_halfwidth68",
        "delta_sigma_halfwidth68",
        "abs_delta_mu_q50",
        "abs_delta_sigma_q50",
        "delta_logdet_cov",
    ]
    props = [
        "snr_median",
        "memory_absA_over_sigma_median",
        "memory_sigma_median",
        "mchirp_source_median",
        "mtot_source_median",
        "mass_ratio_median",
        "chi_eff_median",
        "abs_chi_eff_median",
        "spin_mean_median",
        "luminosity_distance_median",
        "redshift_median",
    ]
    rows = []
    for metric in metrics:
        for prop in props:
            sub = df[[metric, prop]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = np.nan if len(sub) < 3 else sub[metric].corr(sub[prop], method="spearman")
            rows.append({"metric": metric, "property": prop, "spearman_rho": rho, "n": int(len(sub))})
    out = pd.DataFrame(rows)
    out["abs_rho"] = np.abs(out["spearman_rho"])
    return out.sort_values(["metric", "abs_rho"], ascending=[True, False]).reset_index(drop=True)


def _group_profile_lines(df: pd.DataFrame, top_names: Sequence[str]) -> List[str]:
    top = df[df["event_name"].isin(top_names)].copy()
    rest = df[~df["event_name"].isin(top_names)].copy()
    props = [
        ("snr_median", "network SNR"),
        ("memory_absA_over_sigma_median", "median |A_hat/A_sigma|"),
        ("memory_sigma_median", "median A_sigma"),
        ("mchirp_source_median", "source chirp mass"),
        ("mtot_source_median", "source total mass"),
        ("mass_ratio_median", "mass ratio q"),
        ("chi_eff_median", "chi_eff"),
        ("spin_mean_median", "mean spin magnitude"),
        ("luminosity_distance_median", "luminosity distance"),
    ]
    lines = []
    for col, label in props:
        all_vals = df[col].to_numpy(dtype=float)
        top_med = float(np.nanmedian(top[col])) if len(top) else np.nan
        rest_med = float(np.nanmedian(rest[col])) if len(rest) else np.nan
        pct = _percentile_of_value(all_vals, top_med)
        lines.append(
            f"  - {label}: top median = {_fmt(top_med)}, rest median = {_fmt(rest_med)}, "
            f"top-median percentile among all events = {_fmt(pct, 1)}%"
        )
    return lines


def _top_subset(df: pd.DataFrame, metric: str, top_k: int) -> pd.DataFrame:
    return df.dropna(subset=[metric]).nlargest(min(top_k, len(df)), metric).copy()


def _write_top_csvs(df: pd.DataFrame, outdir: Path, top_k: int) -> None:
    _top_subset(df, "delta_mu_halfwidth68", top_k).to_csv(outdir / "top_mu_precision_events.csv", index=False)
    _top_subset(df, "delta_sigma_halfwidth68", top_k).to_csv(outdir / "top_sigma_precision_events.csv", index=False)
    _top_subset(df, "delta_logdet_cov", top_k).to_csv(outdir / "top_area_events.csv", index=False)


def _write_text_report(df: pd.DataFrame, corr_df: pd.DataFrame, outpath: Path, top_k: int) -> None:
    top_mu_precision = _top_subset(df, "delta_mu_halfwidth68", top_k)
    top_sigma_precision = _top_subset(df, "delta_sigma_halfwidth68", top_k)
    top_mu_shift = _top_subset(df, "abs_delta_mu_q50", top_k)
    top_sigma_shift = _top_subset(df, "abs_delta_sigma_q50", top_k)
    top_area = _top_subset(df, "delta_logdet_cov", top_k)

    with open(outpath, "w") as f:
        f.write("Leave-one-out influence summary for the memory-only (mu_tgr, sigma_tgr) posterior\n")
        f.write("=" * 84 + "\n\n")

        f.write(f"Top {len(top_mu_precision)} mu_A precision drivers (largest increase in mu_A 68% half-width when removed):\n")
        for _, row in top_mu_precision.iterrows():
            f.write(
                f"  {row['event_name']}: Δmu_hw={_fmt(row['delta_mu_halfwidth68'], 4)}, "
                f"Δsigma_hw={_fmt(row['delta_sigma_halfwidth68'], 4)}, "
                f"|Δmu_med|={_fmt(row['abs_delta_mu_q50'], 4)}, "
                f"Δlogdet={_fmt(row['delta_logdet_cov'], 4)}, "
                f"SNR={_fmt(row['snr_median'])}, q={_fmt(row['mass_ratio_median'])}, "
                f"Mchirp={_fmt(row['mchirp_source_median'])}, chi_eff={_fmt(row['chi_eff_median'])}, "
                f"A_sigma={_fmt(row['memory_sigma_median'])}, DL={_fmt(row['luminosity_distance_median'])}\n"
            )
        f.write("\n")

        f.write(f"Top {len(top_sigma_precision)} sigma_A precision drivers (largest increase in sigma_A 68% half-width when removed):\n")
        for _, row in top_sigma_precision.iterrows():
            f.write(
                f"  {row['event_name']}: Δsigma_hw={_fmt(row['delta_sigma_halfwidth68'], 4)}, "
                f"Δmu_hw={_fmt(row['delta_mu_halfwidth68'], 4)}, "
                f"|Δsigma_med|={_fmt(row['abs_delta_sigma_q50'], 4)}, "
                f"Δlogdet={_fmt(row['delta_logdet_cov'], 4)}, "
                f"SNR={_fmt(row['snr_median'])}, q={_fmt(row['mass_ratio_median'])}, "
                f"Mchirp={_fmt(row['mchirp_source_median'])}, chi_eff={_fmt(row['chi_eff_median'])}, "
                f"A_sigma={_fmt(row['memory_sigma_median'])}, DL={_fmt(row['luminosity_distance_median'])}\n"
            )
        f.write("\n")

        f.write(f"Top {len(top_mu_shift)} mu_A mean-pulling events (largest |shift| in mu_A median when removed):\n")
        for _, row in top_mu_shift.iterrows():
            f.write(
                f"  {row['event_name']}: |Δmu_med|={_fmt(row['abs_delta_mu_q50'], 4)}, "
                f"Δmu_hw={_fmt(row['delta_mu_halfwidth68'], 4)}, "
                f"Δsigma_hw={_fmt(row['delta_sigma_halfwidth68'], 4)}, "
                f"Δlogdet={_fmt(row['delta_logdet_cov'], 4)}, "
                f"SNR={_fmt(row['snr_median'])}, q={_fmt(row['mass_ratio_median'])}, chi_eff={_fmt(row['chi_eff_median'])}\n"
            )
        f.write("\n")

        f.write(f"Top {len(top_sigma_shift)} sigma_A mean-pulling events (largest |shift| in sigma_A median when removed):\n")
        for _, row in top_sigma_shift.iterrows():
            f.write(
                f"  {row['event_name']}: |Δsigma_med|={_fmt(row['abs_delta_sigma_q50'], 4)}, "
                f"Δsigma_hw={_fmt(row['delta_sigma_halfwidth68'], 4)}, "
                f"Δmu_hw={_fmt(row['delta_mu_halfwidth68'], 4)}, "
                f"Δlogdet={_fmt(row['delta_logdet_cov'], 4)}, "
                f"SNR={_fmt(row['snr_median'])}, q={_fmt(row['mass_ratio_median'])}, chi_eff={_fmt(row['chi_eff_median'])}\n"
            )
        f.write("\n")

        f.write(f"Top {len(top_area)} joint-area drivers (largest increase in log det Cov[mu_A, sigma_A] when removed):\n")
        for _, row in top_area.iterrows():
            f.write(
                f"  {row['event_name']}: Δlogdet={_fmt(row['delta_logdet_cov'], 4)}, "
                f"area_ratio≈{_fmt(row['area_ratio_approx'], 4)}, "
                f"Δmu_hw={_fmt(row['delta_mu_halfwidth68'], 4)}, "
                f"Δsigma_hw={_fmt(row['delta_sigma_halfwidth68'], 4)}, "
                f"SNR={_fmt(row['snr_median'])}, q={_fmt(row['mass_ratio_median'])}, "
                f"Mchirp={_fmt(row['mchirp_source_median'])}, A_sigma={_fmt(row['memory_sigma_median'])}\n"
            )
        f.write("\n")

        f.write("Property profile of the sigma_A precision-driving subset versus the rest:\n")
        for line in _group_profile_lines(df, list(top_sigma_precision["event_name"])):
            f.write(line + "\n")
        f.write("\n")

        f.write("Property profile of the joint-area-driving subset versus the rest:\n")
        for line in _group_profile_lines(df, list(top_area["event_name"])):
            f.write(line + "\n")
        f.write("\n")

        f.write("Strongest property correlations with influence metrics (top 5 by |Spearman rho| for each metric):\n")
        for metric in [
            "delta_mu_halfwidth68",
            "delta_sigma_halfwidth68",
            "abs_delta_mu_q50",
            "abs_delta_sigma_q50",
            "delta_logdet_cov",
        ]:
            f.write(f"  {metric}:\n")
            sub = corr_df[corr_df["metric"] == metric].copy().sort_values("abs_rho", ascending=False).head(5)
            for _, row in sub.iterrows():
                f.write(
                    f"    - {row['property']}: rho={_fmt(row['spearman_rho'], 3)} (n={int(row['n'])})\n"
                )
        f.write("\n")


def _make_bar_plot(df: pd.DataFrame, metric: str, outpath: Path, top_k: int, xlabel: str, title: str) -> None:
    plot_df = df.dropna(subset=[metric]).nlargest(min(top_k, len(df)), metric).sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["event_name"], plot_df[metric])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Event")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"), help="Output directory from run_hierarchical_analysis.py (default: results)")
    p.add_argument("--posterior-nc", type=Path, help="Override path to result_memory.nc")
    p.add_argument("--event-files", type=Path, help="Override path to event_files.txt")
    p.add_argument("--memory-dir", type=Path, help="Override path to the per-event memory-results directory")
    p.add_argument("--seed-override", type=int, help="Override the original seed (useful if the original run used --seed 0)")
    p.add_argument("--n-samples-per-event", type=int, help="Override the original N_samples_per_event")
    p.add_argument("--ifar-threshold", type=float, help="Override the original IFAR threshold")
    p.add_argument("--waveform", type=str, help="Override the original waveform selection")
    p.add_argument("--scale-tgr", dest="scale_tgr", action="store_true", help="Override: enable scale_tgr")
    p.add_argument("--no-scale-tgr", dest="scale_tgr", action="store_false", help="Override: disable scale_tgr")
    p.set_defaults(scale_tgr=None)
    p.add_argument(
        "--ignore-memory-weights",
        dest="ignore_memory_weights",
        action="store_true",
        help="Override: ignore log_weight in the reconstructed memory likelihood",
    )
    p.add_argument(
        "--use-memory-weights",
        dest="ignore_memory_weights",
        action="store_false",
        help="Override: use log_weight in the reconstructed memory likelihood",
    )
    p.set_defaults(ignore_memory_weights=None)
    p.add_argument("--max-draws", type=int, default=10000, help="Maximum number of posterior draws to use")
    p.add_argument("--top-k", type=int, default=20, help="Number of top events to show in reports and plots")
    p.add_argument("--outdir", type=Path, help="Output directory (default: <results-dir>/ranking)")
    p.add_argument("--seed-subsample", type=int, default=12345, help="Seed for posterior-draw subsampling")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    outdir = args.outdir or (args.results_dir / "ranking")
    outdir.mkdir(parents=True, exist_ok=True)

    mu, sigma, names, posteriors, A_hats, A_sigmas, log_weights, A_scale, cfg = _load_run_inputs(
        results_dir=args.results_dir,
        posterior_nc=args.posterior_nc,
        event_files_path=args.event_files,
        memory_dir=args.memory_dir,
        seed_override=args.seed_override,
        n_samples_override=args.n_samples_per_event,
        ifar_override=args.ifar_threshold,
        waveform_override=args.waveform,
        scale_tgr_override=args.scale_tgr,
        ignore_memory_weights_override=args.ignore_memory_weights,
    )

    if args.max_draws is not None and len(mu) > args.max_draws:
        rng = np.random.default_rng(args.seed_subsample)
        idx = rng.choice(len(mu), size=args.max_draws, replace=False)
        mu = mu[idx]
        sigma = sigma[idx]

    mu_model = mu / A_scale if cfg.scale_tgr and A_scale != 0 else mu.copy()
    sigma_model = sigma / A_scale if cfg.scale_tgr and A_scale != 0 else sigma.copy()

    properties = [
        _event_properties(name, ps, A_hats[i], A_sigmas[i])
        for i, (name, ps) in enumerate(zip(names, posteriors))
    ]
    properties_df = pd.DataFrame(properties)

    loglikes = _event_loglike_matrix(mu_model, sigma_model, A_hats, A_sigmas, log_weights)
    influence_df, full = _rank_events(names, mu, sigma, loglikes, properties_df)
    corr_df = _property_correlation_table(influence_df)

    influence_df.to_csv(outdir / "event_influence.csv", index=False)
    corr_df.to_csv(outdir / "property_correlations.csv", index=False)
    _write_top_csvs(influence_df, outdir, top_k=args.top_k)
    _write_text_report(influence_df, corr_df, outdir / "top_events_report.txt", top_k=args.top_k)
    _make_bar_plot(
        influence_df,
        "delta_mu_halfwidth68",
        outdir / "top_mu_precision_drivers.png",
        top_k=min(args.top_k, len(influence_df)),
        xlabel=r"Increase in $\mu_A$ 68% half-width when event is removed",
        title=r"Leading $\mu_A$ precision-driving events in the memory-only analysis",
    )
    _make_bar_plot(
        influence_df,
        "delta_sigma_halfwidth68",
        outdir / "top_sigma_precision_drivers.png",
        top_k=min(args.top_k, len(influence_df)),
        xlabel=r"Increase in $\sigma_A$ 68% half-width when event is removed",
        title=r"Leading $\sigma_A$ precision-driving events in the memory-only analysis",
    )
    _make_bar_plot(
        influence_df,
        "delta_logdet_cov",
        outdir / "top_area_drivers.png",
        top_k=min(args.top_k, len(influence_df)),
        xlabel=r"Increase in $\log\det\,\mathrm{Cov}(\mu_A,\sigma_A)$ when event is removed",
        title=r"Leading joint-area-driving events in the memory-only analysis",
    )

    print(f"Reconstructed {len(names)} events from {args.results_dir}")
    print(
        f"Full posterior: mu_tgr median = {full['mu_q50']:.4g}, "
        f"mu 68% half-width = {full['mu_halfwidth68']:.4g}, "
        f"sigma_tgr median = {full['sigma_q50']:.4g}, "
        f"sigma 68% half-width = {full['sigma_halfwidth68']:.4g}"
    )
    top_mu = influence_df.sort_values("rank_mu_precision").iloc[0]["event_name"]
    top_sigma = influence_df.sort_values("rank_sigma_precision").iloc[0]["event_name"]
    top_area = influence_df.sort_values("rank_area").iloc[0]["event_name"]
    print(f"Top mu_A precision driver: {top_mu}")
    print(f"Top sigma_A precision driver: {top_sigma}")
    print(f"Top joint-area driver: {top_area}")
    for name in [
        "event_influence.csv",
        "property_correlations.csv",
        "top_mu_precision_events.csv",
        "top_sigma_precision_events.csv",
        "top_area_events.csv",
        "top_events_report.txt",
        "top_mu_precision_drivers.png",
        "top_sigma_precision_drivers.png",
        "top_area_drivers.png",
    ]:
        print(f"Wrote: {outdir / name}")


if __name__ == "__main__":
    main()
