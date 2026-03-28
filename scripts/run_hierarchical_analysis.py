#!/usr/bin/env python3
"""Run hierarchical Bayesian population analysis with GW memory TGR parameters.

This script performs hierarchical inference on gravitational-wave posterior
samples.  Three analysis modes are available, selected via ``--analyze``:

* **astro** — fits astrophysical population parameters only (mass function,
  mass ratio, redshift, spin magnitudes) without any TGR component.  Uses all
  loaded events; does not require ``--memory-dir``.
* **memory** — fits only the TGR hyperparameters (mu_tgr, sigma_tgr) using a
  simple Gaussian population model directly on per-event (A_hat, A_sigma)
  measurements.  Requires ``--memory-dir``.
* **joint** — simultaneously fits astrophysical population parameters and TGR
  hyperparameters (mu_tgr, sigma_tgr) with selection-effect corrections.
  Uses only events that have memory results.  Requires ``--memory-dir``.

All models treat the per-sample memory measurement as a Gaussian likelihood
N(A | A_hat, A_sigma) convolved analytically with the population prior
N(A | mu_tgr, sigma_tgr).

Outputs (written to ``--outdir``):
    result_astro.nc / result_joint.nc / result_memory.nc
        ArviZ InferenceData with full MCMC posterior (NetCDF).
    fit_astro_samples.dat / fit_joint_samples.dat / fit_memory_samples.dat
        Flat CSV of posterior samples (space-delimited).
    astro_corner.png
        Corner plot of astrophysical parameters (astro and/or joint runs).
    population_distribution.png, hyperparameters.png, tgr_comparison_corner.png
        TGR diagnostic plots (joint and/or memory runs).
    joint_model_corner.png
        Full corner plot including TGR parameters (joint run only).
    All plots are skipped with ``--no-plots``.
    injection_file.txt, event_files.txt, memory_dir.txt, command.txt
        Provenance files recording the exact inputs and command line.

Environment variables:
    TGRPOP_PLATFORM      Force JAX platform ('cpu' or 'gpu'); auto-detected
                         if unset.
    TGRPOP_DEVICE_COUNT  Number of JAX devices for chain parallelisation
                         (default 1).
    OMP_NUM_THREADS      Set to 1 at import time to avoid thread contention.
"""

import logging
import sys
import os
import argparse
from glob import glob
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import h5py
import jax
import numpyro
from numpyro.diagnostics import gelman_rubin, effective_sample_size
from numpyro.infer import MCMC, NUTS, init_to_value, init_to_feasible
import arviz as az

# --- JAX / numpyro platform configuration (must precede library imports) ---
device_count = int(os.environ.get("TGRPOP_DEVICE_COUNT", 1))
numpyro.set_host_device_count(device_count)
platform = os.environ.get("TGRPOP_PLATFORM")
if platform is None:
    try:
        jax.devices("gpu")
        platform = "gpu"
    except Exception:
        platform = "cpu"
numpyro.set_platform(platform)
numpyro.enable_x64()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("Using %d devices on %s", device_count, platform)

from memory.hierarchical import (
    generate_data,
    generate_tgr_only_data,
    load_memory_data,
    make_tgr_only_model,
    make_joint_model,
    get_samples_df,
    create_plots,
)
from memory.hierarchical.data import (
    MIN_DETECTOR_FRAME_TOTAL_MASS,
    MIN_MASS_RATIO,
    _pick_waveform_label,
    _resolve_waveform_label,
    validate_posterior_prior_consistency,
    compute_log_prior_from_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_injection_file(args):
    """Return the injection file path from CLI args."""
    if args.injection_file is not None:
        return args.injection_file
    if args.injection_runs == "o4a":
        return os.path.join(
            REPO_DIR,
            "data/selection/mixture-real_o4a-cartesian_spins_20250503134659UTC.hdf",
        )
    if args.injection_runs == "o3+o4a":
        return os.path.join(
            REPO_DIR,
            "data/selection/mixture-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf",
        )
    raise ValueError(f"Unrecognized injection runs: {args.injection_runs}")


def _normalize_waveform_arg(waveform):
    """Return None for auto-selection, otherwise the requested waveform."""
    if waveform is None:
        return None
    normalized = waveform.strip()
    return None if normalized.lower() == "auto" else normalized


def _should_apply_nrsur_injection_cuts(waveform):
    """Return True when *waveform* explicitly restricts to NRSur7dq4."""
    return waveform is not None and "NRSur7dq4" in waveform


def _check_output_files_exist(outdir, analyze, no_plots):
    """Return (all_exist, existing, missing) for expected output files."""
    required = []
    for name in analyze:
        required += [
            os.path.join(outdir, f"result_{name}.nc"),
            os.path.join(outdir, f"fit_{name}_samples.dat"),
        ]
    if not no_plots:
        if analyze & {"memory", "joint"}:
            required += [
                os.path.join(outdir, "population_distribution.png"),
                os.path.join(outdir, "hyperparameters.png"),
                os.path.join(outdir, "tgr_comparison_corner.png"),
            ]
        if "joint" in analyze:
            required.append(os.path.join(outdir, "joint_model_corner.png"))
        if analyze & {"astro", "joint"}:
            required.append(os.path.join(outdir, "astro_corner.png"))

    existing = [f for f in required if os.path.exists(f)]
    missing = [f for f in required if not os.path.exists(f)]
    return len(existing) == len(required), existing, missing


def _collect_event_files(data_paths, exclude):
    """Glob *data_paths* and partition into kept / discarded event files."""
    kept, discarded = [], []
    for pattern in data_paths:
        for path in glob(pattern):
            if any(e in path for e in exclude):
                discarded.append(path)
            else:
                kept.append(path)
    return kept, discarded


def _filter_to_memory_events(event_files, memory_dir):
    """Keep only events that have a memory_results.h5 in *memory_dir*."""
    import re
    kept, skipped = [], []
    for ef in event_files:
        m = re.search(r"(GW\d{6}_\d{6})", os.path.basename(ef))
        if m and os.path.exists(
            os.path.join(memory_dir, m.group(1), "memory_results.h5")
        ):
            kept.append(ef)
        else:
            skipped.append(ef)
    return kept, skipped


def _save_provenance(outdir, injection_file, event_files, memory_dir):
    """Write small text files recording inputs for reproducibility."""
    with open(os.path.join(outdir, "injection_file.txt"), "w") as f:
        f.write(f"{injection_file}\n")
    with open(os.path.join(outdir, "event_files.txt"), "w") as f:
        for ef in event_files:
            f.write(f"{ef}\n")
    with open(os.path.join(outdir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(outdir, "memory_dir.txt"), "w") as f:
        f.write(f"{memory_dir}\n")


def _load_event_posteriors(event_files, waveform, per_event_labels=None):
    """Read posterior samples from each HDF5 event file.

    Each file is expected to contain at least one HDF5 group with a
    ``posterior_samples`` dataset.

    Selection priority (highest to lowest):
    1. *per_event_labels* — exact group name keyed by event name
       (GW\d{6}_\d{6}); used to match PE posteriors to pre-computed memory
       data that was generated from a specific waveform run.
    2. *waveform* — resolve the highest available ``CXX:<waveform>`` label
       in each file.
    3. Auto-select via NRSur > SEOB > IMRPhenom, C01 > C00 hierarchy when
       waveform is None.

    Returns
    -------
    posteriors : list of ndarray
    kept_files : list of str
        Subset of *event_files* for which posteriors were successfully loaded.
    """
    import re as _re2
    per_event_labels = per_event_labels or {}
    posteriors = []
    kept_files = []
    used_labels = {}
    for filename in event_files:
        basename = os.path.basename(filename)
        m = _re2.search(r"(GW\d{6}_\d{6})", basename)
        event_name = m.group(1) if m else None

        with h5py.File(filename, "r") as f:
            all_ps_keys = [
                k for k in f.keys()
                if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]
            ]
            # Prefer exact label from memory data for sample-count consistency
            if event_name and event_name in per_event_labels:
                exact = per_event_labels[event_name]
                if exact in all_ps_keys:
                    chosen = exact
                else:
                    logger.warning(
                        "%s: memory label '%s' not in PE file (%s); falling back",
                        basename, exact, all_ps_keys,
                    )
                    chosen = None
            else:
                chosen = None

            if chosen is None:
                try:
                    chosen = _resolve_waveform_label(all_ps_keys, waveform)
                except KeyError:
                    logger.warning(
                        "Skipping %s: no group with 'posterior_samples' matching"
                        " waveform=%r", basename, waveform,
                    )
                    continue
                if waveform is not None:
                    requested = waveform.split(":", 1)[-1]
                    matching_keys = [
                        key for key in all_ps_keys
                        if key.split(":", 1)[-1] == requested
                    ]
                else:
                    matching_keys = all_ps_keys
                logger.info(
                    "%s: selected PE group '%s'%s",
                    basename, chosen,
                    f" (available: {matching_keys})" if len(matching_keys) > 1 else "",
                )

            # Ensure the chosen group has log_prior/prior, or compute it
            ps_fields = f[chosen]["posterior_samples"].dtype.names
            computed_log_prior = None
            if "log_prior" not in ps_fields and "prior" not in ps_fields:
                # Try to reconstruct log_prior from the stored config
                _tmp_ps = f[chosen]["posterior_samples"][()]
                computed_log_prior, params_used = compute_log_prior_from_config(
                    f[chosen], _tmp_ps
                )
                if computed_log_prior is not None:
                    logger.info(
                        "%s: '%s' has no log_prior; reconstructed from config "
                        "(params: %s)",
                        basename, chosen, ", ".join(params_used[:6]),
                    )
                else:
                    # Fall back to next-best group that already has log_prior/prior
                    fallback_keys = [
                        k for k in all_ps_keys
                        if k != chosen
                        and (
                            "log_prior" in f[k]["posterior_samples"].dtype.names
                            or "prior" in f[k]["posterior_samples"].dtype.names
                        )
                    ]
                    if fallback_keys:
                        try:
                            fallback = _resolve_waveform_label(fallback_keys, waveform)
                        except KeyError:
                            fallback = _pick_waveform_label(fallback_keys)
                        logger.warning(
                            "%s: chosen group '%s' has no log_prior/prior field; "
                            "falling back to '%s'",
                            basename, chosen, fallback,
                        )
                        chosen = fallback
                    else:
                        logger.warning(
                            "%s: no group with log_prior/prior field found; skipping",
                            basename,
                        )
                        continue

            if computed_log_prior is not None:
                # Reuse the samples we already read for prior computation
                posterior_samples = _tmp_ps
                import numpy.lib.recfunctions as _rfn
                posterior_samples = _rfn.append_fields(
                    posterior_samples, "log_prior",
                    computed_log_prior, dtypes=float, usemask=False,
                )
            else:
                posterior_samples = f[chosen]["posterior_samples"][()]
            validate_posterior_prior_consistency(
                f[chosen],
                posterior_samples,
                filename=filename,
                label=chosen,
            )
            posteriors.append(posterior_samples)
            kept_files.append(filename)
            if event_name:
                used_labels[event_name] = chosen
    return posteriors, kept_files, used_labels


def _rescale_tgr_posterior(fit, A_scale):
    """Undo the --scale-tgr normalisation in-place on an ArviZ posterior.

    Multiplies ``mu_tgr`` and ``sigma_tgr`` by *A_scale* so the saved
    result is in physical (unscaled) units.
    """
    if A_scale != 1:
        fit.posterior["mu_tgr"] = fit.posterior["mu_tgr"] * A_scale
        fit.posterior["sigma_tgr"] = fit.posterior["sigma_tgr"] * A_scale


def _log_summary(fit, label):
    """Log mean, std, R-hat, and ESS for every non-diagnostic variable."""
    logger.info("%s results:", label)
    num_chains = fit.posterior.sizes.get("chain", 1)
    for var in fit.posterior:
        if "neff" in var:
            continue
        v = fit.posterior[var].values
        if v.ndim != 2:  # skip non-scalar variables (e.g. fracs)
            continue
        logger.info("  %s: %.3f +/- %.3f", var, float(v.mean()), float(v.std()))
        if num_chains >= 2:
            logger.info("    Rhat: %.3f", float(gelman_rubin(v)))
            logger.info("    ESS: %.1f", float(effective_sample_size(v)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run hierarchical GW population analysis with TGR parameters",
    )

    # -- Positional ---------------------------------------------------------
    parser.add_argument(
        "data_paths",
        type=str,
        nargs="+",
        help="Glob pattern(s) for PE posterior HDF5 files",
    )

    # -- Data selection -----------------------------------------------------
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="/mnt/home/kmitman/work/memory_pop/analysis",
        help=(
            "Directory with per-event memory results "
            "({dir}/{event_name}/memory_results.h5). "
            "Default: /mnt/home/kmitman/work/memory_pop/analysis"
        ),
    )
    parser.add_argument(
        "--waveform",
        type=str,
        default="auto",
        help=(
            "Waveform family to use for both PE and memory data. "
            "Use 'auto' to pick the best available label, or pass a bare "
            "waveform name like 'NRSur7dq4' to select the highest available "
            "'CXX:<waveform>' label per event."
        ),
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        help="Path to selection-injection HDF5 (supersedes --injection-runs)",
    )
    parser.add_argument(
        "--injection-runs",
        default="o4a",
        choices=["o4a", "o3+o4a"],
        help="Which injection campaign to use (default: o4a)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Substrings of event filenames to exclude (e.g. GW15 GW17)",
    )

    # -- MCMC configuration -------------------------------------------------
    parser.add_argument(
        "--n-warmup", type=int, default=1000, help="NUTS warmup iterations"
    )
    parser.add_argument(
        "--n-sample", type=int, default=1000, help="NUTS sampling iterations"
    )
    parser.add_argument(
        "--n-chains", type=int, default=4, help="Number of MCMC chains"
    )
    parser.add_argument(
        "--n-samples-per-event",
        type=int,
        default=2000,
        help="Posterior samples drawn per event for KDE / likelihood evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=150914, help="PRNG seed (0 = random)"
    )

    # -- Model selection ----------------------------------------------------
    parser.add_argument(
        "--analyze",
        nargs="+",
        choices=["memory", "astro", "joint"],
        default=["memory", "astro", "joint"],
        metavar="ANALYSIS",
        help=(
            "Which analyses to run (default: memory astro joint). "
            "'memory' = TGR-only model; 'astro' = astrophysical-only joint model "
            "(all events, no memory reweighting); 'joint' = astrophysical + TGR "
            "joint model (memory events only). "
            "'memory' and 'joint' require --memory-dir."
        ),
    )
    parser.add_argument(
        "--ignore-memory-weights",
        action="store_true",
        help=(
            "Zero out memory log-likelihood-ratio weights (log_weight) before "
            "passing to the model. Useful for diagnosing their effect."
        ),
    )
    parser.add_argument(
        "--scale-tgr",
        action="store_true",
        help=(
            "Normalise A_hat / A_sigma by their pooled std before fitting "
            "(posteriors are rescaled back before saving)"
        ),
    )
    parser.add_argument(
        "--mu-tgr-scale",
        type=float,
        default=None,
        help="Half-width of Uniform prior on mu_tgr (default: auto from data)",
    )
    parser.add_argument(
        "--sigma-tgr-scale",
        type=float,
        default=None,
        help="Upper bound of Uniform prior on sigma_tgr (default: auto from data)",
    )

    # -- Selection cuts -----------------------------------------------------
    parser.add_argument(
        "--ifar-threshold",
        type=float,
        default=1000,
        help="Inverse false-alarm rate threshold in years (default: 1000)",
    )
    # -- Output -------------------------------------------------------------
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        help="Output directory (default: results_memory)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip diagnostic plots"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if all output files already exist",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point: parse args, prepare data, run MCMC, save results."""
    args = _build_parser().parse_args()

    # --- Seed --------------------------------------------------------------
    if args.seed == 0:
        seed = np.random.randint(1 << 32)
        logger.warning("No PRNG key provided, using random seed: %d", seed)
    else:
        seed = args.seed
    prng = jax.random.PRNGKey(seed)

    # --- Paths & output directory -----------------------------------------
    injection_file = _resolve_injection_file(args)
    logger.info("Using injection file: %s", injection_file)

    outdir = args.outdir or "results_memory"
    os.makedirs(outdir, exist_ok=True)

    # --- Which analyses to run --------------------------------------------
    analyze = set(args.analyze)
    run_memory = "memory" in analyze
    run_astro  = "astro"  in analyze
    run_joint  = "joint"  in analyze
    need_memory_data = run_memory or run_joint
    waveform = _normalize_waveform_arg(args.waveform)

    if need_memory_data and args.memory_dir is None:
        raise ValueError(
            "--memory-dir is required for 'memory' and 'joint' analyses"
        )

    # Early exit if results already present
    all_exist, existing, missing = _check_output_files_exist(
        outdir, analyze, args.no_plots,
    )
    if all_exist and not args.force:
        logger.info("All output files already exist in %s", outdir)
        for f in existing:
            logger.info("  %s", os.path.basename(f))
        logger.info("Use --force to re-run the analysis")
        sys.exit(0)
    elif all_exist:
        logger.info("Output files exist but --force provided; re-running...")
    else:
        logger.info("Missing %d output files:", len(missing))
        for f in missing:
            logger.info("  %s", os.path.basename(f))

    logger.info("Running in output directory: %s", outdir)

    # --- Discover event files and load all posteriors ---------------------
    exclude = args.exclude + ["GW15", "GW17"]
    event_files, discarded_files = _collect_event_files(args.data_paths, exclude)

    logger.info("Discarded %d files", len(discarded_files))
    for f in discarded_files:
        logger.info("  %s", os.path.basename(f))

    if not event_files:
        raise FileNotFoundError(f"No event files found: {args.data_paths}")
    if not os.path.exists(injection_file):
        raise FileNotFoundError(f"Injection file not found: {injection_file}")

    # --- Filter to memory events and load memory data (if needed) ---------
    # Load memory data BEFORE PE posteriors so we can use the per-event
    # waveform labels to ensure PE samples match memory samples exactly.
    if need_memory_data:
        mem_files, skipped_files = _filter_to_memory_events(
            event_files, args.memory_dir
        )
        logger.info(
            "Found %d events with memory results (%d skipped)",
            len(mem_files), len(skipped_files),
        )
        if not mem_files:
            raise FileNotFoundError(
                f"No events with memory results found in {args.memory_dir}"
            )
        requested_mem_files = list(mem_files)
        memory_data = load_memory_data(mem_files, args.memory_dir, waveform)
        import re as _re
        loaded_memory_names = {md["event_name"] for md in memory_data}
        mem_files = [
            path for path in mem_files
            if _re.search(r"(GW\d{6}_\d{6})", os.path.basename(path)).group(1)
            in loaded_memory_names
        ]
        skipped_waveform_files = [
            path for path in requested_mem_files if path not in mem_files
        ]
        for path in skipped_waveform_files:
            logger.warning(
                "Skipping %s: requested waveform %r not found in memory results",
                os.path.basename(path),
                waveform,
            )
        logger.info("Loaded memory data for %d events", len(memory_data))
        # Build per-event label dict so PE loading uses the same waveform group
        per_event_labels = {md["event_name"]: md["waveform_label"] for md in memory_data}
    else:
        mem_files, memory_data, per_event_labels = [], None, {}

    all_posteriors, all_event_files, used_labels = _load_event_posteriors(
        event_files, waveform, per_event_labels=per_event_labels,
    )
    logger.info("Loaded posteriors for %d events", len(all_posteriors))

    if need_memory_data:
        # Build memory posteriors list aligned with mem_files (now guaranteed
        # to use matching waveform groups)
        import re as _re
        mem_file_set = set(mem_files)
        mem_posteriors = [
            p for p, f in zip(all_posteriors, all_event_files)
            if f in mem_file_set
        ]
        # Re-align memory_data to files that were actually loaded
        kept_names = {
            _re.search(r"(GW\d{6}_\d{6})", os.path.basename(f)).group(1)
            for f in all_event_files
            if f in mem_file_set
        }
        memory_data = [md for md in memory_data if md["event_name"] in kept_names]

        # Reconcile memory_data with actually-used PE groups.  When
        # _load_event_posteriors fell back to a different waveform (because
        # the memory-selected group lacks log_prior), reload the memory data
        # from that fallback group so sample counts always match.
        fixed_memory_data = []
        fixed_mem_posteriors = []
        for md, post in zip(memory_data, mem_posteriors):
            name = md["event_name"]
            used = used_labels.get(name)
            if used is not None and used != md["waveform_label"]:
                mem_path = os.path.join(args.memory_dir, name, "memory_results.h5")
                try:
                    with h5py.File(mem_path, "r") as mf:
                        if used in mf:
                            grp = mf[used]
                            md = {
                                "A_sample": np.asarray(grp["A_sample"][()].real),
                                "A_hat": np.asarray(grp["A_hat"][()].real),
                                "A_sigma": np.asarray(grp["A_sigma"][()].real),
                                "log_weight": np.asarray(grp["log_weight"][()].real),
                                "event_name": name,
                                "waveform_label": used,
                            }
                            logger.info(
                                "%s: reloaded memory using fallback waveform '%s'",
                                name, used,
                            )
                        else:
                            logger.warning(
                                "%s: memory file has no group '%s' (fallback); "
                                "skipping event",
                                name, used,
                            )
                            continue
                except Exception as exc:
                    logger.warning(
                        "%s: failed to reload memory for fallback group '%s': %s; "
                        "skipping event",
                        name, used, exc,
                    )
                    continue
            fixed_memory_data.append(md)
            fixed_mem_posteriors.append(post)
        memory_data = fixed_memory_data
        mem_posteriors = fixed_mem_posteriors
    else:
        mem_files, mem_posteriors = [], []

    # --- Provenance -------------------------------------------------------
    _save_provenance(
        outdir, injection_file,
        mem_files if need_memory_data else all_event_files,
        args.memory_dir,
    )

    # --- Build data arrays ------------------------------------------------
    apply_nrsur_injection_cuts = _should_apply_nrsur_injection_cuts(
        waveform
    )
    if apply_nrsur_injection_cuts:
        logger.info(
            "Applying NRSur7dq4 injection cuts (Mtot_det >= %.1f, q >= %.3f) "
            "for explicit NRSur7dq4 run (waveform=%r)",
            MIN_DETECTOR_FRAME_TOTAL_MASS,
            MIN_MASS_RATIO,
            waveform,
        )
    else:
        logger.info(
            "Not applying detector-frame mass or mass-ratio injection cuts "
            "because --waveform does not explicitly restrict the run to "
            "NRSur7dq4."
        )

    # Extract event names from file paths for IFAR lookup.
    _all_event_names = []
    for _f in all_event_files:
        _m = re.search(r"(GW\d{6}_\d{6})", os.path.basename(_f))
        _all_event_names.append(_m.group(1) if _m else os.path.basename(_f))
    _ifar_cache_file = os.path.join(outdir, "event_ifars.txt")

    _gen_kwargs = dict(
        injection_file=injection_file,
        ifar_threshold=args.ifar_threshold,
        min_detector_frame_total_mass=(
            MIN_DETECTOR_FRAME_TOTAL_MASS
            if apply_nrsur_injection_cuts else None
        ),
        min_mass_ratio=(
            MIN_MASS_RATIO if apply_nrsur_injection_cuts else None
        ),
        N_samples=args.n_samples_per_event,
        prng=seed,
        ignore_memory_weights=args.ignore_memory_weights,
        ifar_cache_file=_ifar_cache_file,
    )

    # astro: all events, uniform weights (no memory reweighting)
    if run_astro:
        (event_data_astro, inj_data_astro, BW_astro, BW_sel_astro,
         Nobs_astro, Ndraw_astro, _) = generate_data(
            all_posteriors, memory_data=None, use_tgr=False,
            scale_tgr=False, event_names=_all_event_names, **_gen_kwargs,
        )

    # joint: memory-filtered events, memory-reweighted samples
    # (use_tgr=True so event names are taken from memory_data)
    if run_joint:
        (event_data_joint, inj_data_joint, BW_joint, BW_sel_joint,
         Nobs_joint, Ndraw_joint, A_scale_joint) = generate_data(
            mem_posteriors, memory_data=memory_data, use_tgr=True,
            scale_tgr=args.scale_tgr, **_gen_kwargs,
        )

    # memory: TGR-only model data
    if run_memory:
        A_hats_mem, A_sigmas_mem, log_weights_mem, Nobs_mem, A_scale_mem = generate_tgr_only_data(
            mem_posteriors, memory_data,
            N_samples=args.n_samples_per_event, prng=seed,
            scale_tgr=args.scale_tgr,
            ignore_memory_weights=args.ignore_memory_weights,
            ifar_threshold=args.ifar_threshold,
            ifar_cache_file=_ifar_cache_file,
        )

    # --- MCMC -------------------------------------------------------------
    prng_astro, prng_joint, prng_mem = jax.random.split(prng, 3)

    def _run_joint_mcmc(prng_key, event_data, inj_data, BW, BW_sel,
                        Nobs, Ndraw, use_tgr, A_scale):
        # Physically motivated initialization avoids gradient explosion:
        # alpha_1 near the true BBH slope (~3.5) keeps neff_sel above threshold,
        # preventing the 4th-power neff penalty from collapsing the step size.
        # Init values based on results from a partially converged run on GWTC-4
        # events.  Starting near the posterior avoids the neff_sel gradient
        # explosion that collapses the NUTS step size.
        _init = {
            "alpha_1":     6.6,
            "alpha_2":    -0.7,
            "b":           0.23,
            "beta":        5.1,
            "fracs":       np.array([0.33, 0.40, 0.27]),
            "mu_peak_1":   10.0,
            "sigma_peak_1": 3.0,
            "mu_peak_2":   55.0,
            "sigma_peak_2": 3.0,
            "mu_spin":     0.20,
            "sigma_spin":  0.15,
            "lamb":        9.6,
        }
        if use_tgr:
            _init["mu_tgr"]    = 1.0
            _init["sigma_tgr"] = 0.5
        _init["f_iso"]      = 0.5
        _init["sigma_tilt"] = 1.0
        kernel = NUTS(make_joint_model,
                      init_strategy=init_to_value(values=_init))
        mcmc = MCMC(kernel, num_warmup=args.n_warmup,
                    num_samples=args.n_sample, num_chains=args.n_chains)
        mcmc.run(prng_key, event_data, inj_data, BW, BW_sel,
                 Nobs, Ndraw, use_tgr,
                 args.mu_tgr_scale, args.sigma_tgr_scale)
        fit = az.from_numpyro(mcmc)
        _rescale_tgr_posterior(fit, A_scale)
        return fit

    fit_astro = None
    if run_astro:
        logger.info("Running astro model (%d events)...", Nobs_astro)
        fit_astro = _run_joint_mcmc(
            prng_astro, event_data_astro, inj_data_astro,
            BW_astro, BW_sel_astro, Nobs_astro, Ndraw_astro,
            use_tgr=False, A_scale=1,
        )
        fname = os.path.join(outdir, "result_astro.nc")
        fit_astro.to_netcdf(fname)
        logger.info("Saved astro results: %s", fname)

    fit_joint = None
    if run_joint:
        logger.info("Running joint model (%d events)...", Nobs_joint)
        fit_joint = _run_joint_mcmc(
            prng_joint, event_data_joint, inj_data_joint,
            BW_joint, BW_sel_joint, Nobs_joint, Ndraw_joint,
            use_tgr=True, A_scale=A_scale_joint,
        )
        fname = os.path.join(outdir, "result_joint.nc")
        fit_joint.to_netcdf(fname)
        logger.info("Saved joint results: %s", fname)

    fit_memory = None
    if run_memory:
        logger.info("Running memory model (%d events)...", Nobs_mem)
        kernel = NUTS(make_tgr_only_model, init_strategy=init_to_feasible())
        mcmc = MCMC(kernel, num_warmup=args.n_warmup,
                    num_samples=args.n_sample, num_chains=args.n_chains)
        mcmc.run(prng_mem, A_hats_mem, A_sigmas_mem, log_weights_mem,
                 Nobs_mem, args.mu_tgr_scale, args.sigma_tgr_scale)
        fit_memory = az.from_numpyro(mcmc)
        _rescale_tgr_posterior(fit_memory, A_scale_mem)
        fname = os.path.join(outdir, "result_memory.nc")
        fit_memory.to_netcdf(fname)
        logger.info("Saved memory results: %s", fname)

    # --- Plots & sample CSVs ---------------------------------------------
    if not args.no_plots:
        logger.info("Creating plots...")
        create_plots(fit_astro, fit_joint, fit_memory, outdir)
    else:
        for fit, name in [
            (fit_astro,  "astro"),
            (fit_joint,  "joint"),
            (fit_memory, "memory"),
        ]:
            if fit is not None:
                get_samples_df(fit).to_csv(
                    os.path.join(outdir, f"fit_{name}_samples.dat"),
                    index=False, sep=" ",
                )

    # --- Summary statistics -----------------------------------------------
    for fit, label in [
        (fit_astro,  "Astro model"),
        (fit_joint,  "Joint model"),
        (fit_memory, "Memory model"),
    ]:
        if fit is not None:
            _log_summary(fit, label)

    logger.info("Analysis complete! Results saved to %s", outdir)


if __name__ == "__main__":
    main()
