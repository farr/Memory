#!/usr/bin/env python3
"""Run hierarchical Bayesian population analysis with GW memory TGR parameters.

This script performs a hierarchical inference on gravitational-wave posterior
samples to constrain the population distribution of a memory-amplitude
deviation parameter *A*.  Two models are available:

* **joint** — simultaneously fits astrophysical population parameters
  (mass function, spin distribution, redshift evolution) and TGR
  hyperparameters (mu_tgr, sigma_tgr) using selection-effect corrections.
* **tgr** — fits only the TGR hyperparameters, marginalising over
  astrophysical parameters via importance-weighted posterior samples.

Both models treat the per-sample memory measurement as a Gaussian
likelihood N(A | A_hat, A_sigma) and analytically convolve it with the
population prior N(A | mu_tgr, sigma_tgr).

Outputs (written to ``--outdir``):
    result_joint.nc / result_tgr.nc
        ArviZ InferenceData with full MCMC posterior (NetCDF).
    fit_joint_samples.dat / fit_tgr_samples.dat
        Flat CSV of posterior samples (space-delimited).
    population_distribution.png, hyperparameters.png,
    tgr_comparison_corner.png, joint_model_corner.png
        Diagnostic plots (skipped with ``--no-plots``).
    injection_file.txt, event_files.txt, memory_dir.txt, command.txt
        Provenance files recording the exact inputs and command line.

Environment variables:
    TGRPOP_PLATFORM   Force JAX platform ('cpu' or 'gpu'); auto-detected
                      if unset.
    TGRPOP_DEVICE_COUNT  Number of JAX devices for chain parallelisation
                         (default 1).
    OMP_NUM_THREADS   Set to 1 at import time to avoid thread contention.
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
from numpyro.infer import MCMC, NUTS, init_to_feasible
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


def _check_output_files_exist(outdir, model, no_plots):
    """Return (all_exist, existing, missing) for expected output files."""
    required = []
    if model in ("joint", "both"):
        required += [
            os.path.join(outdir, "result_joint.nc"),
            os.path.join(outdir, "fit_joint_samples.dat"),
        ]
    if model in ("tgr", "both"):
        required += [
            os.path.join(outdir, "result_tgr.nc"),
            os.path.join(outdir, "fit_tgr_samples.dat"),
        ]
    if not no_plots:
        required += [
            os.path.join(outdir, "population_distribution.png"),
            os.path.join(outdir, "hyperparameters.png"),
            os.path.join(outdir, "tgr_comparison_corner.png"),
        ]
        if model in ("joint", "both"):
            required.append(os.path.join(outdir, "joint_model_corner.png"))

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


def _load_event_posteriors(event_files, param_key):
    """Read posterior samples from each HDF5 event file.

    Each file is expected to contain exactly one HDF5 group with a
    ``posterior_samples`` dataset.  If *param_key* is given, only groups
    whose name contains that key are considered.
    """
    posteriors = []
    for filename in event_files:
        with h5py.File(filename, "r") as f:
            if param_key:
                keys = [
                    k for k in f.keys()
                    if (
                        param_key in k
                        and isinstance(f[k], h5py.Group)
                        and "posterior_samples" in f[k]
                    )
                ]
            else:
                keys = [
                    k for k in f.keys()
                    if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]
                ]
            if len(keys) != 1:
                raise KeyError(
                    f"Expected 1 key with 'posterior_samples' in {filename}, "
                    f"found {len(keys)}: {keys}"
                )
            posteriors.append(f[keys[0]]["posterior_samples"][()])
    return posteriors


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
        mean_val = fit.posterior[var].mean().values
        std_val = fit.posterior[var].std().values
        logger.info("  %s: %.3f +/- %.3f", var, mean_val, std_val)
        if num_chains >= 2:
            logger.info(
                "    Rhat: %.3f",
                gelman_rubin(fit.posterior[var].values),
            )
            logger.info(
                "    ESS: %.1f",
                effective_sample_size(fit.posterior[var].values),
            )


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
        "parameter",
        type=str,
        help="TGR parameter label used for plot axes (e.g. 'dchi_2')",
    )
    parser.add_argument(
        "data_paths",
        type=str,
        nargs="+",
        help="Glob pattern(s) for PE posterior HDF5 files",
    )

    # -- Data selection -----------------------------------------------------
    parser.add_argument(
        "--param-key",
        type=str,
        help="HDF5 group name filter for posterior samples (default: auto)",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        required=True,
        help=(
            "Directory with per-event memory results "
            "({dir}/{event_name}/memory_results.h5)"
        ),
    )
    parser.add_argument(
        "--waveform-label",
        type=str,
        default=None,
        help=(
            "HDF5 group inside memory results files "
            "(falls back to --param-key, then first available group)"
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
        "--model",
        choices=["joint", "tgr", "both"],
        default="both",
        help="Which model(s) to run (default: both)",
    )
    parser.add_argument(
        "--use-tilts",
        action="store_true",
        help="Include spin-tilt mixture model in the joint fit",
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
    parser.add_argument(
        "--snr-cut", type=float, default=0, help="Network SNR threshold"
    )
    parser.add_argument(
        "--snr-inspiral-cut",
        type=float,
        default=6,
        help="Inspiral SNR threshold (default: 6)",
    )

    # -- Output -------------------------------------------------------------
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        help="Output directory (default: results_{parameter})",
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

    outdir = args.outdir or f"results_{args.parameter}"
    os.makedirs(outdir, exist_ok=True)

    # Early exit if results already present
    all_exist, existing, missing = _check_output_files_exist(
        outdir, args.model, args.no_plots,
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

    # --- Discover and filter event files ----------------------------------
    exclude = args.exclude + ["GW15", "GW17"]
    event_files, discarded_files = _collect_event_files(args.data_paths, exclude)

    logger.info("Discarded %d files", len(discarded_files))
    for f in discarded_files:
        logger.info("  %s", os.path.basename(f))

    if not event_files:
        raise FileNotFoundError(f"No event files found: {args.data_paths}")
    if not os.path.exists(injection_file):
        raise FileNotFoundError(f"Injection file not found: {injection_file}")

    logger.info("Found %d event files", len(event_files))

    # --- Provenance -------------------------------------------------------
    _save_provenance(outdir, injection_file, event_files, args.memory_dir)

    # --- Load data --------------------------------------------------------
    waveform_label = args.waveform_label or args.param_key
    memory_data = load_memory_data(event_files, args.memory_dir, waveform_label)
    logger.info(
        "Loaded memory data for %d events from %s",
        len(memory_data), args.memory_dir,
    )

    event_posteriors = _load_event_posteriors(event_files, args.param_key)

    # --- Build data arrays for the joint model ----------------------------
    (
        event_data_array,
        injection_data_array,
        BW_matrices,
        BW_matrices_sel,
        Nobs,
        Ndraw,
        A_scale_joint,
    ) = generate_data(
        event_posteriors,
        injection_file,
        memory_data,
        ifar_threshold=args.ifar_threshold,
        use_tgr=True,
        snr_inspiral_cut=args.snr_inspiral_cut,
        N_samples=args.n_samples_per_event,
        snr_cut=args.snr_cut,
        use_tilts=args.use_tilts,
        prng=seed,
        scale_tgr=args.scale_tgr,
    )

    # --- Build data arrays for the TGR-only model -------------------------
    if args.model in ("tgr", "both"):
        A_hats_tgr, A_sigmas_tgr, _, A_scale_tgr = generate_tgr_only_data(
            event_posteriors, memory_data,
            N_samples=args.n_samples_per_event, prng=seed,
            scale_tgr=args.scale_tgr,
        )
    else:
        A_hats_tgr, A_sigmas_tgr, _, A_scale_tgr = None, None, None, 1

    # --- MCMC -------------------------------------------------------------
    prng0, prng1 = jax.random.split(prng, 2)

    fit_joint = None
    if args.model in ("joint", "both"):
        logger.info("Running joint model...")
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
        _rescale_tgr_posterior(fit_joint, A_scale_joint)

        fname = os.path.join(outdir, "result_joint.nc")
        fit_joint.to_netcdf(fname)
        logger.info("Saved joint results: %s", fname)

    fit_tgr = None
    if args.model in ("tgr", "both"):
        logger.info("Running TGR-only model...")
        kernel = NUTS(make_tgr_only_model, init_strategy=init_to_feasible())
        mcmc = MCMC(
            kernel,
            num_warmup=args.n_warmup,
            num_samples=args.n_sample,
            num_chains=args.n_chains,
        )
        mcmc.run(
            prng1,
            A_hats_tgr,
            A_sigmas_tgr,
            Nobs,
            args.mu_tgr_scale,
            args.sigma_tgr_scale,
        )
        fit_tgr = az.from_numpyro(mcmc)
        _rescale_tgr_posterior(fit_tgr, A_scale_tgr)

        fname = os.path.join(outdir, "result_tgr.nc")
        fit_tgr.to_netcdf(fname)
        logger.info("Saved TGR-only results: %s", fname)

    # --- Plots & sample CSVs ---------------------------------------------
    if not args.no_plots:
        logger.info("Creating plots...")
        create_plots(fit_joint, fit_tgr, args.parameter, outdir)
    else:
        if fit_joint is not None:
            get_samples_df(fit_joint).to_csv(
                os.path.join(outdir, "fit_joint_samples.dat"),
                index=False, sep=" ",
            )
        if fit_tgr is not None:
            get_samples_df(fit_tgr).to_csv(
                os.path.join(outdir, "fit_tgr_samples.dat"),
                index=False, sep=" ",
            )

    # --- Summary statistics -----------------------------------------------
    if fit_joint is not None:
        _log_summary(fit_joint, "Joint model")
    if fit_tgr is not None:
        _log_summary(fit_tgr, "TGR-only model")

    logger.info("Analysis complete! Results saved to %s", outdir)


if __name__ == "__main__":
    main()
