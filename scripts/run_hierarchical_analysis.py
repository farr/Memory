#!/usr/bin/env python3
"""
Script to run gravitational wave population analysis with TGR parameters.
Converted from analysis_notebook.ipynb
"""

import sys
import os
import argparse
from glob import glob
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"

# Import required libraries
import numpy as np
import h5py
import jax
import numpyro
from numpyro.diagnostics import gelman_rubin, effective_sample_size
from numpyro.infer import MCMC, NUTS, init_to_feasible
import arviz as az

# Configure numpyro
device_count = int(os.environ.get("TGRPOP_DEVICE_COUNT", 1))
numpyro.set_host_device_count(device_count)
platform = os.environ.get("TGRPOP_PLATFORM")
if platform is None:
    try:
        # Prefer GPU when present, otherwise fall back to CPU.
        jax.devices("gpu")
        platform = "gpu"
    except Exception:
        platform = "cpu"
numpyro.set_platform(platform)
numpyro.enable_x64()

print(f"Using {device_count} devices on {platform}")

# Import library functions (after JAX/numpyro platform config)
from memory.hierarchical import (
    generate_data,
    generate_tgr_only_data,
    load_memory_data,
    make_tgr_only_model,
    make_joint_model,
    get_samples_df,
    create_plots,
)


def main():
    """CLI entry point for the hierarchical population analysis.

    Parses command-line arguments, loads event posteriors from HDF5 files,
    runs the joint and/or TGR-only MCMC via numpyro NUTS, saves results
    as NetCDF and CSV, optionally creates diagnostic plots, and prints
    summary statistics with R-hat and effective sample size.
    """
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
    parser.add_argument(
        "--memory-dir",
        type=str,
        required=True,
        help=(
            "Directory with per-event memory results "
            "({dir}/{event_name}/memory_results.h5). "
            "A_sample and log_weight from these files supply the TGR parameter."
        ),
    )
    parser.add_argument(
        "--waveform-label",
        type=str,
        default=None,
        help=(
            "HDF5 group name in memory results files "
            "(falls back to --param-key, then first available group)"
        ),
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
            injection_file = os.path.join(REPO_DIR,
                                      "data/selection/mixture-real_o4a-cartesian_spins_20250503134659UTC.hdf")
        elif args.injection_runs == 'o3+o4a':
            injection_file = os.path.join(REPO_DIR,
                                      "data/selection/mixture-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf")
        else:
            raise ValueError(f"Unrecognized injection runs: {args.injection_runs}")
    else:
        injection_file = args.injection_file
    print(f"Using injection file: {injection_file}")

    # Generate the output directory
    outdir = args.outdir or f'results_{args.parameter}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def check_output_files_exist():
        """Check if all expected output files already exist in outdir.

        Returns (all_exist, existing_files, missing_files).
        """
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
            "Output files exist but --force flag provided. Re-running analysis..."
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

    # Load memory data
    waveform_label = args.waveform_label or args.param_key
    memory_data = load_memory_data(event_files, args.memory_dir, waveform_label)
    print(f"Loaded memory data for {len(memory_data)} events from {args.memory_dir}")

    # Save memory_dir path to output directory
    memory_dir_path = os.path.join(outdir, "memory_dir.txt")
    with open(memory_dir_path, "w") as f:
        f.write(f"{args.memory_dir}\n")
    print(f"Saved memory_dir path to: {memory_dir_path}")

    # Loading in the event posteriors
    event_posteriors = []
    for filename in event_files:
        with h5py.File(filename, "r") as f:
            if args.param_key:
                # Search for matching groups with posterior samples.
                keys = [
                    k for k in f.keys()
                    if (
                        args.param_key in k
                        and isinstance(f[k], h5py.Group)
                        and "posterior_samples" in f[k]
                    )
                ]
            else:
                # Search for groups containing "posterior_samples".
                keys = [
                    k for k in f.keys()
                    if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]
                ]
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

    # store dphi_scale in output directory
    path = os.path.join(outdir, "dphi_scale_joint.txt")
    with open(path, "w") as f:
        f.write(f"{dphi_scale}\n")
    print(f"Saved dphi_scale to: {path}")

    if args.model in ("tgr", "both"):
        A_hats_tgr, A_sigmas_tgr, _, dphi_scale = generate_tgr_only_data(
            event_posteriors, memory_data,
            N_samples=args.n_samples_per_event, prng=seed,
            scale_tgr=args.scale_tgr,
        )
    else:
        A_hats_tgr, A_sigmas_tgr, _, dphi_scale = None, None, None, 1

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
            A_hats_tgr,
            A_sigmas_tgr,
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
