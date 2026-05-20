"""Compare joint_model_corner across any number of result .nc files.

Usage
-----
    uv run scripts/plot_compare_corner.py result1.nc result2.nc [result3.nc ...]
        [--labels run_a run_b run_c]
        [--output comparison_corner.png]
        [--vars alpha_1 mu_tgr ...]   # subset of variables to plot
        [--dpi 150]

If --vars is not given the union of _ASTRO_VARS + TGR vars is used, restricted
to variables that are present in every loaded file.

1-D marginals use density=True so histograms are properly normalised even when
runs have different numbers of samples.
"""

import argparse
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from corner import corner


_ASTRO_VARS = [
    "alpha_1", "alpha_2", "b", "beta",
    "frac_bpl", "frac_peak_1", "mu_peak_1", "sigma_peak_1",
    "frac_peak_2", "mu_peak_2", "sigma_peak_2",
    "mu_spin", "sigma_spin", "lamb",
    "f_iso", "mu_tilt", "sigma_tilt",
]

_TGR_VARS = ["mu_tgr", "sigma_tgr"]


def load_flat_samples(path, var_names):
    """Return (n_samples, n_vars) array from an ArviZ NetCDF file."""
    idata = az.from_netcdf(path)
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    cols = []
    for v in var_names:
        arr = stacked[v].values
        if arr.ndim != 1:
            raise ValueError(
                f"Variable '{v}' in {path} has shape {arr.shape}; "
                "only scalar variables are supported."
            )
        cols.append(arr)
    return np.column_stack(cols)


def available_vars(path):
    """Return the set of scalar variable names in an nc file's posterior."""
    idata = az.from_netcdf(path)
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    return {
        k for k, v in stacked.data_vars.items()
        if v.values.ndim == 1
    }


def main():
    parser = argparse.ArgumentParser(
        description="Corner-plot comparison across multiple result .nc files."
    )
    parser.add_argument("nc_files", nargs="+", metavar="NC_FILE",
                        help="ArviZ NetCDF result files to compare.")
    parser.add_argument("--labels", nargs="*", metavar="LABEL",
                        help="Display labels (one per file); defaults to run0, run1, …")
    parser.add_argument("--output", "-o", default="comparison_corner.png",
                        help="Output file path (default: comparison_corner.png).")
    parser.add_argument("--vars", nargs="+", metavar="VAR",
                        help="Variables to plot (default: all shared astro+TGR vars).")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150).")
    args = parser.parse_args()

    nc_files = args.nc_files
    labels = args.labels or [f"run{i}" for i in range(len(nc_files))]

    if len(labels) != len(nc_files):
        parser.error(
            f"--labels has {len(labels)} entries but {len(nc_files)} files were given."
        )

    # Determine variables to plot
    if args.vars:
        var_names = args.vars
    else:
        # Start with desired ordering; keep only vars present in every file
        desired = _ASTRO_VARS + _TGR_VARS
        shared = None
        for path in nc_files:
            avail = available_vars(path)
            shared = avail if shared is None else shared & avail
        var_names = [v for v in desired if v in shared]
        if not var_names:
            print("ERROR: no shared scalar variables found across all files.", file=sys.stderr)
            sys.exit(1)

    print(f"Plotting {len(var_names)} variables: {var_names}")

    # Cycle through matplotlib default colors
    colors = [f"C{i}" for i in range(len(nc_files))]

    fig = None
    for path, label, color in zip(nc_files, labels, colors):
        print(f"  Loading {path} ({label}) …", end=" ", flush=True)
        data = load_flat_samples(path, var_names)
        print(f"{data.shape[0]} samples")

        fig = corner(
            data,
            labels=var_names,
            fig=fig,
            color=color,
            plot_density=False,
            plot_contours=True,
            # density=True normalises each 1-D histogram independently so
            # runs with different sample counts are directly comparable
            hist_kwargs={"density": True},
            contour_kwargs={"linewidths": 1.0},
        )

    # Legend on the first diagonal axis
    axes = np.array(fig.axes).reshape(len(var_names), len(var_names))
    for label, color in zip(labels, colors):
        axes[0, 0].plot([], [], color=color, label=label)
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
