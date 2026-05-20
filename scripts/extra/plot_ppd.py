#!/usr/bin/env python3
"""Plot 1D Population Predictive Distributions (PPDs) from hierarchical analysis.

For each input NetCDF file, draws the PPD (median + 90% CI shaded band) for:
  - Primary mass  m1          on semi-log-y axes
  - Mass ratio    q = m2/m1   marginalised over m1
  - Spin magnitude a           (Gaussian model, truncated to [0, 1])

Multiple NetCDF files can be overlaid on the same axes for comparison.

Rate mode
---------
When the posterior trace contains ``R`` (the local merger-rate density
R(z=0) in Gpc^-3 yr^-1, drawn in the model via the Gamma auxiliary
variable), the m1 panel automatically shows the differential merger rate

    dR/dm1(m1, z=0.2) = R_0 * (1+0.2)^lambda * p(m1 | Lambda)

in units of Gpc^-3 yr^-1 M_sun^-1, matching the LVK populations paper
convention.

Usage:
    python scripts/plot_ppd.py result_astro.nc
    python scripts/plot_ppd.py result_astro.nc result_joint.nc --labels astro joint
    python scripts/plot_ppd.py result_astro.nc --outdir /path/to/dir --n-ppd 1000
"""

import argparse
import os

from memory.hierarchical.ppd import generate_ppd


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
    parser.add_argument(
        "--n-m1-grid", type=int, default=1200,
        help=(
            "Number of primary-mass grid cells used for the PPD integral "
            "(default: 1200). Higher values reduce visible stair-step "
            "artifacts in the mass-ratio panel."
        ),
    )
    parser.add_argument(
        "--n-q-grid", type=int, default=1200,
        help=(
            "Number of mass-ratio grid points used for plotting "
            "(default: 1200). Higher values make the q PPD look smoother."
        ),
    )
    parser.add_argument(
        "--n-m1-q-grid", type=int, default=None,
        help=(
            "Number of primary-mass grid cells used internally when "
            "marginalizing to p(q). Defaults to max(--n-m1-grid, 4000) "
            "for a smoother q panel."
        ),
    )
    parser.add_argument(
        "--n-a-grid", type=int, default=200,
        help="Number of spin-magnitude grid points used for plotting (default: 200).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.nc_files):
        parser.error("--labels must match the number of nc_files")

    labels = args.labels or [
        os.path.splitext(os.path.basename(f))[0] for f in args.nc_files
    ]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.nc_files[0]))

    generate_ppd(
        nc_files=args.nc_files,
        outdir=outdir,
        labels=labels,
        n_ppd=args.n_ppd,
        seed=args.seed,
        n_m1_grid=args.n_m1_grid,
        n_q_grid=args.n_q_grid,
        n_m1_q_grid=args.n_m1_q_grid,
        n_a_grid=args.n_a_grid,
    )


if __name__ == "__main__":
    main()
