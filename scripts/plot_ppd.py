#!/usr/bin/env python3
"""Plot 1D Population Predictive Distributions (PPDs) from hierarchical analysis.

For each input NetCDF file, draws the PPD (median + 90% CI shaded band) for:
  - Primary mass  m1          on semi-log-y axes
  - Mass ratio    q = m2/m1   marginalised over m1
  - Spin magnitude a           (Gaussian model, truncated to [0, 1])

Multiple NetCDF files can be overlaid on the same axes for comparison.

Rate mode (--injection-file)
----------------------------
Without an injection file the m1 panel shows the normalised population PDF
p(m1).  Supplying --injection-file switches the m1 panel to the differential
merger rate dR/dm1 [Gpc^-3 yr^-1 M_sun^-1]:

    dR/dm1(m1 | Lambda) = R(Lambda) * p(m1 | Lambda)

The total volumetric merger rate is drawn from the conditional posterior
given each hyperposterior sample:

    p(R | Lambda, data) = Gamma(k=N_obs, rate=T_obs * beta(Lambda))     (1)

where:
  - N_obs   is the number of observed events used in the analysis
  - T_obs   is the live observing time in years (from injection file metadata)
  - beta(Lambda) is the effective surveyed comoving volume [Gpc^3]:

        beta(Lambda) = (1/N_draw) * sum_{found} p_pop(theta | Lambda) / p_draw(theta)  (2)

  - N_draw  is the total number of injections attempted (found + missed)
  - theta   denotes the parameters (m1, q, z, a1, a2, cos_tilt_1, cos_tilt_2)
            of each found injection
  - p_pop   is the population model evaluated at theta (same parameterisation as
            the MCMC model; normalised so that the z integral gives Gpc^3)
  - p_draw  is the injection draw density in
            (m1, q, z, a1, a2, cos_tilt_1, cos_tilt_2) space

Equation (2) is evaluated once per posterior sample to obtain beta(Lambda),
then one conditional Gamma draw is taken for each sample to obtain
R_samples. Under the usual log-uniform rate prior pi(R) propto 1/R, this is
the exact rate posterior conditional on Lambda.
The PPD is evaluated at z = 0.2 (the LVK populations paper convention):

    R(z=0.2) = R0 * (1+0.2)^lambda

where lambda is the redshift power-law index from the posterior.  No
dz/dVc Jacobian is needed: beta already has units of Gpc^3 (comoving
volume), so R0 is already per unit comoving volume.  The (1+z)^lambda
factor simply evaluates the merger rate at z=0.2 instead of z=0.

The PPD is then dR/dm1(z=0.2) = R(z=0.2)[:, newaxis] * p(m1 | Lambda_samples).

Draw prior convention
~~~~~~~~~~~~~~~~~~~~~
For consistency with the archived hierarchical inference results, this script
uses the same injection draw-density convention as
`memory.hierarchical.data.read_injection_file()` when computing beta. In
particular, the per-injection `weights` field enters with a minus sign in
`log_p_draw`, so subtracting `log_p_draw` in the importance sum upweights
injections from more sensitive observing periods in the same way as the model.

Usage:
    python scripts/plot_ppd.py result_astro.nc
    python scripts/plot_ppd.py result_astro.nc result_joint.nc --labels astro joint
    python scripts/plot_ppd.py result_astro.nc --outdir /path/to/dir --n-ppd 1000
    python scripts/plot_ppd.py result_astro.nc --injection-file sel.hdf --n-obs 43
"""

import argparse
import os

from memory.hierarchical.ppd import generate_ppd, _default_rate_workers


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
        "--rate-chunk", type=int, default=50,
        help=(
            "Posterior chunk size for the rate calculation (default: 50). "
            "Lower values reduce memory; higher values may run faster."
        ),
    )
    parser.add_argument(
        "--rate-workers", type=int, default=_default_rate_workers(),
        help=(
            "Number of worker processes for the rate calculation "
            "(default: min(4, cpu_count))."
        ),
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
    parser.add_argument(
        "--injection-file", default=None,
        help=(
            "HDF5 sensitivity-estimate file.  When provided, the m1 panel "
            "shows dR/dm1 [Gpc⁻³ yr⁻¹ M☉⁻¹] instead of p(m1)."
        ),
    )
    parser.add_argument(
        "--n-obs", type=int, default=None,
        help=(
            "Number of observed events used in the analysis "
            "(required when --injection-file is given)."
        ),
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.nc_files):
        parser.error("--labels must match the number of nc_files")

    if args.injection_file is not None and args.n_obs is None:
        parser.error("--n-obs is required when --injection-file is given")

    labels = args.labels or [
        os.path.splitext(os.path.basename(f))[0] for f in args.nc_files
    ]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.nc_files[0]))

    generate_ppd(
        nc_files=args.nc_files,
        outdir=outdir,
        labels=labels,
        injection_file=args.injection_file,
        n_obs=args.n_obs,
        n_ppd=args.n_ppd,
        seed=args.seed,
        rate_chunk=args.rate_chunk,
        rate_workers=args.rate_workers,
        n_m1_grid=args.n_m1_grid,
        n_q_grid=args.n_q_grid,
        n_m1_q_grid=args.n_m1_q_grid,
        n_a_grid=args.n_a_grid,
    )


if __name__ == "__main__":
    main()
