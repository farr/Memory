# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian hierarchical analysis pipeline for testing general relativity (TGR) using gravitational wave memory effects in binary mergers. Uses posterior samples from GW parameter estimation runs, builds population-level models with numpyro/JAX, and performs MCMC inference.

**Python 3.11 only** (pinned in `.python-version`). Package manager is **uv**.

## Setup and Commands

### Install dependencies
```bash
uv sync
```

### Download waveform data assets
```bash
cd data && ./download.sh
```
This fetches `NRSur7dq4_v1.0.h5` (surrogate model) and `posterior_samples_NRSur7dq4.h5` from Zenodo, and creates the `NRSur7dq4.h5` symlink.

### Run the main analysis
```bash
uv run python scripts/run_hierarchical_analysis.py \
    dchi_2 "path/to/posteriors/*.h5" \
    --injection-file path/to/injections.hdf \
    --model both --n-warmup 1000 --n-sample 1000 --n-chains 4 \
    --outdir results/
```
Key args: `--model {joint,tgr,both}`, `--scale-tgr`, `--use-tilts`, `--no-plots`, `--force`.

### End-to-end smoke test
```bash
./tests/get_test_data.sh --num-events 2          # download small dataset from LIGO servers
./tests/test_run_analysis_e2e.sh --num-events 2 --model both  # run analysis
```
Results go to `data/test_e2e/`. The test script auto-downloads data if missing and falls back to available TGR parameters if the requested one isn't present.

### Environment variables
- `TGRPOP_PLATFORM`: `cpu` or `gpu` (auto-detected; test scripts default to `cpu`)
- `TGRPOP_DEVICE_COUNT`: JAX device parallelization count
- `OMP_NUM_THREADS`: thread count (defaults to 1 in test scripts)
- `REMOTE_USER`, `REMOTE_HOST`: SSH credentials for `get_test_data.sh` (default host: `ldas-grid.ligo.caltech.edu`)

## Architecture

### `memory/` package

**`memory/gw_residuals.py`** — Low-level GW data handling and residual computation.
- `AnalysisConfig` dataclass for config management
- Config parsing with multi-convention fallback (`_get_cfg`, `_cfg_first_match`)
- GWOSC strain downloading, bilby WaveformGenerator/InterferometerList setup
- Spline calibration attachment with flexible key matching
- `compute_one_sample_fd()`: main entry point for frequency-domain residual computation

**`memory/gw_memory.py`** — Surrogate waveform evaluation, memory physics, and detector projection.
- `evaluate_surrogate_with_LAL()`: evaluate NRSur7dq4 (or other approximants) via LALSimulation for SH modes
- `compute_memory_correction()`: GW memory correction via Wigner 3j angular integrals
- `compute_memory_and_map_to_polarizations()`: memory correction mapped to h+/hx polarizations
- `make_memories()`: end-to-end memory computation from a residual result object
- `polarizations_to_FD()`: FFT polarizations to frequency domain with roll-on window
- `project_to_detectors()`: project polarizations onto detector network
- `compute_memory_variables_likelihoods_and_weights()`: compute memory SNRs, likelihoods, and Bayes factors

**`memory/kde_contour.py`** — Bounded 1D/2D KDE implementations for posterior visualization.

**`memory/hierarchical/`** — Subpackage for hierarchical TGR population analysis (data loading, numpyro models, plotting).

- **`memory/hierarchical/data.py`** — Data loading and preparation.
  - `read_injection_file()`: reads an HDF5 injection/selection file, applies IFAR and SNR cuts, computes derived spin quantities (chi_eff, chi_p), and returns a dict of arrays with injection parameters and draw priors
  - `generate_data()`: builds per-event data arrays for the joint model — resamples posteriors with importance weights, assembles (m1, q, spins, redshift, TGR param) arrays, computes KDE bandwidth matrices via conditional covariance, and loads the injection data
  - `generate_tgr_only_data()`: builds the simplified data arrays for the TGR-only model — resamples the TGR parameter from each event posterior and computes per-event 1D KDE bandwidths

- **`memory/hierarchical/models.py`** — Numpyro model definitions and cosmology globals.
  - Module-level cosmology setup: `Planck15_LAL`, `zinterp`, `dVdzdt_interp`
  - `make_tgr_only_model()`: numpyro model with two hyperparameters (mu_tgr, sigma_tgr) describing a Gaussian population distribution for the TGR deviation parameter; likelihood is a KDE-smoothed sum over per-event posterior samples
  - `make_joint_model()`: numpyro model jointly fitting astrophysical population (broken power law + two Gaussian peaks mass function, power-law mass ratio, power-law redshift, spin magnitude/tilt distributions) and TGR hyperparameters; includes selection-effect correction and effective-sample-size regularization

- **`memory/hierarchical/plotting.py`** — Plotting and ArviZ post-processing.
  - `get_samples_df()`: converts an ArviZ InferenceData posterior into a flat pandas DataFrame (chains × draws)
  - `create_plots()`: generates diagnostic plots — population KDE, hyperparameter pairplot, TGR corner plot, and full joint corner plot; saves PNGs and per-fit CSV sample files

### Scripts

**`scripts/run_hierarchical_analysis.py`** — Thin CLI wrapper for the hierarchical population analysis. Imports functions from `memory.hierarchical`, handles argument parsing, file I/O, and MCMC orchestration.
- `main()`: CLI entry point — parses arguments, loads event posteriors from HDF5, runs joint and/or TGR-only MCMC via NUTS, saves results as NetCDF/CSV, optionally creates plots, and prints summary statistics with R-hat and ESS
- Outputs: NetCDF (`.nc`), CSV (`.dat`), corner plots (PNG)

**`scripts/compute_gw_memory_for_GWTC.py`** — Catalog-level GW memory computation for GWTC events.
- Computes memory waveforms, detector projections, and likelihoods across a catalog
- Uses multiprocessing for parallel event processing
- Outputs per-event `{output_dir}/{event_name}/memory_results.h5` with datasets: `A_hat` (ML amplitude), `A_sigma` (posterior std), `A_sample` (posterior draws), `log_weight`, `log_likelihood`, grouped by waveform label
- Currently only the surrogate waveform path is working

### Data flow
- **TGR population analysis:** Event posteriors (HDF5) → KDE smoothing → numpyro hierarchical model → NUTS MCMC → ArviZ posterior → NetCDF + plots
- **Memory computation:** Event posteriors → surrogate waveform → memory correction → detector projection → per-event `memory_results.h5` (amplitude posteriors, likelihoods, weights)

### Memory likelihood math (Farr et al.)

The memory signal model adds a memory waveform `h_m(θ)` scaled by amplitude `A_m` to the residual `r(θ) = d - R(θ)h(θ)`. GR predicts `A_m = 1`.

Key quantities per posterior sample (computed in `compute_memory_variables_likelihoods_and_weights`):
- **Inner products:** `hrs = <h_m|r>`, `hhs = <h_m|h_m>`, `rrs = <r|r>` (noise-weighted, summed over detectors)
- **ML amplitude:** `A_hat = Re{hrs} / hhs`
- **Amplitude uncertainty:** `A_sigma = 1 / sqrt(hhs)`
- **`log_weight`** = log likelihood ratio (memory-marginalized vs no-memory): `0.5 * A_hat * hrs - 0.5 * log(2π * hhs)`. Used as importance weight to reweight original posterior samples to include memory effects.
- **`log_likelihood`** = full amplitude-marginalized log-likelihood: `-0.5 * rrs + log_weight`

### Injection draw prior Jacobian correction (`data.py:read_injection_file`)

The injection file field `lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z` stores the draw prior density in **Cartesian spin coordinates**. The population model uses spherical spin parameters (`a`, `cos_tilt`) or aligned-spin (`chi_z`). A Jacobian correction of `2*log(a_1) + 2*log(a_2)` is needed to convert from the Cartesian density (which includes a `1/a²` factor per spin) to the spherical/marginalized density used by the model.

This correction applies to both the `use_tilts=True` and `use_tilts=False` branches. For `use_tilts=False`, an additional `AlignedSpin` prior factor converts from the full spherical draw prior to the marginal draw prior over `chi_z = a*cos(tilt)`.

**Sources consulted:**
- O1+O2+O3 Search Sensitivity Estimates (Zenodo 5636816): *"sampling PDFs are computed in terms of the variates recorded in the summary files"* and *"We define the injected spin distribution over Cartesian spin components."*
- GWTC-4.0 Cumulative Search Sensitivity Estimates (Zenodo 16740128): provides both Cartesian and polar spin versions of the same injection files, confirming they are *"completely equivalent and only differ in the spin variables over which the draw probabilities are defined."*
- O3 Search Sensitivity Estimates (Zenodo 5546676)

### Key patterns
- HDF5 parameter key lookup uses multiple naming conventions with fallbacks
- bilby config parsing handles several alias patterns for the same setting
- JAX platform/device setup happens at module level in `run_hierarchical_analysis.py` before other imports
- The `data/NRSur7dq4.h5` symlink must exist for surrogate evaluation to work
- The `memory` package is installed in dev mode via `uv sync` (hatchling build-system in `pyproject.toml`)
