# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian hierarchical analysis pipeline for testing general relativity (TGR) using gravitational wave memory effects in binary mergers. Uses posterior samples from GW parameter estimation runs, builds population-level models with numpyro/JAX, and performs MCMC inference.

**Python 3.11 only** (pinned in `.python-version`). Package manager is **uv**.

## Key Reference Papers

- **GWTC-4 populations paper**: arXiv:2508.18083 — "GWTC-4.0: Population Properties
  of Merging Compact Binaries" (LVK, 2025).  The Broken Power Law + 2 Peaks BBH
  mass model used here follows Appendix B.3 of this paper.  The `PRIOR` dict in
  `memory/hierarchical/models.py` is calibrated against Table 6 of this paper.
  `mmin = 3 M_sun` comes from the triangular prior lower bound (eq. B23).

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
    "path/to/posteriors/*.h5" \
    --injection-file path/to/injections.hdf \
    --analyze astro memory joint \
    --n-warmup 1000 --n-sample 1000 --n-chains 4 \
    --outdir results/
```
Key args: `--analyze {astro,memory,joint}` (default: all three), `--memory-dir`, `--scale-tgr`, `--use-tilts`, `--no-plots`, `--force`.

- `astro` — astrophysical population only; no `--memory-dir` required
- `memory` — TGR hyperparameters only; requires `--memory-dir`
- `joint` — astrophysical + TGR jointly; requires `--memory-dir`

### End-to-end smoke test
```bash
./tests/get_test_data.sh --num-events 2          # download small dataset from LIGO servers
./tests/test_run_analysis_e2e.sh --num-events 2  # run astro-only smoke test (default)
# With memory results available:
./tests/test_run_analysis_e2e.sh --analyze "memory joint" --memory-dir /path/to/memory
```
Results go to `data/test_e2e/results_astro/`. The test script auto-downloads data if missing.
`--analyze` accepts one or more of `astro`, `memory`, `joint`; `memory` and `joint` require `--memory-dir`.

### Production runs (cluster / SLURM + disBatch)

Production analyses run on the cluster using 4 GPUs via SLURM and disBatch.
disBatch is a task-parallel scheduler that reads a file of shell commands and
dispatches them across SLURM-allocated resources.

```bash
module load disBatch          # must be loaded before submitting
bash submit.sh taskfiles/TaskFileMemory_astro
```

`submit.sh` calls:
```bash
sbatch -p gpu -n 1 --cpus-per-task=16 --gpus-per-task=4 --gpu-bind=closest -t 0-6 disBatch <taskfile>
```

Task files live in `taskfiles/`.  Each file contains one shell command per
line; the astro-only run uses `taskfiles/TaskFileMemory_astro`.  Always set
`TGRPOP_DEVICE_COUNT=4` inside the task command to use all 4 allocated GPUs.

**Never run production analyses locally** — always submit via `submit.sh`.

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

**`scripts/run_hierarchical_analysis.py`** — CLI for the hierarchical population analysis. Imports functions from `memory.hierarchical`, handles argument parsing, file I/O, and MCMC orchestration.
- `main()`: parses arguments, loads event posteriors from HDF5, runs one or more of the three analysis modes (`astro`, `memory`, `joint`) via NUTS, saves results as NetCDF/CSV, optionally creates plots, and prints summary statistics with R-hat and ESS
- Outputs: `result_{astro,joint,memory}.nc`, `fit_{astro,joint,memory}_samples.dat`, corner plots (PNG), provenance text files

**`scripts/plot_ppd.py`** — Post-processing script that plots 1D Population Predictive Distributions from one or more NetCDF result files.
- Panels: primary mass `m1`, mass ratio `q`, spin magnitude `a`
- With `--injection-file` and `--n-obs`: converts `m1` panel to differential merger rate `dR/dm1 [Gpc⁻³ yr⁻¹ M☉⁻¹]` via importance sampling over found injections
- Output: `ppd.png`

**`scripts/compute_gw_memory_for_GWTC.py`** — Catalog-level GW memory computation for GWTC events.
- Computes memory waveforms, detector projections, and likelihoods across a catalog
- Uses multiprocessing for parallel sample processing (`--multiprocess_samples`)
- Outputs per-event `{output_dir}/{event_name}/memory_results.h5` with datasets: `A_hat` (ML amplitude), `A_sigma` (posterior std), `A_sample` (posterior draws), `log_weight`, `log_likelihood`, grouped by waveform label
- Only the surrogate (TD modes) waveform path is fully working; FD-only and ROM approximants fail at the SH-mode step
- Validation run (10 samples, all 176 events): 156 fully complete, 20 partial, 0 total failures
  - 13 SEOBNRv4PHM failures: ISCO limit < f_ref (~9–10 Hz vs 20 Hz) for high-mass BBH; bilby conditioning enforces fstart = max(fmin, fref) = fref so lowering minimum_frequency does not help
  - 7 NRTidal/NSBH model failures: no SH mode support (known unfixable)
  - 1 SpinTaylor failure (GW230704): fRef = fmin edge case in IMRPhenomX PN angles

### Data flow
- **TGR population analysis:** Event posteriors (HDF5) → KDE smoothing → numpyro hierarchical model → NUTS MCMC → ArviZ posterior → NetCDF + plots
- **Memory computation:** Event posteriors → surrogate waveform → memory correction → detector projection → per-event `memory_results.h5` (amplitude posteriors, likelihoods, weights)

### Memory likelihood math (Farr et al., `farr_ms.pdf`)

The reference paper is `farr_ms.pdf` in the repo root.

#### Per-sample memory quantities (Eqs. 6–9)

The memory signal model adds a memory waveform `h_m(θ)` scaled by amplitude `A_m` to the residual `r(θ) = d − R(θ)h(θ)`. GR predicts `A_m = 1`.

Marginalising `A_m` with a flat prior over a Gaussian likelihood (Eq. 5) gives (Eq. 8):

```
log L̃(θ) = −½<r|r> + ½ Â Re<h̃_m|r> − ½ log(2π<h̃_m|h̃_m>) + C
```

Key quantities per posterior sample (computed in `compute_memory_variables_likelihoods_and_weights`):
- **Inner products:** `hrs = <h_m|r>`, `hhs = <h_m|h_m>`, `rrs = <r|r>` (noise-weighted, summed over detectors). bilby returns these as **complex** numbers (`4/T · Σ conj(a)·b/PSD`); only the real parts carry physical meaning.
- **ML amplitude (Eq. 6):** `A_hat = Re(hrs) / hhs`
- **Amplitude uncertainty (Eq. 7):** `A_sigma = 1 / sqrt(hhs)`
- **`log_weight` (Eq. 9):** log of the memory-to-no-memory likelihood ratio, used to importance-weight PE samples:
  ```
  log_weight.real = ½ A_hat · Re(hrs) − ½ log(2π · hhs)
                  = ½ (A_hat/A_sigma)² + log(A_sigma) − ½ log(2π)
  ```
  Stored as complex128 in HDF5; `load_memory_data` takes `.real`. Verified analytically: `log_weight.real − ½(A_hat/A_sigma)² − log(A_sigma) = −½log(2π)` exactly for all samples.
- **`log_likelihood`:** full amplitude-marginalized log-likelihood: `−½ rrs + log_weight` (Eq. 8).

#### Hierarchical population analysis (Eqs. 10–15)

The goal is to infer population hyperparameters Λ = (Λ_θ, μ_A, σ_A) from a catalog.

Starting from per-event PE samples θ_i ~ p(θ|d) (original no-memory posterior), and the conditional `p(A|θ_i) = N(A | A_hat_i, A_sigma_i)` (Eq. 10), the hierarchical integral (Eq. 11) reduces to (Eq. 15):

```
I ≈ (1/N) Σ_i  [p(θ_i | Λ_θ) / W(θ_i)]  · N(A_hat_i | μ_A, sqrt(A_sigma_i² + σ_A²))
```

where `W(θ)` is the PE sampling prior and the integral over `A` has been performed analytically.

**The `log_weight` importance weights (Eq. 9) are applied in `generate_data` and `generate_tgr_only_data` when resampling PE samples**, to reweight from the original no-memory posterior to the memory-marginalised posterior (Eq. 9). The Farr et al. paper notes this correction is small when the memory SNR is low per event.

#### Practical limitations with current data

Memory signals are DC/step-function-like, concentrated well below 10 Hz where detector noise is large. Consequently:
- `hhs = <h_m|h_m>` is very small → `A_sigma = 1/sqrt(hhs)` is large (O(10)–O(300) observed)
- `A_hat = Re(hrs)/hhs` is poorly constrained and noise-dominated → values up to O(10,000) observed for most events
- `log_weight ∝ (A_hat/A_sigma)²` can reach thousands → ESS after memory reweighting collapses to ~1 for most events
- Only events where the memory template happens to have low noise-weighted cross-correlation (e.g., GW230924_124453) contribute meaningfully to the hierarchical analysis
- This is a fundamental observational limitation, not a code error — the formulas have been verified to be consistent with the paper

### Injection draw prior Jacobian correction (`data.py:read_injection_file`)

The injection file field `lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z` stores the draw prior density in **Cartesian spin coordinates**. The population model uses spherical spin parameters (`a`, `cos_tilt`). A Jacobian correction of `2*log(a_1) + 2*log(a_2)` is needed to convert from the Cartesian density (which includes a `1/a²` factor per spin) to the spherical density used by the model.

**Sources consulted:**
- O1+O2+O3 Search Sensitivity Estimates (Zenodo 5636816): *"sampling PDFs are computed in terms of the variates recorded in the summary files"* and *"We define the injected spin distribution over Cartesian spin components."*
- GWTC-4.0 Cumulative Search Sensitivity Estimates (Zenodo 16740128): provides both Cartesian and polar spin versions of the same injection files, confirming they are *"completely equivalent and only differ in the spin variables over which the draw probabilities are defined."*
- O3 Search Sensitivity Estimates (Zenodo 5546676)

### Waveform starting-frequency handling (`gw_residuals.py` + `gw_memory.py`)

The bilby residuals path (`compute_one_sample_fd`) and the memory waveform path
(`evaluate_surrogate_with_LAL`) must use the **same starting frequency** for each
posterior sample.  Several mechanisms co-operate to achieve this:

#### ISCO frequency retry
SEOBNRv4PHM (and related SEOB models) raise "Initial frequency is too high, the
limit is X Hz" when `minimum_frequency` exceeds the spin-dependent ISCO for a
given sample's mass/spin parameters.  This error text is written to **C-level
stderr** (fd 2), not to the Python exception string — bilby and the LAL SWIG
bindings surface only `"Internal function call failed: Input domain error"`.

Fix: `_CaptureCStderr` context manager (in `gw_residuals.py`) redirects fd 2 to
a pipe around each LAL/bilby call, drains and re-emits the captured text, then
searches both the Python exception and the C stderr for the `"the limit is X"`
pattern.  When found, `minimum_frequency` is lowered to `0.99 × limit` and the
call is retried.  This loop repeats (up to 20 times) until either the waveform
starts successfully, the new limit is no lower than the current f_min (no
progress), or f_min falls below 1 Hz.

The same `_CaptureCStderr`-based retry loop is used in `evaluate_surrogate_with_LAL`
for direct `SimInspiralChooseTDModes` calls.

#### Per-sample f_min synchronisation
`minimum_frequency` starts as `config.minimum_frequency[ifo]` (e.g. 20 Hz).
After each sample's ISCO retry resolves to a lower value, the effective f_min is
written back to the sample dict (`s["minimum_frequency"] = eff_fmin`).
`evaluate_surrogate_with_LAL` reads `sample.get("minimum_frequency", config.minimum_frequency[...])`,
so the memory waveform starts at exactly the same frequency the bilby residuals
used — no independent retry needed.

Before each sample, the main loop in `compute_gw_residuals_for_BBH_posterior`
resets the shared waveform generator's `minimum_frequency` (and `lmax_nyquist`)
to their original values, preventing state from one sample's retry from bleeding
into the next.

#### `lmax_nyquist` retry (SEOBNRv5PHM / pyseobnr)
pyseobnr raises "ringdown frequency of (N,N) mode greater than maximum frequency
from Nyquist theorem" for high-ℓ modes.  The bilby path retries with
`lmax_nyquist=2` (check only ℓ=2), then `lmax_nyquist=1` (disable entirely,
used by PE for low-mass NSBH events like GW230529).  The value is read from
`waveform_arguments_dict` in the PE config if present.

#### `waveform_arguments_dict` propagation
Extra PE waveform flags (e.g. `{'lmax_nyquist': 1}`, SpinTaylor PN flags
`PhenomXPrecVersion=320`) are read from the `[config]` section and merged into
the bilby `WaveformGenerator.waveform_arguments`.  For the LAL path, a
`_build_lal_dict_from_waveform_args` helper maps dict keys to
`SimInspiralWaveformParamsInsert{Key}` functions to build a LALDict.

### Key patterns
- HDF5 parameter key lookup uses multiple naming conventions with fallbacks
- bilby config parsing handles several alias patterns for the same setting
- JAX platform/device setup happens at module level in `run_hierarchical_analysis.py` before other imports
- The `data/NRSur7dq4.h5` symlink must exist for surrogate evaluation to work
- The `memory` package is installed in dev mode via `uv sync` (hatchling build-system in `pyproject.toml`)
