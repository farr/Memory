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
uv run --with "jax numpyro arviz corner" python scripts/run_hierarchical_analysis.py \
    dchi_2 "path/to/posteriors/*.h5" \
    --injection-file path/to/injections.hdf \
    --model both --n-warmup 1000 --n-sample 1000 --n-chains 4 \
    --outdir results/
```
Key args: `--model {joint,tgr,both}`, `--scale-tgr`, `--use-tilts`, `--no-plots`, `--force`.

Note: all dependencies including `jax`, `numpyro`, `arviz`, and `corner` are declared in `pyproject.toml`.

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

### `utilities/` package

**`utilities/gw_residuals.py`** — Low-level GW data handling and residual computation.
- `AnalysisConfig` dataclass for config management
- Config parsing with multi-convention fallback (`_get_cfg`, `_cfg_first_match`)
- GWOSC strain downloading, bilby WaveformGenerator/InterferometerList setup
- Spline calibration attachment with flexible key matching
- `compute_one_sample_fd()`: main entry point for frequency-domain residual computation

**`utilities/gw_memory.py`** — Surrogate waveform evaluation, memory physics, and detector projection.
- `evaluate_surrogate_with_LAL()`: evaluate NRSur7dq4 (or other approximants) via LALSimulation for SH modes
- `compute_memory_correction()`: GW memory correction via Wigner 3j angular integrals
- `compute_memory_and_map_to_polarizations()`: memory correction mapped to h+/hx polarizations
- `make_memories()`: end-to-end memory computation from a residual result object
- `polarizations_to_FD()`: FFT polarizations to frequency domain with roll-on window
- `project_to_detectors()`: project polarizations onto detector network
- `compute_memory_variables_likelihoods_and_weights()`: compute memory SNRs, likelihoods, and Bayes factors

**`utilities/kde_contour.py`** — Bounded 1D/2D KDE implementations for posterior visualization.

### Scripts

**`scripts/run_hierarchical_analysis.py`** — Population-level MCMC inference orchestrator.
- Two model types: `make_tgr_only_model()` (2-param mu/sigma TGR) and `make_joint_model()` (full population + TGR)
- KDE-based likelihood from posterior samples
- numpyro NUTS sampler with multi-chain/multi-device support
- Outputs: NetCDF (`.nc`), CSV (`.dat`), corner plots (PNG)

**`scripts/compute_gw_memory_for_GWTC.py`** — Catalog-level GW memory computation for GWTC events.
- Computes memory waveforms, detector projections, and likelihoods across a catalog
- Uses multiprocessing for parallel event processing
- Outputs per-event `{output_dir}/{event_name}/memory_results.h5` with datasets: `A_hat` (ML amplitude), `A_sigma` (posterior std), `A_sample` (posterior draws), `log_weight`, `log_likelihood`, grouped by waveform label
- Currently only the surrogate waveform path is working

### Data flow
- **TGR population analysis:** Event posteriors (HDF5) → KDE smoothing → numpyro hierarchical model → NUTS MCMC → ArviZ posterior → NetCDF + plots
- **Memory computation:** Event posteriors → surrogate waveform → memory correction → detector projection → per-event `memory_results.h5` (amplitude posteriors, likelihoods, weights)

### Key patterns
- HDF5 parameter key lookup uses multiple naming conventions with fallbacks
- bilby config parsing handles several alias patterns for the same setting
- JAX platform/device setup happens at module level in `run_hierarchical_analysis.py` before other imports
- The `data/NRSur7dq4.h5` symlink must exist for surrogate evaluation to work
- Scripts add the project root to `sys.path` and import from `utilities/` directly
