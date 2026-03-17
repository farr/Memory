Bayesian hierarchical analysis pipeline for testing general relativity (TGR) using gravitational wave memory effects in binary mergers. Uses posterior samples from GW parameter estimation runs, builds population-level models with numpyro/JAX, and performs MCMC inference.

## Setup and Commands

### Install dependencies

```bash
uv sync
```

Python 3.11 is required; the version is pinned in `.python-version`.

### Download waveform data assets

```bash
cd data && ./download.sh
```

This fetches `NRSur7dq4_v1.0.h5` (surrogate model) and
`posterior_samples_NRSur7dq4.h5` from Zenodo, then creates the
`NRSur7dq4.h5` symlink expected by the waveform code.

### Run the main analysis

```bash
uv run python scripts/run_hierarchical_analysis.py \
    "path/to/posteriors/*.h5" \
    --injection-file path/to/injections.hdf \
    --analyze astro memory joint \
    --n-warmup 1000 --n-sample 1000 --n-chains 4 \
    --outdir results/
```

Key args: `--analyze {astro,memory,joint}` (default: all three),
`--memory-dir`, `--scale-tgr`, `--use-tilts`, `--no-plots`, `--force`.

- `astro` — astrophysical population only; no `--memory-dir` required
- `memory` — TGR hyperparameters only; requires `--memory-dir`
- `joint` — astrophysical + TGR jointly; requires `--memory-dir`

### End-to-end smoke test

```bash
./tests/get_test_data.sh --num-events 2
./tests/test_run_analysis_e2e.sh --num-events 2

# With memory results available:
./tests/test_run_analysis_e2e.sh --analyze "memory joint" --memory-dir /path/to/memory
```

Results go to `data/test_e2e/results_astro/`. The test script auto-downloads
data if missing.

### Production runs (cluster / SLURM + disBatch)

Production analyses run on the cluster using 4 GPUs via SLURM and disBatch.
disBatch is a task-parallel scheduler that reads a file of shell commands and
dispatches them across SLURM-allocated resources.

```bash
module load disBatch
bash submit.sh taskfiles/TaskFileMemory_astro
```

`submit.sh` calls:

```bash
sbatch -p gpu -n 1 --cpus-per-task=16 --gpus-per-task=4 --gpu-bind=closest -t 0-6 disBatch <taskfile>
```

Task files live in `taskfiles/`. Each file contains one shell command per line;
the astro-only run uses `taskfiles/TaskFileMemory_astro`. Always set
`TGRPOP_DEVICE_COUNT=4` inside the task command to use all 4 allocated GPUs.

Never run production analyses locally; always submit them via `submit.sh`.

### Environment variables

- `TGRPOP_PLATFORM`: `cpu` or `gpu` (auto-detected; test scripts default to `cpu`)
- `TGRPOP_DEVICE_COUNT`: JAX device parallelization count
- `OMP_NUM_THREADS`: thread count (defaults to 1 in test scripts)
- `REMOTE_USER`, `REMOTE_HOST`: SSH credentials for `get_test_data.sh`
  (default host: `ldas-grid.ligo.caltech.edu`)

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
  - 13 SEOBNRv4PHM failures (prior to guard-bug fix): ISCO limit ~9–10 Hz; a guard-logic bug in `compute_one_sample_fd` fired the no-progress check before any retry was attempted for events where `f_ref < ISCO limit` (e.g. GW190403: f_ref=5 Hz, ISCO=9.9 Hz → new_fmin=9.8 Hz ≥ curr_fmin=5.0 Hz triggered the guard); fixed by only applying the no-progress guard when `minimum_frequency` is already explicitly set in `waveform_arguments`
  - 7 NRTidal/NSBH model failures: no SH mode support (known unfixable)
  - 1 SpinTaylor failure (GW230704): fRef = fmin edge case in IMRPhenomX PN angles

### Data flow
- **TGR population analysis:** Event posteriors (HDF5) → KDE smoothing → numpyro hierarchical model → NUTS MCMC → ArviZ posterior → NetCDF + plots
- **Memory computation:** Event posteriors → surrogate waveform → memory correction → detector projection → per-event `memory_results.h5` (amplitude posteriors, likelihoods, weights)

## Waveform starting-frequency handling (`gw_residuals.py` + `gw_memory.py`)

The bilby residuals path (`compute_one_sample_fd`) and the memory waveform path
(`evaluate_surrogate_with_LAL`) must use the same starting frequency for each
posterior sample. Several mechanisms co-operate to achieve this.

### ISCO frequency retry

SEOBNRv4PHM (and related SEOB models) raise `"Initial frequency is too high, the
limit is X Hz"` when `minimum_frequency` exceeds the spin-dependent ISCO for a
given sample's mass/spin parameters. This error text is written to **C-level
stderr** (fd 2), not to the Python exception string; bilby and the LAL SWIG
bindings usually surface only `"Internal function call failed: Input domain error"`.

`_CaptureCStderr` in `memory/gw_residuals.py` redirects fd 2 to a pipe around
each LAL/bilby call, drains and re-emits the captured text, then searches both
the Python exception and the captured stderr for the `"the limit is X"` pattern.
When found, `minimum_frequency` is lowered to `0.99 * limit` and the call is
retried. This loop repeats until either the waveform starts successfully,
`f_min` falls below 1 Hz, or no progress can be made.

The no-progress check only compares against
`waveform_arguments["minimum_frequency"]` when that key is already explicitly
set by a prior retry. On the first ISCO failure, `minimum_frequency` is not yet
in `waveform_arguments`, so the guard is skipped and the retry must proceed.
This matters for events where `f_ref < ISCO limit` (for example GW190403:
`f_ref = 5 Hz`, `ISCO ~= 9.9 Hz`): using `f_ref` as the fallback `curr_fmin`
caused the guard to fire immediately since `9.8 Hz >= 5 Hz`, preventing any
retry.

The same `_CaptureCStderr`-based retry loop is used in
`evaluate_surrogate_with_LAL()` for direct `SimInspiralChooseTDModes` calls.

### Per-sample `f_min` synchronisation

`minimum_frequency` starts as `config.minimum_frequency[ifo]` (for example
20 Hz). After each sample's ISCO retry resolves to a lower value, the effective
`f_min` is written back to the sample dict (`s["minimum_frequency"] = eff_fmin`).
`evaluate_surrogate_with_LAL()` reads
`sample.get("minimum_frequency", config.minimum_frequency[...])`, so the memory
waveform starts at exactly the same frequency the bilby residuals used.

Before each sample, the main loop in `compute_gw_residuals_for_BBH_posterior`
resets the shared waveform generator's `minimum_frequency` and
`lmax_nyquist` to their original values, preventing state from one sample's
retry from bleeding into the next.

### `lmax_nyquist` retry (SEOBNRv5PHM / pyseobnr)

pyseobnr raises `"ringdown frequency of (N,N) mode greater than maximum
frequency from Nyquist theorem"` for high-ell modes. The bilby path retries
with `lmax_nyquist=2` (check only ell = 2), then `lmax_nyquist=1` (disable the
check entirely; this is needed for some low-mass NSBH events such as GW230529).
The initial value is read from `waveform_arguments_dict` in the PE config when
present.

### `waveform_arguments_dict` propagation

Extra PE waveform flags (for example `{"lmax_nyquist": 1}` or SpinTaylor PN
flags such as `PhenomXPrecVersion=320`) are read from the `[config]` section
and merged into the bilby `WaveformGenerator.waveform_arguments`. For the LAL
path, `_build_lal_dict_from_waveform_args` maps dict keys to
`SimInspiralWaveformParamsInsert{Key}` functions to build a `LALDict`.

## Memory likelihood math (Farr et al., `farr_ms.pdf`)

The reference paper is `farr_ms.pdf` in the repo root.

### Per-sample memory quantities (Eqs. 6–9)

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

### Hierarchical population analysis (Eqs. 10–15)

The goal is to infer population hyperparameters Λ = (Λ_θ, μ_A, σ_A) from a catalog.

Starting from per-event PE samples θ_i ~ p(θ|d) (original no-memory posterior), and the conditional `p(A|θ_i) = N(A | A_hat_i, A_sigma_i)` (Eq. 10), the hierarchical integral (Eq. 11) reduces to (Eq. 15):

```
I ≈ (1/N) Σ_i  [p(θ_i | Λ_θ) / W(θ_i)]  · N(A_hat_i | μ_A, sqrt(A_sigma_i² + σ_A²))
```

where `W(θ)` is the PE sampling prior and the integral over `A` has been performed analytically.

**The `log_weight` importance weights (Eq. 9) are applied in `generate_data` and `generate_tgr_only_data` when resampling PE samples**, to reweight from the original no-memory posterior to the memory-marginalised posterior (Eq. 9). The Farr et al. paper notes this correction is small when the memory SNR is low per event.

### Practical limitations with current data

Memory signals are DC/step-function-like, concentrated well below 10 Hz where detector noise is large. Consequently:
- `hhs = <h_m|h_m>` is very small → `A_sigma = 1/sqrt(hhs)` is large (O(10)–O(300) observed)
- `A_hat = Re(hrs)/hhs` is poorly constrained and noise-dominated → values up to O(10,000) observed for most events
- `log_weight ∝ (A_hat/A_sigma)²` can reach thousands → ESS after memory reweighting collapses to ~1 for most events
- Only events where the memory template happens to have low noise-weighted cross-correlation (e.g., GW230924_124453) contribute meaningfully to the hierarchical analysis
- This is a fundamental observational limitation, not a code error — the formulas have been verified to be consistent with the paper

## Post-processing: PPD and rate plots

`scripts/plot_ppd.py` visualises the results of a hierarchical run by drawing
1D Population Predictive Distributions (PPDs) — the marginal population
density averaged over the posterior — for primary mass `m1`, mass ratio `q`,
and spin magnitude `a`.

### Basic usage

```bash
# Single run
uv run python scripts/plot_ppd.py results/run/result_astro.nc

# Overlay two runs
uv run python scripts/plot_ppd.py result_astro.nc result_joint.nc \
    --labels "astro only" joint

# Limit to 500 posterior draws (faster)
uv run python scripts/plot_ppd.py result_astro.nc --n-ppd 500
```

Output is `ppd.png` in the same directory as the first input file (override
with `--outdir`).

### Rate mode: dR/dm1 instead of p(m1)

Passing `--injection-file` converts the `m1` panel from the normalised PDF
`p(m1)` to the differential merger rate `dR/dm1 [Gpc^-3 yr^-1 M_sun^-1]`.
`--n-obs` must also be provided (the number of events used in the analysis).

```bash
uv run python scripts/plot_ppd.py result_astro.nc \
    --injection-file /path/to/sensitivity-estimate.hdf \
    --n-obs 43
```

### Rate calculation

The total volumetric merger rate is estimated as

```
R(Λ) = N_obs / (T_obs * β(Λ))
```

where `T_obs` is the live observing time (read from the injection file) and
`β(Λ)` is the **effective surveyed comoving volume** [Gpc^3]:

```
β(Λ) = (1/N_draw) * Σ_{found}  p_pop(θ | Λ) / p_draw(θ)
```

`N_draw` is the total number of injections attempted (found + missed), and the
sum runs over found injections only.  The population model density is

```
p_pop(θ | Λ) = p(m1) * p(q|m1) * p(z) * p(a1) * p(a2)
```

where `p(z) ∝ (1+z)^λ * dVc/dz/(1+z)` is left **unnormalised**: the integral
over redshift gives a volume in Gpc^3, so `β` has units Gpc^3 and `R` has
units Gpc^-3 yr^-1.

The differential rate is evaluated at z = 0.2 (following the LVK populations
paper convention):

```
R(z=0.2 | Λ) = R(Λ) * (1 + 0.2)^lambda
dR/dm1(z=0.2) = R(z=0.2 | Λ) * p(m1 | Λ)
```

No `dz/dVc` Jacobian is needed: `β` already has units of Gpc³ (comoving
volume), so `R` is already per unit comoving volume at z=0.  The
`(1+z)^lambda` factor shifts the evaluation to z=0.2.

Both `R(z=0)` and `R(z=0.2)` are printed to stdout; the plot y-axis shows
the rate at z=0.2.

#### Draw prior Jacobian

The injection file records the draw prior as a log-density in Cartesian spin
coordinates, `lnpdraw(m1, m2, z, sx, sy, sz)`.  Converting to `(m1, q, z, a1,
a2)` requires two Jacobian factors:

| Change of variables | Jacobian factor |
|---|---|
| `m2 → q = m2/m1` | `m1` |
| `(sx, sy, sz) → a` (isotropic direction integrated out) | `4π a²` per spin |

Including the per-injection sensitivity weight `w` (network duty cycle):

```
log p_draw(m1, q, z, a1, a2) = lnpdraw
    + log(m1)
    + log(4π a1²)
    + log(4π a2²)
    + log(w)
```

## Hierarchical population model

The pipeline provides two numpyro models (selected via `--model {joint,tgr,both}`):

- **Joint model** (`make_joint_model`): fits astrophysical population parameters
  and TGR parameters simultaneously.
- **TGR-only model** (`make_tgr_only_model`): fits only the TGR
  hyperparameters, ignoring astrophysical population structure.

Both are defined in `memory/hierarchical/models.py` and sampled with NUTS via
numpyro. The per-event likelihood uses importance-weighted KDE-smoothed sums
over posterior samples from individual-event parameter estimation.

### Joint model components

The full joint population density factorises as:

```
p(m1, q, z, a1, a2, cos_t1, cos_t2, A) =
    p(m1) * p(q | m1) * p(z) * p(a1, a2) * [p(cos_t1, cos_t2)] * [p(A)]
```

where brackets denote optional components (spin tilts via `--use-tilts`, TGR
amplitude via `--model joint` or `--model both`).

#### Primary mass: broken power law + two Gaussian peaks

Following the standard LVK parameterisation (cf. gwpopulation / GWTC-3).

The power law has two slopes separated at a break mass:

```
m_break = mmin + b * (mmax - mmin)
```

where `mmin = 3`, `mmax = 100`, and `b` is the break fraction. Each segment
is a normalised power law `PL(m; a, lo, hi)`:

```
PL(m; a, lo, hi) = (1-a) * m^{-a} / (hi^{1-a} - lo^{1-a})     (a != 1)
                 = m^{-1} / ln(hi/lo)                            (a = 1)
```

A continuity correction `C = PL_high(m_break) / PL_low(m_break)` ensures the
density is continuous at the break:

```
BPL(m) = [ C * PL(m; a1, mmin, m_break) * I(m < m_break)
         +     PL(m; a2, m_break, mmax) * I(m >= m_break) ] / (1 + C)
```

Two Gaussian components capture features not well described by a power law.
Their mean priors are non-overlapping to reduce degeneracies:

- **Peak 1** (low-mass): `mu_peak_1 ~ U(3, 15)`
- **Peak 2** (high-mass): `mu_peak_2 ~ U(20, 50)`

Full mixture:

```
p(m1) = (1 - frac_peak_1 - frac_peak_2) * BPL(m1)
      +  frac_peak_1 * N(m1 | mu_peak_1, sigma_peak_1)
      +  frac_peak_2 * N(m1 | mu_peak_2, sigma_peak_2)
```

with the constraint `frac_peak_1 + frac_peak_2 < 1`.

#### Mass ratio: power law

The mass ratio `q = m2/m1` follows a normalised power law on
`[mmin/m1, 1]`:

```
p(q | m1) ~ q^beta,    q >= mmin / m1
```

#### Redshift: power law in (1+z)

The merger rate density evolves as a power law in `(1+z)` weighted by the
differential comoving volume per unit source time:

```
p(z) ~ (1+z)^lamb * dVc/dz / (1+z)
```

where `dVc/dz` is computed from a Planck15-like cosmology (H0=67.9,
Om0=0.3065) and interpolated on a grid up to `z = 2.5`.

#### Spin magnitudes

The spin magnitudes `(a1, a2)` are modelled as a bivariate normal
distribution with shared mean `mu_spin` and shared variance `sigma_spin^2`,
convolved with per-event KDE bandwidth matrices to account for measurement
uncertainty:

```
p(a1, a2) = MVN([mu_spin, mu_spin], BW + sigma_spin^2 * I)
```

#### Spin tilts (optional, `--use-tilts`)

When enabled, the cosine tilt angles `(cos_t1, cos_t2)` follow a mixture of
an isotropic component (uniform on `[-1, 1]^2`) and a Gaussian component
peaked at aligned spins `(cos_t = 1)`:

```
p(cos_t1, cos_t2) = f_iso / 4
    + (1 - f_iso) * N2d((cos_t1, cos_t2) | (1, 1), sigma_tilt^2 * I)
```

#### TGR deviation parameter (optional)

The population distribution for the TGR memory amplitude `A` is Gaussian with
hyperparameters `mu_tgr` and `sigma_tgr`. GR predicts `mu_tgr = 0` and
`sigma_tgr = 0` (i.e., no deviation from the GR memory prediction). The
per-sample measurement uncertainty `A_sigma` is convolved analytically:

```
p(A_hat | mu_tgr, sigma_tgr) = N(A_hat | mu_tgr, sqrt(A_sigma^2 + sigma_tgr^2))
```

Prior widths on `mu_tgr` and `sigma_tgr` are auto-scaled from the data (1.5x
the maximum absolute `A_hat`), or set manually via `--mu-tgr-scale` and
`--sigma-tgr-scale`.

### Selection effects

The joint model corrects for selection bias using an injection set. The
expected detection fraction is estimated by importance-reweighting simulated
injections:

```
<det> = (1/Ndraw) * sum_i p_pop(theta_i) / p_draw(theta_i)
```

The log-likelihood includes a `-Nobs * log(<det>)` correction term.

### Effective sample size regularization

Two soft N_eff penalties discourage regions of parameter space where the
importance weights become too concentrated:

- **Per-event N_eff**: the minimum across events must exceed `Nobs`.
- **Selection N_eff**: must exceed `4 * Nobs`.

Both use a smooth boundary function that applies a steep penalty
(`-x^10` scaling) when N_eff drops below the threshold.

### TGR-only model

A simpler two-parameter model fitting only `mu_tgr` and `sigma_tgr` directly
to the per-event `(A_hat, A_sigma)` measurements, without modelling any
astrophysical population structure. Uses the same analytic Gaussian
convolution as the joint model's TGR component.

### All hyperparameters and priors

All prior bounds and the `MMIN`/`MMAX` constants are defined in
`memory/hierarchical/models.py` (`PRIOR` dict) and importable from there.

| Parameter       | Prior            | Component       | Description                       |
|-----------------|------------------|-----------------|-----------------------------------|
| `alpha_1`       | U(-4, 12)        | Primary mass    | PL slope below break              |
| `alpha_2`       | U(-4, 12)        | Primary mass    | PL slope above break              |
| `b`             | U(0, 1)          | Primary mass    | Break fraction                    |
| `frac_peak_1`   | Dirichlet(1,1,1) | Primary mass    | Fraction in low-mass Gaussian     |
| `mu_peak_1`     | U(3, 15)         | Primary mass    | Mean of low-mass Gaussian         |
| `sigma_peak_1`  | U(0.5, 8)        | Primary mass    | Std of low-mass Gaussian          |
| `frac_peak_2`   | Dirichlet(1,1,1) | Primary mass    | Fraction in high-mass Gaussian    |
| `mu_peak_2`     | U(15, 75)        | Primary mass    | Mean of high-mass Gaussian        |
| `sigma_peak_2`  | U(0.5, 8)        | Primary mass    | Std of high-mass Gaussian         |
| `beta`          | U(-4, 12)        | Mass ratio      | Power-law slope for q             |
| `lamb`          | U(-30, 30)       | Redshift        | Power-law index on (1+z)          |
| `mu_spin`       | U(0, 0.7)        | Spin magnitudes | Shared mean of (a1, a2)           |
| `sigma_spin`    | U(0.01, 0.5)     | Spin magnitudes | Shared std of (a1, a2)            |
| `f_iso`         | U(0, 1)          | Spin tilts      | Isotropic fraction (if enabled)   |
| `sigma_tilt`    | U(0.05, 10)      | Spin tilts      | Tilt peak width (if enabled)      |
| `mu_tgr`        | U(-s, s)         | TGR             | Population mean of A (auto-scaled)|
| `sigma_tgr`     | U(0, s)          | TGR             | Population std of A (auto-scaled) |