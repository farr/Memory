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

This fetches `NRSur7dq4_v1.0.h5` (surrogate model),
`posterior_samples_NRSur7dq4.h5`, and the default polar-spin selection
files from Zenodo, then creates the `NRSur7dq4.h5` symlink expected by
the waveform code.

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
`--memory-dir`, `--waveform`, `--scale-tgr`, `--no-plots`,
`--force`.

- `astro` — astrophysical population only; no `--memory-dir` required
- `memory` — TGR hyperparameters only; requires `--memory-dir`
- `joint` — astrophysical + TGR jointly; requires `--memory-dir`

### End-to-end smoke test

```bash
./tests/get_test_data.sh --num-events 2
./tests/test_run_analysis_e2e.sh --num-events 2

# With memory results available:
./tests/test_run_analysis_e2e.sh --analyze "memory joint" --memory-dir /path/to/memory --waveform NRSur7dq4
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

### Input PE data conventions

The script accepts multiple glob patterns as positional arguments, so GWTC-3
and GWTC-4 posteriors can be passed together:

```bash
scripts/run_hierarchical_analysis.py \
    "/path/GWTC-4/IGWN-GWTC4p0-*-combined_PEDataRelease.hdf5" \
    "/path/GWTC-3/IGWN-GWTC3p0-v2-GW*_PEDataRelease_mixed_nocosmo.h5" \
    ...
```

**Why `nocosmo` for GWTC-3 (not `cosmo`)?**

The hierarchical model importance-weights PE samples by $p_{\mathrm{pop}}(\theta) / p_{\mathrm{draw}}(\theta)$,
where `p_draw` is read from each file's `log_prior` field.  The correct
computation requires that `log_prior` accurately reflects the distribution the
samples were drawn from.

- **GWTC-4** files were originally sampled with `bilby.gw.prior.UniformSourceFrame`
  (uniform in comoving volume per source time), and `log_prior` correctly records
  that prior.
- **GWTC-3 `cosmo`** files are produced by PESummary rejection-sampling the
  original posteriors (which used `PowerLaw(alpha=2)` in luminosity distance) to
  reweight them toward `UniformSourceFrame`.  Crucially, PESummary does **not**
  update the `log_prior` field — it still records the original `PowerLaw(alpha=2)`
  prior.  Using these samples with their stored `log_prior` therefore introduces a
  spurious per-sample factor of $p_{\mathrm{USF}}(\theta) / p_{\mathrm{PL2}}(\theta) \propto \frac{\mathrm{d}V_c/\mathrm{d}z}{(1+z)\,d_L^2}$
  that biases the redshift weighting for every O3 event.
- **GWTC-3 `nocosmo`** files retain the full original sample set with
  `log_prior` $=$ $\log p_{\mathrm{PL2}}(\theta)$ and samples drawn from $L \cdot p_{\mathrm{PL2}}$ — internally
  consistent.  The hierarchical model divides out $p_{\mathrm{PL2}}$ correctly for O3 events
  and $p_{\mathrm{USF}}$ correctly for O4a events; mixing PE priors across events is fine
  because the `Z_pe` normalisation cancels in importance-sampling ratios.

**Injection file:** use `mixture-real_o3_o4a-*` (not `o4a`-only) whenever O3
events are included, since the selection function must cover the full observing
baseline.

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
  - `make_tgr_only_model()`: numpyro model with two hyperparameters $(\mu_{\mathrm{tgr}}, \sigma_{\mathrm{tgr}})$ describing a Gaussian population distribution for the TGR deviation parameter; likelihood is a KDE-smoothed sum over per-event posterior samples
  - `make_joint_model()`: numpyro model jointly fitting astrophysical population (broken power law + two Gaussian peaks mass function, power-law mass ratio, power-law redshift, spin magnitude/tilt distributions) and TGR hyperparameters; includes selection-effect correction and effective-sample-size regularization

- **`memory/hierarchical/plotting.py`** — Plotting and ArviZ post-processing.
  - `get_samples_df()`: converts an ArviZ InferenceData posterior into a flat pandas DataFrame (chains $\times$ draws)
  - `create_plots()`: generates diagnostic plots — population KDE, hyperparameter pairplot, TGR corner plot, and full joint corner plot; saves PNGs and per-fit CSV sample files

### Scripts

**`scripts/run_hierarchical_analysis.py`** — CLI for the hierarchical population analysis. Imports functions from `memory.hierarchical`, handles argument parsing, file I/O, and MCMC orchestration.
- `main()`: parses arguments, loads event posteriors from HDF5, runs one or more of the three analysis modes (`astro`, `memory`, `joint`) via NUTS, saves results as NetCDF/CSV, optionally creates plots, and prints summary statistics with R-hat and ESS
- Outputs: `result_{astro,joint,memory}.nc`, `fit_{astro,joint,memory}_samples.dat`, corner plots (PNG), provenance text files

**`scripts/plot_ppd.py`** — Post-processing script that plots 1D Population Predictive Distributions from one or more NetCDF result files.
- Panels: primary mass $m_1$, mass ratio $q$, spin magnitude $a$
- With `--injection-file` and `--n-obs`: converts `m1` panel to differential merger rate $\mathrm{d}R/\mathrm{d}m_1$ $[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,M_\odot^{-1}]$ via importance sampling over found injections
- Output: `ppd.png`

**`scripts/compute_gw_memory_for_GWTC.py`** — Catalog-level GW memory computation for GWTC events.
- Computes memory waveforms, detector projections, and likelihoods across a catalog
- Uses multiprocessing for parallel sample processing (`--multiprocess_samples`)
- Outputs per-event `{output_dir}/{event_name}/memory_results.h5` with datasets: `A_hat` (ML amplitude $\hat{A}$), `A_sigma` (posterior std $A_\sigma$), `A_sample` (posterior draws), `log_weight`, `log_likelihood`, grouped by waveform label
- Only the surrogate (TD modes) waveform path is fully working; FD-only and ROM approximants fail at the SH-mode step
- Validation run (10 samples, all 176 events): 165 fully complete, 11 partial, 0 total failures
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
This matters for events where $f_{\mathrm{ref}} < f_{\mathrm{ISCO}}$ (for example GW190403:
$f_{\mathrm{ref}} = 5\,\mathrm{Hz}$, $f_{\mathrm{ISCO}} \approx 9.9\,\mathrm{Hz}$): using $f_{\mathrm{ref}}$ as the fallback $f_{\mathrm{min}}$
caused the guard to fire immediately since $9.8\,\mathrm{Hz} \ge 5.0\,\mathrm{Hz}$, preventing any
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

Before each sample, the main loop in `compute_bbh_residuals_with_spline_calibration`
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

The memory signal model adds a memory waveform $h_m(\theta)$ scaled by amplitude $A_m$ to the residual $r(\theta) = d - R(\theta)\,h(\theta)$. GR predicts $A_m = 1$.

Marginalising $A_m$ with a flat prior over a Gaussian likelihood (Eq. 5) gives (Eq. 8):

$$
\log \tilde{L}(\theta) = -\tfrac{1}{2}\langle r \vert r \rangle + \tfrac{1}{2}\hat{A}\,\Re\langle \tilde{h}_m \vert r \rangle - \tfrac{1}{2}\log\bigl(2\pi\langle \tilde{h}_m \vert \tilde{h}_m \rangle\bigr) + C
$$

Key quantities per posterior sample (computed in `compute_memory_variables_likelihoods_and_weights`):
- **Inner products:** $\mathit{hrs} = \langle h_m\vert r\rangle$, $\mathit{hhs} = \langle h_m\vert h_m\rangle$, $\mathit{rrs} = \langle r\vert r\rangle$ (noise-weighted, summed over detectors). bilby returns these as **complex** numbers ($\frac{4}{T}\sum \overline{a}\cdot b/\mathrm{PSD}$); only the real parts carry physical meaning.
- **ML amplitude (Eq. 6):** $\hat{A} = \Re(\mathit{hrs}) / \mathit{hhs}$
- **Amplitude uncertainty (Eq. 7):** $A_\sigma = 1/\sqrt{\mathit{hhs}}$
- **`log_weight` (Eq. 9):** log of the memory-to-no-memory likelihood ratio, used to importance-weight PE samples:
  $$
  \begin{aligned}
  \log\mathit{w}_{\mathrm{real}} &= \tfrac{1}{2}\hat{A}\,\Re(\mathit{hrs}) - \tfrac{1}{2}\log(2\pi\cdot \mathit{hhs}) \\
  &= \tfrac{1}{2}(\hat{A}/A_\sigma)^2 + \log(A_\sigma) - \tfrac{1}{2}\log(2\pi)
  \end{aligned}
  $$
  Stored as complex128 in HDF5; `load_memory_data` takes `.real`. Verified analytically: $\log\mathit{w}_{\mathrm{real}} - \tfrac{1}{2}(\hat{A}/A_\sigma)^2 - \log(A_\sigma) = -\tfrac{1}{2}\log(2\pi)$ exactly for all samples.
- **`log_likelihood`:** full amplitude-marginalized log-likelihood: $-\tfrac{1}{2}\,\mathit{rrs} + \log\mathit{w}$ (Eq. 8).

### Hierarchical population analysis (Eqs. 10–15)

The goal is to infer population hyperparameters $\Lambda = (\Lambda_\theta, \mu_A, \sigma_A)$ from a catalog.

Starting from per-event PE samples $\theta_i \sim p(\theta\mid d)$ (original no-memory posterior), and the conditional $p(A\mid\theta_i) = \mathcal{N}(A \mid \hat{A}_i, A_{\sigma,i})$ (Eq. 10), the hierarchical integral (Eq. 11) reduces to (Eq. 15):

$$
I \approx \frac{1}{N}\sum_i \left[\frac{p(\theta_i \mid \Lambda_\theta)}{W(\theta_i)}\right] \cdot \mathcal{N}\bigl(\hat{A}_i \mid \mu_A, \sqrt{A_{\sigma,i}^2 + \sigma_A^2}\bigr)
$$

where $W(\theta)$ is the PE sampling prior and the integral over $A$ has been performed analytically.

**The `log_weight` importance weights (Eq. 9) are applied in `generate_data` and `generate_tgr_only_data` when resampling PE samples**, to reweight from the original no-memory posterior to the memory-marginalised posterior (Eq. 9). The Farr et al. paper notes this correction is small when the memory SNR is low per event.

## Post-processing: PPD and rate plots

`scripts/plot_ppd.py` visualises the results of a hierarchical run by drawing
1D Population Predictive Distributions (PPDs) — the marginal population
density averaged over the posterior — for primary mass $m_1$, mass ratio $q$,
and spin magnitude $a$.

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

### Rate mode: $\mathrm{d}R/\mathrm{d}m_1$ instead of $p(m_1)$

Passing `--injection-file` converts the `m1` panel from the normalised PDF
$p(m_1)$ to the differential merger rate $\mathrm{d}R/\mathrm{d}m_1$ $[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,M_\odot^{-1}]$.
`--n-obs` must also be provided (the number of events used in the analysis).

```bash
uv run python scripts/plot_ppd.py result_astro.nc \
    --injection-file /path/to/sensitivity-estimate.hdf \
    --n-obs 43
```

### Rate calculation

The total volumetric merger rate is estimated as

$$
R(\Lambda) = \frac{N_{\mathrm{obs}}}{T_{\mathrm{obs}}\,\beta(\Lambda)}
$$

where $T_{\mathrm{obs}}$ is the live observing time (read from the injection file) and
$\beta(\Lambda)$ is the **effective surveyed comoving volume** $[\mathrm{Gpc}^3]$:

$$
\beta(\Lambda) = \frac{1}{N_{\mathrm{draw}}}\sum_{\mathrm{found}} \frac{p_{\mathrm{pop}}(\theta \mid \Lambda)}{p_{\mathrm{draw}}(\theta)}
$$

$N_{\mathrm{draw}}$ is the total number of injections attempted (found + missed), and the
sum runs over found injections only.  In the joint model the population density is

$$
p_{\mathrm{pop}}(\theta \mid \Lambda) =
p(m_1)\,p(q\mid m_1)\,p(z)\,p(a_1,a_2)\,p(c_{t,1},c_{t,2})
$$

where $p(z) \propto (1+z)^{\lambda}\,\frac{\mathrm{d}V_C/\mathrm{d}z}{1+z}$ is left **unnormalised**: the integral
over redshift gives a volume in $\mathrm{Gpc}^3$, so $\beta$ has units $\mathrm{Gpc}^3$ and $R$ has
units $\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$.  The `plot_ppd.py` rate calculation
now includes the same tilt-mixture term used by `make_joint_model`.

The differential rate is evaluated at $z = 0.2$ (following the LVK populations
paper convention):

$$
\begin{aligned}
R(z=0.2 \mid \Lambda) &= R(\Lambda)\,(1+0.2)^{\lambda} \\
\frac{\mathrm{d}R}{\mathrm{d}m_1}(z=0.2) &= R(z=0.2\mid \Lambda)\,p(m_1\mid \Lambda)
\end{aligned}
$$

No $\mathrm{d}z/\mathrm{d}V_C$ Jacobian is needed: $\beta$ already has units of $\mathrm{Gpc}^3$ (comoving
volume), so $R$ is already per unit comoving volume at $z=0$.  The
$(1+z)^{\lambda}$ factor shifts the evaluation to $z=0.2$.

Both $R(z=0)$ and $R(z=0.2)$ are printed to stdout; the plot y-axis shows
the rate at $z=0.2$.

#### Draw prior Jacobian

The injection file records the draw prior as a log-density in Cartesian spin
coordinates, $\mathrm{lnpdraw}(m_1,m_2,z,s_x,s_y,s_z)$.  Converting to
$(m_1,q,z,a_1,a_2,c_{t,1},c_{t,2})$ requires two Jacobian factors:

| Change of variables | Jacobian factor |
|---|---|
| $m_2 \to q = m_2/m_1$ | $m_1$ |
| $(s_x,s_y,s_z) \to (a,c_t)$ (azimuth integrated out) | $2\pi a^2$ per spin |

Including the per-injection sensitivity weight $w$ (network duty cycle):

$$
\begin{aligned}
\log p_{\mathrm{draw}}(m_1,q,z,a_1,a_2,c_{t,1},c_{t,2}) = {}& \mathrm{lnpdraw} + \log(m_1) \\
&+ \log(2\pi a_1^2) + \log(2\pi a_2^2) + \log(w)
\end{aligned}
$$

## Hierarchical population model

The pipeline provides two numpyro models (selected via `--analyze {astro,memory,joint}`):

- **Joint model** (`make_joint_model`): fits astrophysical population parameters
  and TGR parameters simultaneously.
- **TGR-only model** (`make_tgr_only_model`): fits only the TGR
  hyperparameters, ignoring astrophysical population structure.

Both are defined in `memory/hierarchical/models.py` and sampled with NUTS via
numpyro. The per-event likelihood uses importance-weighted KDE-smoothed sums
over posterior samples from individual-event parameter estimation.

### Joint model components

The full joint population density factorises as:

$$
p(m_1,q,z,a_1,a_2,c_{t,1},c_{t,2},A) =
p(m_1)\,p(q\mid m_1)\,p(z)\,p(a_1,a_2)\,[p(c_{t,1},c_{t,2})]\,[p(A)]
$$

where $c_{t,i}$ are cosine tilts, brackets denote optional components (TGR amplitude, included when running
`--analyze memory` or `--analyze joint`). Spin tilts are always included.

#### Primary mass: broken power law + two Gaussian peaks

Following the standard LVK parameterisation (cf. gwpopulation / GWTC-3).

The power law has two slopes separated at a break mass:

$$
m_{\mathrm{break}} = m_{\min} + b\,(m_{\max}-m_{\min})
$$

where $m_{\min} = 3$, $m_{\max} = 300$, and $b$ is the break fraction. The prior on
$b$ is restricted so that $m_{\mathrm{break}}$ lies between 20 and 50 solar masses. Each
segment
is a normalised power law $\mathrm{PL}(m; a, \mathrm{lo}, \mathrm{hi})$:

$$
\mathrm{PL}(m; a, \mathrm{lo}, \mathrm{hi}) =
\begin{cases}
\dfrac{(1-a)\,m^{-a}}{\mathrm{hi}^{1-a}-\mathrm{lo}^{1-a}} & a \neq 1 \\[0.6em]
\dfrac{m^{-1}}{\ln(\mathrm{hi}/\mathrm{lo})} & a = 1
\end{cases}
$$

A continuity correction $C = \mathrm{PL}_{\mathrm{high}}(m_{\mathrm{break}}) / \mathrm{PL}_{\mathrm{low}}(m_{\mathrm{break}})$ ensures the
density is continuous at the break:

$$
\mathrm{BPL}(m) = \frac{
C\,\mathrm{PL}(m; a_1, m_{\min}, m_{\mathrm{break}})\,\mathbf{1}_{\{m < m_{\mathrm{break}}\}}
+ \mathrm{PL}(m; a_2, m_{\mathrm{break}}, m_{\max})\,\mathbf{1}_{\{m \ge m_{\mathrm{break}}\}}
}{1 + C}
$$

Two Gaussian components capture features not well described by a power law.
Their mean priors are non-overlapping to reduce degeneracies:

- **Peak 1** (low-mass): $\mu_{\mathrm{peak},1} \sim \mathcal{U}(5, 20)$
- **Peak 2** (high-mass): $\mu_{\mathrm{peak},2} \sim \mathcal{U}(25, 60)$

Full mixture:

$$
\begin{aligned}
p(m_1) = {}& f_{\mathrm{bpl}}\,\mathrm{BPL}(m_1) \\
&+ f_{\mathrm{peak},1}\,\mathcal{N}(m_1 \mid \mu_{\mathrm{peak},1}, \sigma_{\mathrm{peak},1}) \\
&+ f_{\mathrm{peak},2}\,\mathcal{N}(m_1 \mid \mu_{\mathrm{peak},2}, \sigma_{\mathrm{peak},2})
\end{aligned}
$$

where $(f_{\mathrm{bpl}}, f_{\mathrm{peak},1}, f_{\mathrm{peak},2}) \sim \mathrm{Dirichlet}(1, 1, 1)$.

#### Mass ratio: power law

The mass ratio $q = m_2/m_1$ follows a normalised power law on
$[m_{\min}/m_1,\,1]$:

$$
p(q \mid m_1) \propto q^{\beta},\qquad q \ge m_{\min}/m_1
$$

#### Redshift: power law in (1+z)

The merger rate density evolves as a power law in $(1+z)$ weighted by the
differential comoving volume per unit source time:

$$
p(z) \propto (1+z)^{\lambda}\,\frac{\mathrm{d}V_C/\mathrm{d}z}{1+z}
$$

where $\mathrm{d}V_C/\mathrm{d}z$ is computed from a Planck15-like cosmology (H0=67.9,
Om0=0.3065) and interpolated on a grid up to $z = 2.5$.

#### Spin magnitudes

The spin magnitudes $(a_1,a_2)$ are modelled as a bivariate normal
distribution with shared mean $\mu_{\mathrm{spin}}$ and shared variance $\sigma_{\mathrm{spin}}^2$,
convolved with per-event KDE bandwidth matrices to account for measurement
uncertainty:

$$
p(a_1,a_2) = \mathcal{N}\!\left(
\begin{pmatrix}\mu_{\mathrm{spin}} \\ \mu_{\mathrm{spin}}\end{pmatrix},\;
\mathrm{BW} + \sigma_{\mathrm{spin}}^2 I
\right)
$$

#### Spin tilts (optional, `--use-tilts`)

The cosine tilt angles $(c_{t,1},c_{t,2})$ follow a mixture of
an isotropic component (uniform on $[-1,1]^2$) and a truncated Gaussian
component with a shared population location and scale:

$$
p(c_{t,1}, c_{t,2}) = \frac{f_{\mathrm{iso}}}{4}
+ (1-f_{\mathrm{iso}})\,\mathcal{N}_{\mathrm{TN}}(c_{t,1} \mid \mu_{\mathrm{tilt}}, \sigma_{\mathrm{tilt}})\,
\mathcal{N}_{\mathrm{TN}}(c_{t,2} \mid \mu_{\mathrm{tilt}}, \sigma_{\mathrm{tilt}})
$$

where $\mathcal{N}_{\mathrm{TN}}$ is a Gaussian truncated to $[-1,1]$.

#### TGR deviation parameter (optional)

The population distribution for the TGR memory amplitude $A$ is Gaussian with
hyperparameters $\mu_{\mathrm{tgr}}$ and $\sigma_{\mathrm{tgr}}$. GR predicts $\mu_{\mathrm{tgr}} = 0$ and
$\sigma_{\mathrm{tgr}} = 0$ (i.e., no deviation from the GR memory prediction). The
per-sample measurement uncertainty $A_\sigma$ is convolved analytically:

$$
p(\hat{A} \mid \mu_{\mathrm{tgr}}, \sigma_{\mathrm{tgr}}) = \mathcal{N}\!\left(\hat{A} \mid \mu_{\mathrm{tgr}}, \sqrt{A_\sigma^2 + \sigma_{\mathrm{tgr}}^2}\right)
$$

Prior widths on $\mu_{\mathrm{tgr}}$ and $\sigma_{\mathrm{tgr}}$ are auto-scaled from the data (1.5×
the maximum absolute $\hat{A}$), or set manually via `--mu-tgr-scale` and
`--sigma-tgr-scale`.

### Selection effects

The joint model corrects for selection bias using an injection set. The
expected detection fraction is estimated by importance-reweighting simulated
injections:

$$
\langle \mathrm{det} \rangle = \frac{1}{N_{\mathrm{draw}}}\sum_i \frac{p_{\mathrm{pop}}(\theta_i)}{p_{\mathrm{draw}}(\theta_i)}
$$

The log-likelihood includes a $-N_{\mathrm{obs}}\log\langle \mathrm{det} \rangle$ correction term.

### Effective sample size regularization

Two soft $N_{\mathrm{eff}}$ penalties discourage regions of parameter space where the
importance weights become too concentrated:

- **Per-event $N_{\mathrm{eff}}$**: the minimum across events must exceed `Nobs`.
- **Selection $N_{\mathrm{eff}}$**: must exceed $4 \times N_{\mathrm{obs}}$.

Both use a smooth boundary function that applies a steep penalty
($-x^{10}$ scaling) when $N_{\mathrm{eff}}$ drops below the threshold.

### TGR-only model

A simpler two-parameter model fitting only $\mu_{\mathrm{tgr}}$ and $\sigma_{\mathrm{tgr}}$ directly
to the per-event $(\hat{A}, A_\sigma)$ measurements, without modelling any
astrophysical population structure. Uses the same analytic Gaussian
convolution as the joint model's TGR component.

### All hyperparameters and priors

The uniform prior bounds and the `MMIN`/`MMAX` constants are defined in
`memory/hierarchical/models.py` (`PRIOR` dict); the primary-mass mixture
fractions are sampled separately there via `Dirichlet(1, 1, 1)`.

| Parameter       | Prior            | Component       | Description                       |
|-----------------|------------------|-----------------|-----------------------------------|
| `alpha_1`       | U(-4, 12)        | Primary mass    | PL slope below break              |
| `alpha_2`       | U(-4, 12)        | Primary mass    | PL slope above break              |
| `b`             | U((20-3)/(300-3), (50-3)/(300-3)) | Primary mass | Break fraction, i.e. $m_{\mathrm{break}} \in [20, 50]$ |
| `frac_bpl`      | Dirichlet(1,1,1) | Primary mass    | Broken-power-law mixture weight   |
| `frac_peak_1`   | Dirichlet(1,1,1) | Primary mass    | Fraction in low-mass Gaussian     |
| `mu_peak_1`     | U(5, 20)         | Primary mass    | Mean of low-mass Gaussian         |
| `sigma_peak_1`  | U(0.05, 6)       | Primary mass    | Std of low-mass Gaussian          |
| `frac_peak_2`   | Dirichlet(1,1,1) | Primary mass    | Fraction in high-mass Gaussian    |
| `mu_peak_2`     | U(25, 60)        | Primary mass    | Mean of high-mass Gaussian        |
| `sigma_peak_2`  | U(0.05, 10)      | Primary mass    | Std of high-mass Gaussian         |
| `beta`          | U(-2, 7)         | Mass ratio      | Power-law slope for q             |
| `lamb`          | U(-10, 10)       | Redshift        | Power-law index on $(1+z)$          |
| `mu_spin`       | U(0, 0.7)        | Spin magnitudes | Shared mean of $(a_1, a_2)$           |
| `sigma_spin`    | U(0.01, 1)       | Spin magnitudes | Shared std of $(a_1, a_2)$            |
| `f_iso`         | U(0, 1)          | Spin tilts      | Isotropic fraction                |
| `mu_tilt`       | U(-1, 1)         | Spin tilts      | Shared truncated-Gaussian mean    |
| `sigma_tilt`    | U(0.05, 10)      | Spin tilts      | Tilt peak width                   |
| `mu_tgr`        | U(-s, s)         | TGR             | Population mean of A (auto-scaled)|
| `sigma_tgr`     | U(0, s)          | TGR             | Population std of A (auto-scaled) |