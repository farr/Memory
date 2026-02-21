Playing around with fitting the memory effect.

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

The differential rate is then

```
dR/dm1 = R(Λ) * p(m1 | Λ)
```

evaluated on a grid over the posterior to produce the shaded PPD band.

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

- **Peak 1** (low-mass): `mu_peak_1 ~ U(5, 20)`
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

| Parameter       | Prior           | Component       | Description                       |
|-----------------|-----------------|-----------------|-----------------------------------|
| `alpha_1`       | U(-4, 12)       | Primary mass    | PL slope below break              |
| `alpha_2`       | U(-4, 12)       | Primary mass    | PL slope above break              |
| `b`             | U(0, 1)         | Primary mass    | Break fraction                    |
| `frac_peak_1`   | U(0, 1)         | Primary mass    | Fraction in low-mass Gaussian     |
| `mu_peak_1`     | U(5, 20)        | Primary mass    | Mean of low-mass Gaussian         |
| `sigma_peak_1`  | U(1, 20)        | Primary mass    | Std of low-mass Gaussian          |
| `frac_peak_2`   | U(0, 1)         | Primary mass    | Fraction in high-mass Gaussian    |
| `mu_peak_2`     | U(20, 50)       | Primary mass    | Mean of high-mass Gaussian        |
| `sigma_peak_2`  | U(1, 20)        | Primary mass    | Std of high-mass Gaussian         |
| `beta`          | U(-4, 12)       | Mass ratio      | Power-law slope for q             |
| `lamb`          | U(-30, 30)      | Redshift        | Power-law index on (1+z)          |
| `mu_spin`       | U(0, 0.7)       | Spin magnitudes | Shared mean of (a1, a2)           |
| `sigma_spin`    | U(0.05, 10)     | Spin magnitudes | Shared std of (a1, a2)            |
| `f_iso`         | U(0, 1)         | Spin tilts      | Isotropic fraction (if enabled)   |
| `sigma_tilt`    | U(0.05, 10)     | Spin tilts      | Tilt peak width (if enabled)      |
| `mu_tgr`        | U(-s, s)        | TGR             | Population mean of A (auto-scaled)|
| `sigma_tgr`     | U(0, s)         | TGR             | Population std of A (auto-scaled) |