"""Numpyro model definitions for hierarchical TGR population analysis."""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from astropy import cosmology as cosmo
import astropy.units as u

# Setting up the redshift interpolant dV_c/dz /(1+z)
Planck15_LAL = cosmo.FlatLambdaCDM(H0=67.90, Om0=0.3065, name="Planck15_LAL")
zmax = 2.5
zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 1024))
dVdzdt_interp = (
    4
    * np.pi
    * Planck15_LAL.differential_comoving_volume(zinterp)
    .to(u.Gpc**3 / u.sr)
    .value
    / (1 + zinterp)
)


def make_tgr_only_model(A_hats, A_sigmas, Nobs, mu_tgr_scale=None,
                        sigma_tgr_scale=None):
    """Numpyro model for TGR-only hierarchical inference.

    Defines a two-hyperparameter model (mu_tgr, sigma_tgr) describing a
    Gaussian population distribution for the TGR deviation parameter.
    The per-event likelihood uses an analytic Gaussian convolution:
    the per-sample measurement uncertainty A_sigma is added in quadrature
    with sigma_tgr.

    Parameters
    ----------
    A_hats : jnp.ndarray
        Per-sample ML amplitude estimates, shape (Nobs, N_samples).
    A_sigmas : jnp.ndarray
        Per-sample amplitude uncertainties, shape (Nobs, N_samples).
    Nobs : int
        Number of observed events.
    mu_tgr_scale : float or None
        Half-width of the uniform prior on mu_tgr. If None, auto-scaled
        from the data.
    sigma_tgr_scale : float or None
        Upper bound of the uniform prior on sigma_tgr. If None, auto-scaled
        from the data.
    """
    A_scale = jnp.maximum(1e-6, jnp.max(jnp.abs(A_hats)))
    if mu_tgr_scale is None:
        mu_tgr_scale = 1.5 * A_scale
    if sigma_tgr_scale is None:
        sigma_tgr_scale = 1.5 * A_scale

    mu_tgr = numpyro.sample("mu_tgr", dist.Uniform(-mu_tgr_scale, mu_tgr_scale))
    sigma_tgr = numpyro.sample("sigma_tgr", dist.Uniform(0, sigma_tgr_scale))

    sigma_eff = jnp.sqrt(jnp.square(A_sigmas) + jnp.square(sigma_tgr))
    log_wts = dist.Normal(mu_tgr, sigma_eff).log_prob(A_hats)

    log_like = logsumexp(log_wts, axis=1)
    log_like = jnp.nan_to_num(log_like, neginf=-1e20, posinf=1e20)
    numpyro.factor("log_likelihood", jnp.sum(log_like))


def make_joint_model(
    event_data_array,
    injection_data_array,
    BW_matrices,
    BW_matrices_sel,
    Nobs,
    Ndraw,
    use_tilts,
    use_tgr,
    mu_tgr_scale=None,
    sigma_tgr_scale=None,
):
    """Numpyro model jointly fitting astrophysical population and TGR parameters.

    Combines a broken power law + two Gaussian peaks primary mass function,
    power-law mass ratio, power-law-in-(1+z) redshift distribution, Beta-like spin
    magnitude distribution (via multivariate normal in KDE space), and
    optional spin tilt mixture model, with Gaussian TGR hyperparameters
    (mu_tgr, sigma_tgr). The TGR dimension uses analytic Gaussian
    convolution rather than KDE smoothing. Includes selection-effect
    correction via importance-weighted injection sums and
    effective-sample-size regularization terms.

    Parameters
    ----------
    event_data_array : ndarray
        Shape (11, Nobs, N_samples): rows are m1, q, cos_tilt_1,
        cos_tilt_2, a_1, a_2, A_hat, A_sigma, z, log_pdraw, kde_weights.
    injection_data_array : ndarray
        Shape (8, N_inj): rows are m1, q, cos_tilt_1, cos_tilt_2,
        a_1/spin1z, a_2/spin2z, z, log_prior.
    BW_matrices : ndarray
        Per-event KDE bandwidth matrices for spin dimensions,
        shape (Nobs, 2, 2).
    BW_matrices_sel : ndarray
        Per-event KDE bandwidth matrices for spin dimensions (selection),
        shape (Nobs, 2, 2).
    Nobs : int
        Number of observed events.
    Ndraw : int
        Total number of simulated injections drawn.
    use_tilts : bool
        Whether to include the spin tilt mixture model.
    use_tgr : bool
        Whether to include TGR hyperparameters in the model.
    mu_tgr_scale : float or None
        Half-width of the uniform prior on mu_tgr.
    sigma_tgr_scale : float or None
        Upper bound of the uniform prior on sigma_tgr.
    """
    m1s = event_data_array[0]
    qs = event_data_array[1]
    cost1s = event_data_array[2]
    cost2s = event_data_array[3]
    a1_a2s = event_data_array[4:6]
    A_hats = event_data_array[6]
    A_sigmas = event_data_array[7]
    zs = event_data_array[8]
    log_pdraw = event_data_array[9]

    m1s_sel = injection_data_array[0]
    qs_sel = injection_data_array[1]
    cost1s_sel = injection_data_array[2]
    cost2s_sel = injection_data_array[3]
    a1_a2_sel = injection_data_array[4:6]
    zs_sel = injection_data_array[6]
    log_pdraw_sel = injection_data_array[7]

    # Model parameters
    # Mass distribution: broken power law + two Gaussian peaks
    alpha_1 = numpyro.sample("alpha_1", dist.Uniform(-4, 12))
    alpha_2 = numpyro.sample("alpha_2", dist.Uniform(-4, 12))
    beta = numpyro.sample("beta", dist.Uniform(-4, 12))
    mmin = 5
    mmax = 100
    b = numpyro.sample("b", dist.Uniform(0, 1))

    # Dirichlet prior ensures frac_bpl + frac_peak_1 + frac_peak_2 = 1
    # with all fractions strictly positive — avoids the hard constraint /
    # gradient-killing penalty that arises from two independent Uniform(0,1)
    # draws that can sum to > 1.
    fracs = numpyro.sample("fracs", dist.Dirichlet(jnp.ones(3)))
    frac_bpl    = numpyro.deterministic("frac_bpl",    fracs[0])
    frac_peak_1 = numpyro.deterministic("frac_peak_1", fracs[1])
    frac_peak_2 = numpyro.deterministic("frac_peak_2", fracs[2])

    mu_peak_1 = numpyro.sample("mu_peak_1", dist.Uniform(mmin, 15))
    # Peak widths bounded to [0.5, 8]: prevents the Gaussian components from
    # becoming degenerate broad components that absorb the BPL, which creates
    # multi-modal posteriors and poor NUTS conditioning.
    sigma_peak_1 = numpyro.sample("sigma_peak_1", dist.Uniform(0.5, 8))

    mu_peak_2 = numpyro.sample("mu_peak_2", dist.Uniform(15, 75))
    sigma_peak_2 = numpyro.sample("sigma_peak_2", dist.Uniform(0.5, 8))

    # Spin magnitude distribution parameters.
    # sigma_spin prior is restricted to [0.01, 0.5]: the posterior sits at
    # ~0.13, which was only 0.9% through the old [0.05, 10] range, mapping
    # to logit ≈ -4.65 in the unconstrained NUTS space and collapsing the
    # step size.  With [0.01, 0.5] the same posterior is at ~27% (logit ≈ -1).
    mu_spin = numpyro.sample("mu_spin", dist.Uniform(0, 0.7))
    sigma_spin = numpyro.sample("sigma_spin", dist.Uniform(0.01, 0.5))

    # Redshift parameters
    lamb = numpyro.sample("lamb", dist.Uniform(-30, 30))

    # Defining the models
    def log_m1_density(primary_masses):
        huge_neg = -1.0e12
        indicator = jnp.where(jnp.greater_equal(primary_masses, mmin), 0.0, huge_neg)

        m_break = mmin + b * (mmax - mmin)

        # Log of normalized power law on [lo, hi]
        def log_norm_pl(m, a, lo, hi):
            log_unnorm = -a * jnp.log(m)
            log_norm = jax.lax.select(
                jnp.isclose(a, 1.0),
                -jnp.log(jnp.log(hi / lo)),
                jnp.log((1 - a) / (hi ** (1 - a) - lo ** (1 - a))),
            )
            return log_unnorm + log_norm

        # Log-density of each PL segment
        log_pl_low = log_norm_pl(primary_masses, alpha_1, mmin, m_break)
        log_pl_high = log_norm_pl(primary_masses, alpha_2, m_break, mmax)

        # Continuity correction: C = PL_high(m_break) / PL_low(m_break)
        log_C = (log_norm_pl(m_break, alpha_2, m_break, mmax)
                 - log_norm_pl(m_break, alpha_1, mmin, m_break))

        # BPL: select segment, apply continuity weight, normalize
        is_low = primary_masses < m_break
        log_bpl = (jnp.where(is_low, log_pl_low + log_C, log_pl_high)
                   - jnp.logaddexp(0.0, log_C))

        # Two Gaussian peaks
        log_gauss_1 = dist.Normal(mu_peak_1, sigma_peak_1).log_prob(
            primary_masses.T
        ).T
        log_gauss_2 = dist.Normal(mu_peak_2, sigma_peak_2).log_prob(
            primary_masses.T
        ).T

        # Three-component mixture (fracs sum to 1 by Dirichlet construction)
        comp_bpl = jnp.log(jnp.maximum(frac_bpl, 1e-30)) + log_bpl
        comp_1 = jnp.log(jnp.maximum(frac_peak_1, 1e-30)) + log_gauss_1
        comp_2 = jnp.log(jnp.maximum(frac_peak_2, 1e-30)) + log_gauss_2

        return (logsumexp(jnp.stack([comp_bpl, comp_1, comp_2], axis=0), axis=0)
                + indicator)

    def log_q_powerlaw_density(mass_ratios, primary_masses):
        low = mmin / primary_masses

        log_norm = jax.lax.select(
            jnp.isclose(beta, -1),
            jnp.log(-1 / jnp.log(low)),
            jnp.log((1 + beta) / (1 - low ** (1 + beta))),
        )

        log_norm = jax.lax.select(
            jnp.isnan(log_norm), -jnp.inf * jnp.ones(jnp.shape(log_norm)), log_norm
        )

        indicator = jnp.where(jnp.greater_equal(mass_ratios, low), 0.0, -1.0e12)
        return beta * jnp.log(mass_ratios) + log_norm + indicator

    def log_redshift_powerlaw(redshifts):
        interp_val = jnp.interp(redshifts, zinterp, dVdzdt_interp)
        interp_val = jnp.clip(interp_val, a_min=1e-300)
        return lamb * jnp.log1p(redshifts) + jnp.log(interp_val)

    log_wts = (
        log_m1_density(m1s)
        + log_q_powerlaw_density(qs, m1s)
        + log_redshift_powerlaw(zs)
        - log_pdraw
    )

    # Evaluate the selection term
    log_sel_wts = (
        log_m1_density(m1s_sel)
        + log_q_powerlaw_density(qs_sel, m1s_sel)
        + log_redshift_powerlaw(zs_sel)
        - log_pdraw_sel
    )

    # Adding the tilt
    if use_tilts:
        f_iso = numpyro.sample("f_iso", dist.Uniform(0, 1))
        sigma_tilt = numpyro.sample("sigma_tilt", dist.Uniform(0.05, 10))

        def log_tilt_density(cost1, cost2):
            quad = ((cost1 - 1) ** 2 + (cost2 - 1) ** 2)
            log_gauss = -quad / (2 * jnp.square(sigma_tilt)) - jnp.log(
                2 * jnp.pi * jnp.square(sigma_tilt)
            )
            term_a = jnp.log(f_iso) - jnp.log(4.0)
            term_b = jnp.log1p(-f_iso) + log_gauss
            return logsumexp(jnp.stack([term_a, term_b], axis=0), axis=0)

        log_wts += log_tilt_density(cost1s, cost2s)
        log_sel_wts += log_tilt_density(cost1s_sel, cost2s_sel)

    # Spin KDE — always 2×2 (a1, a2); the TGR dimension is handled
    # analytically below.
    sigma_evts = BW_matrices + jnp.diag(
        jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])
    )
    sigma_evts = sigma_evts + jnp.eye(2) * 1e-15
    sigma_sel = jnp.diag(
        jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])
    )
    sigma_sel = sigma_sel + jnp.eye(2) * 1e-15
    mu_evts = jnp.array([mu_spin, mu_spin])

    logp_normal_sel = dist.MultivariateNormal(
        jnp.array([mu_spin, mu_spin]), sigma_sel
    ).log_prob(a1_a2_sel.T)
    log_sel_wts += logp_normal_sel

    logp_normal = (
        dist.MultivariateNormal(mu_evts, sigma_evts)
        .log_prob(
            jnp.array(
                [
                    a1_a2s.T,
                ]
            )
        )
        .T[:, :, 0]
    )
    log_wts += logp_normal

    # TGR: analytic Gaussian convolution over the memory amplitude
    if use_tgr:
        A_scale = jnp.maximum(1e-6, jnp.max(jnp.abs(A_hats)))
        if mu_tgr_scale is None:
            mu_tgr_scale = 1.5 * A_scale
        if sigma_tgr_scale is None:
            sigma_tgr_scale = 1.5 * A_scale

        mu_tgr = numpyro.sample("mu_tgr", dist.Uniform(-mu_tgr_scale, mu_tgr_scale))
        sigma_tgr = numpyro.sample("sigma_tgr", dist.Uniform(0, sigma_tgr_scale))

        sigma_eff = jnp.sqrt(jnp.square(A_sigmas) + jnp.square(sigma_tgr))
        log_wts += dist.Normal(mu_tgr, sigma_eff).log_prob(A_hats)

    # Adding the per event likelihood term (stable log-sum-exp)
    log_like = logsumexp(log_wts, axis=1)
    log_like = jnp.nan_to_num(log_like, neginf=-1e20, posinf=1e20)
    numpyro.factor("log_likelihood", jnp.sum(log_like))

    # Selection effect term
    log_sel = logsumexp(log_sel_wts) - jnp.log(Ndraw)
    log_sel = jnp.nan_to_num(log_sel, neginf=-1e20, posinf=1e20)
    numpyro.factor("selection", -Nobs * log_sel)

    # N eff cuts
    def log_smooth_neff_boundary(values, criteria):
        scaled_x = (values - criteria) / (0.05 * criteria)
        # Linear ramp below threshold: gradient is bounded to 1 everywhere,
        # preventing the step-size collapse that power-4 (gradient ~32000 at
        # scaled_x=-20) and power-10 (gradient ~5e12) caused.
        return jnp.minimum(0.0, scaled_x)

    neff = jnp.exp(2 * logsumexp(log_wts, axis=1) - logsumexp(2 * log_wts, axis=1))
    min_neff = jnp.min(neff)
    numpyro.deterministic("neff", neff)
    numpyro.factor(
        "neff_criteria",
        jnp.nan_to_num(log_smooth_neff_boundary(min_neff, Nobs), neginf=-1e20, posinf=1e20),
    )

    log_mu2 = logsumexp(2 * log_sel_wts) - 2 * jnp.log(Ndraw)
    ratio = jnp.exp(2 * log_sel - jnp.log(Ndraw) - log_mu2)
    ratio = jnp.clip(ratio, a_min=0.0, a_max=1.0 - 1e-12)
    log_s2 = log_mu2 + jnp.log1p(-ratio)
    neff_sel = jnp.exp(2 * log_sel - log_s2)
    numpyro.deterministic("neff_sel", neff_sel)
    numpyro.factor(
        "neff_sel_criteria",
        jnp.nan_to_num(
            log_smooth_neff_boundary(neff_sel, 4 * Nobs), neginf=-1e20, posinf=1e20
        ),
    )
