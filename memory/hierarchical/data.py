"""Data loading and preparation for hierarchical TGR population analysis."""

import logging
import os
import re

import astropy.units as u
from astropy.cosmology import z_at_value
import numpy as np
import h5py
from tqdm import tqdm

logger = logging.getLogger(__name__)

IFAR_THRESHOLD = 1
SEMIANALYTIC_SNR_THRESHOLD = 10.0
N_SAMPLES_PER_EVENT = 10000
NRSUR_MIN_DETECTOR_FRAME_TOTAL_MASS = 66.0
NRSUR_MIN_MASS_RATIO = 1.0 / 6.0
MIN_MASS_2_SOURCE = 3.0

# Events explicitly excluded from the memory analysis (outside the population
# model's scope; also caught by the min_mass_2_source cut).
_EXCLUDED_EVENTS = frozenset({
    "GW200105_162426",  # NSBH O3b
    "GW200115_042309",  # NSBH O3b
    "GW230518_125908",  # NSBH O4a
    "GW230529_181500",  # NSBH O4a
})

# Per-sample sanity threshold: individual samples with |A_hat / A_sigma|
# exceeding this value are removed before use.  For well-behaved events the
# memory SNR per sample is << 1
_MAX_SAMPLE_SNR = 10.0

# Event-level sanity threshold: after sample filtering, if the *median*
# |A_hat / A_sigma| still exceeds this value the entire event is skipped.
_MAX_EVENT_MEDIAN_SNR = 5.0


def _waveform_sort_key(label):
    """Sort waveform labels by descending calibration, then alphabetically."""
    match = re.match(r"C(\d+):", label)
    calibration = int(match.group(1)) if match else -1
    return (-calibration, label)


def _imrphenom_sort_key(label):
    """Secondary sort key for IMRPhenom labels: preferred variants rank first.

    Ordering (most to least preferred):
      IMRPhenomXO4a > IMRPhenomXPHM-SpinTaylor > IMRPhenomXPHM > any other IMRPhenom
    Within each sub-tier the calibration/alphabetical key applies.
    """
    bare = label.split(":", 1)[-1]
    if "IMRPhenomXO4a" in bare:
        sub = 0
    elif "IMRPhenomXPHM-SpinTaylor" in bare:
        sub = 1
    elif "IMRPhenomXPHM" in bare:
        sub = 2
    else:
        sub = 3
    return (sub,) + _waveform_sort_key(label)


def _pick_waveform_label(keys):
    """Pick the best available waveform label using the priority hierarchy.

    Priority: NRSur > SEOB > IMRPhenomXO4a > IMRPhenomXPHM-SpinTaylor >
              IMRPhenomXPHM > any other IMRPhenom.
    Within each tier, higher calibration versions win (e.g. C01 > C00).
    Ties within calibration version are broken alphabetically.
    """
    keys_sorted = sorted(keys, key=_waveform_sort_key)
    for k in keys_sorted:
        if "NRSur" in k:
            return k
    for k in keys_sorted:
        if "SEOB" in k:
            return k
    imrphenom = [k for k in keys_sorted if "IMRPhenom" in k]
    if imrphenom:
        return sorted(imrphenom, key=_imrphenom_sort_key)[0]
    return keys_sorted[0]


def _resolve_waveform_label(keys, waveform=None):
    """Resolve *waveform* to the best matching label in *keys*.

    ``None`` or ``"auto"`` uses the default NRSur > SEOB > IMRPhenomXO4a >
    IMRPhenomXPHM-SpinTaylor > IMRPhenomXPHM > other IMRPhenom hierarchy. A bare waveform name like ``"NRSur7dq4"`` selects the
    highest available ``CXX:NRSur7dq4`` label for that file.
    """
    if not keys:
        raise KeyError("No waveform labels available")

    if waveform is None or waveform == "auto":
        return _pick_waveform_label(keys)

    if waveform in keys:
        return waveform

    requested = waveform
    if re.match(r"C\d+:", requested):
        requested = requested.split(":", 1)[1]

    matches = [key for key in keys if key.split(":", 1)[-1] == requested]
    if matches:
        return sorted(matches, key=_waveform_sort_key)[0]

    raise KeyError(
        f"Waveform '{waveform}' not found; available labels: {list(keys)}"
    )


def _decode_hdf5_scalar(value):
    """Return a Python scalar/string from an HDF5 dataset value."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return value


def _get_hdf5_text(group, path):
    """Return decoded text/scalar for *path* inside *group*, or None."""
    if path not in group:
        return None
    return _decode_hdf5_scalar(group[path][()])


def _get_structured_field(table, *names):
    """Return the first matching field from a structured array/table."""
    dtype_names = getattr(table.dtype, "names", ()) or ()
    for name in names:
        if name in dtype_names:
            return table[name]
    raise KeyError(
        f"None of the requested fields {names} were found in {dtype_names}"
    )


def _get_injection_redshift(events):
    """Return the redshift column from a sensitivity-estimate table."""
    return _get_structured_field(events, "redshift", "z")


def _get_injection_spin_data(events):
    """Return spin magnitudes/tilts derived from polar or Cartesian fields."""
    dtype_names = set(getattr(events.dtype, "names", ()) or ())
    polar_fields = {
        "spin1_magnitude",
        "spin1_polar_angle",
        "spin2_magnitude",
        "spin2_polar_angle",
    }
    if polar_fields.issubset(dtype_names):
        a1 = np.asarray(events["spin1_magnitude"])
        a2 = np.asarray(events["spin2_magnitude"])
        sin_tilt_1 = np.sin(events["spin1_polar_angle"])
        sin_tilt_2 = np.sin(events["spin2_polar_angle"])
        cos_tilt_1 = np.clip(np.cos(events["spin1_polar_angle"]), -1.0, 1.0)
        cos_tilt_2 = np.clip(np.cos(events["spin2_polar_angle"]), -1.0, 1.0)
        return {
            "a_1": a1,
            "a_2": a2,
            "sin_tilt_1": np.clip(sin_tilt_1, 0.0, None),
            "sin_tilt_2": np.clip(sin_tilt_2, 0.0, None),
            "cos_tilt_1": cos_tilt_1,
            "cos_tilt_2": cos_tilt_2,
            "spin1z": a1 * cos_tilt_1,
            "spin2z": a2 * cos_tilt_2,
            "draw_coordinates": "polar",
        }

    s1x = np.asarray(_get_structured_field(events, "spin1x"))
    s1y = np.asarray(_get_structured_field(events, "spin1y"))
    s1z = np.asarray(_get_structured_field(events, "spin1z"))
    s2x = np.asarray(_get_structured_field(events, "spin2x"))
    s2y = np.asarray(_get_structured_field(events, "spin2y"))
    s2z = np.asarray(_get_structured_field(events, "spin2z"))
    a1 = np.sqrt(s1x**2 + s1y**2 + s1z**2)
    a2 = np.sqrt(s2x**2 + s2y**2 + s2z**2)
    return {
        "a_1": a1,
        "a_2": a2,
        "sin_tilt_1": np.sqrt(s1x**2 + s1y**2) / np.maximum(a1, 1e-30),
        "sin_tilt_2": np.sqrt(s2x**2 + s2y**2) / np.maximum(a2, 1e-30),
        "cos_tilt_1": np.clip(s1z / np.maximum(a1, 1e-30), -1.0, 1.0),
        "cos_tilt_2": np.clip(s2z / np.maximum(a2, 1e-30), -1.0, 1.0),
        "spin1z": s1z,
        "spin2z": s2z,
        "draw_coordinates": "cartesian",
    }


def _get_injection_log_draw_prior(events):
    """Return the stored log draw density and its spin coordinate system."""
    dtype_names = set(getattr(events.dtype, "names", ()) or ())

    polar_joint_name = (
        "lnpdraw_mass1_source_mass2_source_redshift_"
        "spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_"
        "spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle"
    )
    if polar_joint_name in dtype_names:
        return np.asarray(events[polar_joint_name]), "polar"

    polar_factorized_names = (
        "lnpdraw_mass1_source",
        "lnpdraw_mass2_source_GIVEN_mass1_source",
        "lnpdraw_z",
        "lnpdraw_spin1_magnitude",
        "lnpdraw_spin1_polar_angle",
        "lnpdraw_spin1_azimuthal_angle",
        "lnpdraw_spin2_magnitude",
        "lnpdraw_spin2_polar_angle",
        "lnpdraw_spin2_azimuthal_angle",
    )
    if all(name in dtype_names for name in polar_factorized_names):
        return (
            np.sum(
                [np.asarray(events[name]) for name in polar_factorized_names],
                axis=0,
            ),
            "polar",
        )

    cartesian_joint_name = (
        "lnpdraw_mass1_source_mass2_source_redshift_"
        "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
    )
    if cartesian_joint_name in dtype_names:
        return np.asarray(events[cartesian_joint_name]), "cartesian"

    raise KeyError(
        "Could not identify a supported injection draw-density field in the "
        "selection file."
    )


def _build_prior_eval_namespace():
    """Return the namespace needed to evaluate bilby prior repr strings."""
    import bilby
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, wCDM
    from bilby.core.prior import (
        Constraint,
        Cosine,
        DeltaFunction,
        Gaussian,
        LogUniform,
        PowerLaw,
        Sine,
        TruncatedGaussian,
        Uniform,
    )
    from bilby.gw.prior import (
        AlignedSpin,
        UniformComovingVolume,
        UniformInComponentsChirpMass,
        UniformInComponentsMassRatio,
        UniformSourceFrame,
    )

    return {
        "AlignedSpin": AlignedSpin,
        "bilby": bilby,
        "Constraint": Constraint,
        "Cosine": Cosine,
        "DeltaFunction": DeltaFunction,
        "FlatLambdaCDM": FlatLambdaCDM,
        "FlatwCDM": FlatwCDM,
        "Gaussian": Gaussian,
        "LambdaCDM": LambdaCDM,
        "LogUniform": LogUniform,
        "PowerLaw": PowerLaw,
        "Sine": Sine,
        "TruncatedGaussian": TruncatedGaussian,
        "Uniform": Uniform,
        "UniformComovingVolume": UniformComovingVolume,
        "UniformInComponentsChirpMass": UniformInComponentsChirpMass,
        "UniformInComponentsMassRatio": UniformInComponentsMassRatio,
        "UniformSourceFrame": UniformSourceFrame,
        "wCDM": wCDM,
    }


def _evaluate_prior_repr(prior_repr):
    """Evaluate a bilby prior repr string into a prior object."""
    namespace = _build_prior_eval_namespace()
    return eval(prior_repr, namespace, namespace)


def _metadata_indicates_cosmology_reweighting(group):
    """Return True when the group metadata indicates cosmology reweighting."""
    description = _get_hdf5_text(group, "description")
    new_metafile = _get_hdf5_text(group, "meta_data/reweighting/new_metafile")
    metafile = _get_hdf5_text(group, "meta_data/reweighting/metafile")
    new_cosmology = _get_hdf5_text(group, "meta_data/reweighting/new_cosmology")

    text_fragments = [
        str(description or "").lower(),
        str(new_metafile or "").lower(),
        str(metafile or "").lower(),
        str(new_cosmology or "").lower(),
    ]
    return any(
        token in fragment
        for fragment in text_fragments
        for token in (
            "reweighted posterior and prior samples",
            "posterior_samples_cosmo",
            "new_cosmology",
        )
    )


def _sample_subset_indices(n_total, max_samples=128):
    """Return evenly spaced sample indices for validation."""
    if n_total <= max_samples:
        return np.arange(n_total, dtype=int)
    return np.linspace(0, n_total - 1, max_samples, dtype=int)


def validate_posterior_prior_consistency(
    group,
    posterior_samples,
    *,
    filename=None,
    label=None,
):
    """Best-effort validation that stored prior weights match sample metadata.

    This is primarily aimed at public release files that may contain samples
    reweighted between cosmological conventions. When the metadata indicates a
    cosmology-sensitive reweighting, the function verifies that:

    1. The posterior samples include an explicit prior column (`log_prior` or
       `prior`), since the hierarchical model reuses that weight.
    2. The released `luminosity_distance`, `redshift`, and source-frame masses
       are internally consistent with the cosmology encoded in the file's
       analytic prior metadata.

    The release structure does not always permit an exact reconstruction of the
    full joint `log_prior`, so this function intentionally raises only on
    inconsistencies we can verify directly from metadata and samples.
    """
    prefix = []
    if filename:
        prefix.append(os.path.basename(filename))
    if label:
        prefix.append(label)
    prefix = ": ".join(prefix) if prefix else "posterior_samples"

    dtype_names = posterior_samples.dtype.names or ()
    has_log_prior = "log_prior" in dtype_names
    has_prior = "prior" in dtype_names

    reweighted = _metadata_indicates_cosmology_reweighting(group)
    if reweighted and not (has_log_prior or has_prior):
        raise ValueError(
            f"{prefix}: metadata indicates cosmology reweighting, but the "
            "posterior samples do not contain a 'log_prior' or 'prior' field. "
            "Using these samples in the hierarchical model would risk applying "
            "the wrong prior weight."
        )

    prior_repr = _get_hdf5_text(group, "priors/analytic/luminosity_distance")
    if prior_repr is None:
        return

    try:
        distance_prior = _evaluate_prior_repr(prior_repr)
    except Exception as exc:
        if reweighted:
            raise ValueError(
                f"{prefix}: failed to parse luminosity-distance prior metadata "
                f"for cosmology validation: {exc}"
            ) from exc
        return

    cosmology = getattr(distance_prior, "cosmology", None)
    if cosmology is None:
        return

    required = {"luminosity_distance", "redshift"}
    if not required.issubset(dtype_names):
        raise ValueError(
            f"{prefix}: cosmology-aware distance prior is recorded in the "
            "metadata, but the posterior samples are missing one of the "
            f"required fields {sorted(required)}."
        )

    idx = _sample_subset_indices(len(posterior_samples))
    luminosity_distance = np.asarray(
        posterior_samples["luminosity_distance"][idx], dtype=float
    )
    redshift = np.asarray(posterior_samples["redshift"][idx], dtype=float)
    finite = np.isfinite(luminosity_distance) & np.isfinite(redshift)
    if not np.any(finite):
        raise ValueError(
            f"{prefix}: no finite luminosity-distance/redshift samples were "
            "available for cosmology validation."
        )

    luminosity_distance = luminosity_distance[finite]
    redshift = redshift[finite]

    z_expected = np.array(
        [
            float(
                z_at_value(
                    cosmology.luminosity_distance,
                    float(dl) * u.Mpc,
                    zmin=0.0,
                    zmax=max(10.0, 2.0 * float(np.max(redshift)) + 1.0),
                )
            )
            for dl in luminosity_distance
        ]
    )
    z_mismatch = np.max(np.abs(z_expected - redshift))
    if z_mismatch > 5e-4:
        raise ValueError(
            f"{prefix}: sample redshifts are inconsistent with the cosmology "
            f"encoded in the luminosity-distance prior metadata (max |dz| = "
            f"{z_mismatch:.3e})."
        )

    source_mass_pairs = [
        ("mass_1", "mass_1_source"),
        ("mass_2", "mass_2_source"),
    ]
    for detector_name, source_name in source_mass_pairs:
        if detector_name not in dtype_names or source_name not in dtype_names:
            continue
        detector_mass = np.asarray(posterior_samples[detector_name][idx], dtype=float)[finite]
        source_mass = np.asarray(posterior_samples[source_name][idx], dtype=float)[finite]
        source_expected = detector_mass / (1.0 + redshift)
        denom = np.maximum(np.abs(source_expected), 1e-12)
        rel_mismatch = np.max(np.abs(source_expected - source_mass) / denom)
        if rel_mismatch > 5e-4:
            raise ValueError(
                f"{prefix}: {source_name} is inconsistent with {detector_name} "
                "and the cosmology-adjusted redshift samples "
                f"(max relative mismatch = {rel_mismatch:.3e})."
            )

    if has_log_prior:
        prior_values = np.asarray(posterior_samples["log_prior"][idx], dtype=float)
    elif has_prior:
        prior_values = np.log(
            np.clip(np.asarray(posterior_samples["prior"][idx], dtype=float), 1e-300, None)
        )
    else:
        prior_values = None

    if prior_values is not None:
        prior_values = prior_values[finite]
        if not np.all(np.isfinite(prior_values)):
            raise ValueError(
                f"{prefix}: prior weights contain non-finite values."
            )
        # A cosmology-driven reweighting must leave a non-degenerate prior
        # column behind; a constant column would indicate the update was lost.
        if reweighted and np.nanstd(prior_values) < 1e-6:
            raise ValueError(
                f"{prefix}: metadata indicates cosmology reweighting, but the "
                "stored prior weights are numerically constant."
            )


def _get_config_type(group):
    """Return 'lalinference', 'bilby', or None based on config_file structure."""
    if "config_file" not in group:
        return None
    cfg = group["config_file"]
    if "engine" in cfg:      # LALInference has an [engine] section
        return "lalinference"
    if "config" in cfg:      # bilby stores the full ini under 'config'
        return "bilby"
    return None


def _compute_log_prior_lalinference(group, posterior_samples):
    """Reconstruct log_prior for a LALInference run from its config file.

    Reads prior bounds from the ``[engine]`` section of the stored INI config
    and evaluates the standard O3 LALInference prior at each sample.

    Returns
    -------
    log_prior : ndarray of shape (n_samples,)
    params_used : list of str
    """
    from bilby.core.prior import Cosine, PowerLaw, Sine, Uniform

    eng = group["config_file"]["engine"]

    def _get_float(key, default):
        if key not in eng:
            return default
        v = eng[key][()]
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, list):
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    mc_min = _get_float("chirpmass-min", 5.0)
    mc_max = _get_float("chirpmass-max", 100.0)
    q_min = _get_float("q-min", 0.05)
    a1_max = _get_float("a_spin1-max", 0.99)
    a2_max = _get_float("a_spin2-max", 0.99)
    d_min = _get_float("distance-min", 10.0)
    d_max = _get_float("distance-max", 10000.0)

    # symmetric mass ratio range corresponding to q_min
    eta_min = q_min / (1.0 + q_min) ** 2

    # Prior contributions for importance weighting.
    #
    # Only non-constant terms need to be evaluated: constant (flat-prior)
    # parameters cancel in the importance weight ratio p_pop / p_PE within
    # each event and can safely be omitted.
    #
    # Non-constant terms in the standard O3 LALInference prior:
    #   - luminosity_distance: p(d) ∝ d^2  (PowerLaw alpha=2)
    #   - tilt_1, tilt_2: isotropic spin orientations → p(tilt) ∝ sin(tilt)
    #   - theta_jn: isotropic inclination → p(theta_jn) ∝ sin(theta_jn)
    #   - dec: isotropic sky → p(dec) ∝ cos(dec)
    #
    # Flat-prior parameters (chirp_mass, symmetric_mass_ratio, a_1, a_2,
    # ra, psi, phase, phi_12, phi_jl, geocent_time) are deliberately omitted
    # to avoid -inf from samples that slightly exceed the config bounds due to
    # derived-parameter rounding or post-processing conventions.
    priors = {
        "tilt_1": Sine(),          # p(tilt) = sin(tilt)/2
        "tilt_2": Sine(),
        "theta_jn": Sine(),        # inclination-like angle
        "dec": Cosine(),           # p(dec) = cos(dec)/2, range [-pi/2, pi/2]
        "luminosity_distance": PowerLaw(alpha=2, minimum=d_min, maximum=d_max),
    }

    dtype_names = set(posterior_samples.dtype.names)
    log_prior = np.zeros(len(posterior_samples))
    params_used = []

    for param, prior in priors.items():
        if param not in dtype_names:
            continue
        values = np.asarray(posterior_samples[param], dtype=float)
        lp = prior.ln_prob(values)
        log_prior += np.where(np.isfinite(lp), lp, -np.inf)
        params_used.append(param)

    return log_prior, params_used


def _compute_log_prior_bilby_analytic(group, posterior_samples):
    """Compute log_prior from a bilby run's stored ``priors/analytic`` entries.

    Evaluates each stored prior repr at the matching column in
    *posterior_samples* and sums the log-probabilities.

    Returns
    -------
    log_prior : ndarray of shape (n_samples,) or None
    params_used : list of str
    """
    if "priors" not in group or "analytic" not in group["priors"]:
        return None, []

    analytic_grp = group["priors"]["analytic"]
    keys = list(analytic_grp.keys())
    if not keys:
        return None, []

    dtype_names = set(posterior_samples.dtype.names)
    log_prior = np.zeros(len(posterior_samples))
    params_used = []

    for param in keys:
        if param not in dtype_names:
            continue
        raw = analytic_grp[param][()]
        # HDF5 stores these as bytes or byte-array scalars/lists
        if isinstance(raw, (bytes, np.bytes_)):
            prior_repr = raw.decode()
        elif hasattr(raw, "tolist"):
            raw = raw.tolist()
            if isinstance(raw, list):
                raw = raw[0]
            prior_repr = raw.decode() if isinstance(raw, bytes) else str(raw)
        else:
            prior_repr = str(raw)

        try:
            prior = _evaluate_prior_repr(prior_repr)
            values = np.asarray(posterior_samples[param], dtype=float)
            lp = prior.ln_prob(values)
            log_prior += np.where(np.isfinite(lp), lp, -np.inf)
            params_used.append(param)
        except Exception:
            continue

    if not params_used:
        return None, []
    return log_prior, params_used


def compute_log_prior_from_config(group, posterior_samples):
    """Attempt to reconstruct ``log_prior`` from stored config metadata.

    Tries in order:
    1. LALInference: parse the ``[engine]`` section of the stored INI for
       prior bounds and evaluate the standard O3 prior.
    2. bilby: evaluate the ``priors/analytic`` repr strings stored in the
       group.

    Parameters
    ----------
    group : h5py.Group
        The waveform group inside the PE HDF5 file.
    posterior_samples : structured ndarray
        The posterior samples array for *group*.

    Returns
    -------
    log_prior : ndarray of shape (n_samples,) or None
        ``None`` when computation is not possible.
    params_used : list of str
        Names of the parameters whose log-prior contributions were summed.
    """
    cfg_type = _get_config_type(group)

    if cfg_type == "lalinference":
        try:
            return _compute_log_prior_lalinference(group, posterior_samples)
        except Exception as exc:
            logger.debug(
                "LALInference prior reconstruction failed: %s", exc
            )
            return None, []

    if cfg_type == "bilby":
        result, params = _compute_log_prior_bilby_analytic(
            group, posterior_samples
        )
        if result is not None:
            return result, params

    return None, []


def load_event_ifars(event_names, cache_file=None):
    """Return a dict mapping event name to IFAR (years) for *event_names*.

    If *cache_file* is given and the file exists it is read directly (one
    ``event_name IFAR`` pair per line, whitespace-separated).  Otherwise the
    GWOSC event API is queried for each event.  When *cache_file* is provided
    and the file does not yet exist, the fetched results are written there for
    future use.

    IFAR is computed as ``1 / FAR`` where FAR is reported by GWOSC in units of
    yr^-1.  Events whose FAR is zero or missing are assigned ``inf``.

    Parameters
    ----------
    event_names : list of str
        GW event names, e.g. ``["GW150914_095045", ...]``.
    cache_file : str or None
        Path to the plain-text cache file.  If None the GWOSC API is always
        queried directly (results are not cached).

    Returns
    -------
    dict[str, float]
        Map from event name to IFAR in years.
    """
    if cache_file is not None and os.path.exists(cache_file):
        logger.info("Loading event IFARs from cache: %s", cache_file)
        ifars = {}
        with open(cache_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed line in IFAR cache {cache_file!r}: {line!r}"
                    )
                ifars[parts[0]] = float(parts[1])
        return ifars

    if cache_file is not None:
        logger.info(
            "Fetching IFARs from GWOSC for %d events; will cache to %s",
            len(event_names), cache_file,
        )
    else:
        logger.info(
            "Fetching IFARs from GWOSC for %d events (no cache file)",
            len(event_names),
        )
    try:
        from gwosc import api as _gwosc_api
    except ImportError as exc:
        raise ImportError(
            "gwosc is required to fetch IFARs; install it or provide a cache file"
        ) from exc

    ifars = {}
    for ev in event_names:
        # O1/O2 events are registered without the time suffix (e.g. "GW150914")
        candidates = [ev, re.sub(r"_\d{6}$", "", ev)]
        ifar = float("inf")
        for name in candidates:
            try:
                result = _gwosc_api.fetch_event_json(name)
                evdata = list(result["events"].values())[0]
                far = evdata.get("far")  # units: yr^-1
                ifar = 1.0 / far if far and far > 0 else float("inf")
                break
            except Exception as exc:
                last_exc = exc
        else:
            logger.warning("Could not fetch IFAR for %s from GWOSC: %s", ev, last_exc)
        ifars[ev] = ifar
        logger.debug("%s: IFAR = %.3g yr", ev, ifar)

    if cache_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        with open(cache_file, "w") as fh:
            fh.write("# event_name IFAR_yr\n")
            for ev, ifar in ifars.items():
                fh.write(f"{ev} {ifar}\n")
        logger.info("Saved IFAR cache to %s", cache_file)

    return ifars


def load_memory_data(event_files, memory_dir, waveform_label=None):
    """Load per-event memory results to use as the TGR parameter source.

    For each event file, extracts the event name (e.g., GW190814_211039),
    looks up ``{memory_dir}/{event_name}/memory_results.h5``, and reads
    the ``A_sample`` and ``log_weight`` datasets from the specified
    waveform group.

    Parameters
    ----------
    event_files : list of str
        Paths to the PE posterior files (used to extract event names).
    memory_dir : str
        Directory containing per-event subdirectories with
        ``memory_results.h5`` files.
    waveform_label : str or None
        Requested waveform inside each memory file. If None, the best
        available group is selected using the priority hierarchy
        NRSur > SEOB > IMRPhenom. If a bare waveform name is provided,
        the highest available ``CXX:<waveform>`` label is selected
        separately for each event.

    Returns
    -------
    list of dict
        One dict per event with keys ``'A_sample'``, ``'A_hat'``,
        ``'A_sigma'``, ``'log_weight'`` (1-D float arrays),
        ``'event_name'`` (str), and ``'waveform_label'`` (str).

    Raises
    ------
    FileNotFoundError
        If a memory results file cannot be found for an event.
    KeyError
        If no groups are present, or if auto-selection fails.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    memory_data = []
    n_excluded = 0
    n_missing_waveform = 0
    n_high_snr = 0
    for event_file in event_files:
        basename = os.path.basename(event_file)
        match = re.search(r"(GW\d{6}_\d{6})", basename)
        if match is None:
            raise ValueError(
                f"Could not extract event name from filename: {basename}"
            )
        event_name = match.group(1)

        if event_name in _EXCLUDED_EVENTS:
            _log.warning(
                "%s: skipping explicitly excluded event",
                event_name,
            )
            n_excluded += 1
            continue

        mem_path = os.path.join(memory_dir, event_name, "memory_results.h5")
        if not os.path.exists(mem_path):
            raise FileNotFoundError(
                f"Memory results file not found for {event_name}: {mem_path}"
            )

        with h5py.File(mem_path, "r") as f:
            keys = list(f.keys())
            if not keys:
                raise KeyError(f"No groups found in {mem_path}")
            try:
                chosen_label = _resolve_waveform_label(keys, waveform_label)
            except KeyError:
                if waveform_label is None:
                    raise
                _log.warning(
                    "%s: skipping event — requested waveform '%s' not present "
                    "(available: %s)",
                    event_name, waveform_label, keys,
                )
                n_missing_waveform += 1
                continue
            grp = f[chosen_label]
            a_sample = grp["A_sample"][()].real
            a_hat = grp["A_hat"][()].real
            a_sigma = grp["A_sigma"][()].real
            log_weight = grp["log_weight"][()].real
            _log.info(
                "%s: selected memory waveform '%s' (nsamp = %d)%s",
                event_name, chosen_label, len(a_sample),
                f" (available: {keys})" if len(keys) > 1 else "",
            )

        # --- Sample-level filter -------------------------------------------------
        # Bad samples are zeroed out in-place (log_weight -> -inf) rather than
        # removed, so the arrays stay the same length as the PE posterior.
        # generate_data indexes memory and posterior arrays with the same idxs,
        # so shortening the memory arrays would break that alignment.
        snr = np.abs(a_hat / a_sigma)
        bad = snr > _MAX_SAMPLE_SNR
        if bad.any():
            _log.warning(
                "%s: nullifying %d/%d samples with |A_hat/A_sigma| > %.1f "
                "(setting log_weight=-inf to exclude from importance sums)",
                event_name, int(bad.sum()), len(a_hat), _MAX_SAMPLE_SNR,
            )
            a_hat[bad] = 0.0
            a_sigma[bad] = 1.0
            log_weight[bad] = -np.inf
            a_sample[bad] = 0.0
            snr[bad] = 0.0

        # --- Event-level filter --------------------------------------------------
        finite_snr = snr[np.isfinite(snr)]
        median_snr = float(np.median(finite_snr)) if len(finite_snr) > 0 else 0.0
        if median_snr > _MAX_EVENT_MEDIAN_SNR:
            _log.warning(
                "%s: skipping event with median |A_hat/A_sigma| = %.2f > %.1f "
                "(waveform '%s')",
                event_name, median_snr, _MAX_EVENT_MEDIAN_SNR, chosen_label,
            )
            n_high_snr += 1
            continue

        memory_data.append({
            "A_sample": np.asarray(a_sample),
            "A_hat": np.asarray(a_hat),
            "A_sigma": np.asarray(a_sigma),
            "log_weight": np.asarray(log_weight),
            "event_name": event_name,
            "waveform_label": chosen_label,
        })

    n_total = len(event_files)
    n_loaded = len(memory_data)
    n_skipped = n_excluded + n_missing_waveform + n_high_snr
    skip_parts = []
    if n_excluded:
        skip_parts.append(f"{n_excluded} explicitly excluded")
    if n_missing_waveform:
        skip_parts.append(f"{n_missing_waveform} missing waveform")
    if n_high_snr:
        skip_parts.append(f"{n_high_snr} high memory SNR")
    skip_summary = f" ({', '.join(skip_parts)})" if skip_parts else ""
    if n_skipped:
        _log.warning(
            "load_memory_data: loaded %d/%d events; skipped %d%s",
            n_loaded, n_total, n_skipped, skip_summary,
        )
    else:
        _log.info(
            "load_memory_data: loaded %d/%d events",
            n_loaded, n_total,
        )

    return memory_data


def read_injection_file(
    vt_file,
    ifar_threshold=IFAR_THRESHOLD,
    semianalytic_snr_threshold=SEMIANALYTIC_SNR_THRESHOLD,
    min_detector_frame_total_mass=None,
    min_mass_ratio=None,
    min_mass_2_source=MIN_MASS_2_SOURCE,
):
    """Read an HDF5 injection/selection file and extract relevant data.

    Applies search-pipeline IFAR and semi-analytic observed-SNR cuts, plus
    detector-frame total-mass, mass-ratio, and minimum secondary-mass cuts
    when requested, to determine which injections were "found". It then
    extracts source-frame masses, spins, redshifts, and draw priors.
    Official polar-spin releases are converted from tilt angle to
    cos(tilt), while legacy Cartesian files retain the corresponding
    spin-coordinate Jacobian. Also computes derived spin quantities
    (chi_eff, chi_p) and converts the analysis time to years.

    Sources:
    - https://iopscience.iop.org/article/10.3847/2515-5172/ac2ba7
    - O4a-only injections:
      https://zenodo.org/records/16740117/preview/gwtc-4_o4a_sensitivity-estimates.md
    - O3+O4a cumulative injections:
      https://zenodo.org/records/16740128/preview/gwtc-4_o1234a_sensitivity-estimates.md

    Parameters
    ----------
    vt_file : str
        Path to the HDF5 injection file.
    ifar_threshold : float
        Inverse false-alarm rate threshold (yr); injections with min FAR
        below 1/ifar_threshold are considered found.
    semianalytic_snr_threshold : float or None
        Network observed-SNR threshold for semi-analytic injections. If the
        release contains ``semianalytic_observed_phase_maximized_snr_net``,
        injections above this threshold are also considered found. If None,
        this cut is disabled.
    min_detector_frame_total_mass : float or None
        Minimum detector-frame total mass threshold in solar masses.
        Injections with ``(m1_source + m2_source) * (1 + z)`` below this
        value are excluded. If None, this cut is disabled.
    min_mass_ratio : float or None
        Minimum mass-ratio threshold, where ``q = m2_source / m1_source``.
        Injections with ``q`` below this value are excluded. If None, this
        cut is disabled.
    min_mass_2_source : float or None
        Minimum source-frame secondary mass threshold in solar masses.
        Injections with ``m2_source`` below this value are excluded.
        If None, this cut is disabled.
    Returns
    -------
    dict
        Dictionary with arrays: 'mass_1_source', 'mass_ratio', 'redshift',
        'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'spin1z', 'spin2z',
        'chi_eff', 'chi_p', 'log_prior', 'found', 'total_generated',
        'analysis_time'.
    """
    injections = {}

    with h5py.File(vt_file, "r") as f:
        events = f["events"]
        fars = [events[key] for key in events.dtype.names if "far" in key]
        min_fars = np.min(fars, axis=0)
        found = min_fars < 1 / ifar_threshold
        if (
            semianalytic_snr_threshold is not None
            and "semianalytic_observed_phase_maximized_snr_net" in events.dtype.names
        ):
            semianalytic_snr = events[
                "semianalytic_observed_phase_maximized_snr_net"
            ][()]
            found |= semianalytic_snr >= semianalytic_snr_threshold
        redshift = _get_injection_redshift(events)
        detector_frame_total_mass = (
            (events["mass1_source"] + events["mass2_source"])
            * (1 + redshift)
        )
        if min_detector_frame_total_mass is not None:
            found &= detector_frame_total_mass >= min_detector_frame_total_mass
        mass_ratio = events["mass2_source"] / events["mass1_source"]
        if min_mass_ratio is not None:
            found &= mass_ratio >= min_mass_ratio
        if min_mass_2_source is not None:
            found &= events["mass2_source"] >= min_mass_2_source

        events = events[found]
        spin_data = _get_injection_spin_data(events)
        ln_prior, draw_coordinates = _get_injection_log_draw_prior(events)

        injections["mass_1_source"] = events["mass1_source"]
        injections["mass_ratio"] = (
            events["mass2_source"] / injections["mass_1_source"]
        )
        injections["redshift"] = _get_injection_redshift(events)
        injections["a_1"] = spin_data["a_1"]
        injections["a_2"] = spin_data["a_2"]
        injections["cos_tilt_1"] = spin_data["cos_tilt_1"]
        injections["cos_tilt_2"] = spin_data["cos_tilt_2"]
        injections["spin1z"] = spin_data["spin1z"]
        injections["spin2z"] = spin_data["spin2z"]

        # The population model is written in (m1, q, z, a1, a2, cos_tilt_1,
        # cos_tilt_2), so we always need the m2 -> q Jacobian. Legacy
        # Cartesian mixture files need the a^2 spin-coordinate Jacobian,
        # while official polar-spin releases need dtheta/dcos(theta)=1/sin(theta)
        # for each spin.
        log_jacobian = np.log(np.clip(injections["mass_1_source"], 1e-30, None))
        if draw_coordinates == "cartesian":
            log_jacobian += (
                2 * np.log(np.clip(injections["a_1"], 1e-30, None))
                + 2 * np.log(np.clip(injections["a_2"], 1e-30, None))
            )
        else:
            log_jacobian -= (
                np.log(np.clip(spin_data["sin_tilt_1"], 1e-30, None))
                + np.log(np.clip(spin_data["sin_tilt_2"], 1e-30, None))
            )

        log_prior = ln_prior + log_jacobian

        injections["log_prior"] = log_prior - np.log(events["weights"])

        q = injections["mass_ratio"]
        a1 = injections["a_1"]
        a2 = injections["a_2"]
        c1 = np.clip(injections["cos_tilt_1"], -1.0, 1.0)
        c2 = np.clip(injections["cos_tilt_2"], -1.0, 1.0)
        s1 = np.sin(np.arccos(c1))
        s2 = np.sin(np.arccos(c2))
        injections["chi_eff"] = (a1 * c1 + q * a2 * c2) / (1 + q)
        injections["chi_p"] = np.max(
            [a1 * s1, a2 * s2 * q * (4 * q + 3) / (4 + 3 * q)],
            axis=0,
        )
        injections["prior_effective_spin"] = injections["log_prior"]

        injections["found"] = found.sum()
        injections["total_generated"] = f.attrs["total_generated"]

        for key in "analysis_time", "total_analysis_time", "analysis_time_s":
            if key in f.attrs:
                injections["analysis_time"] = f.attrs[key]
        if "analysis_time" not in injections:
            logger.warning(
                "analysis_time not found in injection file; "
                "setting T_obs=1 yr as placeholder — "
                "absolute merger-rate R(z=0) will be WRONG"
            )
            injections["analysis_time"] = 1.0
        else:
            injections["analysis_time"] /= 60 * 60 * 24 * 365.25

    for key in injections:
        injections[key] = np.asarray(injections[key])

    return injections


def _compute_A_scale(memory_data, scale_tgr):
    """Return the TGR amplitude scale factor for normalisation.

    If *scale_tgr* is True, returns the pooled standard deviation of
    ``A_hat`` across all events (clamped to 1e-12).  Otherwise returns 1.
    """
    if scale_tgr:
        pooled = np.concatenate([md["A_hat"].ravel() for md in memory_data])
        return max(np.nanstd(pooled), 1e-12)
    return 1


def _sample_memory_event(md, idxs, A_scale):
    """Extract resampled memory arrays for one event.

    Parameters
    ----------
    md : dict
        Single-event memory data dict with keys ``A_hat``, ``A_sigma``,
        ``log_weight``.
    idxs : ndarray of int
        Sample indices (already drawn by the caller).
    A_scale : float
        Divisor applied to A_hat and A_sigma for normalisation.

    Returns
    -------
    A_hat, A_sigma, log_weight : ndarray
        Resampled arrays of length ``len(idxs)``.
    """
    return (
        md["A_hat"][idxs] / A_scale,
        md["A_sigma"][idxs] / A_scale,
        md["log_weight"][idxs],
    )


def _write_analyzed_events(ifar_cache_file, event_names):
    """Write *event_names* to ``analyzed_events.txt`` beside *ifar_cache_file*."""
    if ifar_cache_file is None:
        return
    path = os.path.join(
        os.path.dirname(os.path.abspath(ifar_cache_file)),
        "analyzed_events.txt",
    )
    with open(path, "w") as fh:
        fh.write(
            "# Final set of events used in hierarchical analysis"
            " (post IFAR/mass cuts)\n"
        )
        fh.writelines(name + "\n" for name in event_names)
    logger.info("Wrote analyzed event list to %s", path)


def generate_data(
    event_posteriors,
    injection_file,
    memory_data=None,
    use_tgr=True,
    ifar_threshold=IFAR_THRESHOLD,
    semianalytic_snr_threshold=SEMIANALYTIC_SNR_THRESHOLD,
    min_detector_frame_total_mass=None,
    min_mass_ratio=None,
    min_mass_2_source=MIN_MASS_2_SOURCE,
    N_samples=N_SAMPLES_PER_EVENT,
    prng=None,
    scale_tgr=False,
    ignore_memory_weights=False,
    event_names=None,
    ifar_cache_file=None,
):
    """Build per-event data arrays for the joint population model.

    Resamples posterior samples with importance weights, assembles arrays of
    (m1, q, spins, redshift, A_hat, A_sigma) per event.
    Also loads and processes the injection data for selection effects.

    Parameters
    ----------
    event_posteriors : list of structured ndarray
        Per-event posterior sample arrays with named fields.
    injection_file : str
        Path to the HDF5 injection/selection file.
    memory_data : list of dict or None
        Per-event memory data from `load_memory_data`.  Required when
        ``use_tgr=True``; ignored when ``use_tgr=False``.
    use_tgr : bool
        Whether to include the TGR amplitude in the returned data arrays.
    ifar_threshold : float
        IFAR threshold passed to `read_injection_file`.
    semianalytic_snr_threshold : float or None
        Semi-analytic observed-SNR threshold passed to `read_injection_file`.
    min_detector_frame_total_mass : float or None
        Minimum detector-frame total mass cut passed to
        `read_injection_file`. If None, the cut is disabled.
    min_mass_ratio : float or None
        Minimum mass-ratio cut passed to `read_injection_file`. If None,
        the cut is disabled.
    min_mass_2_source : float or None
        Minimum source-frame secondary mass (solar masses) applied to both
        injections and observed events. Events whose median posterior
        ``m2_source = m1_source * mass_ratio`` falls below this threshold
        are excluded with a warning. If None, the cut is disabled.
    N_samples : int
        Number of posterior samples to draw per event.
    prng : None, int, or numpy.random.Generator
        Random state for reproducible resampling.
    scale_tgr : bool
        If True, divide TGR parameter values by their pooled standard
        deviation across all events.
    ignore_memory_weights : bool
        If True, set all log_weights to zero (i.e. do not use the
        memory likelihood ratios as importance weights in the model).
        Useful for diagnosing the effect of the memory weights.
    event_names : list of str or None
        GW event names parallel to *event_posteriors*.  Required for the
        IFAR cut when ``use_tgr=False``; inferred from *memory_data* when
        ``use_tgr=True``.
    ifar_cache_file : str or None
        Path passed to `load_event_ifars` as the cache file.  IFARs are
        always fetched (from the cache if it exists, otherwise from GWOSC)
        and events whose IFAR falls below *ifar_threshold* are excluded,
        in addition to the injection-file cut on the selection function.
        When None, IFARs are fetched directly from GWOSC without caching.

    Returns
    -------
    tuple
        (event_data_array, injection_data_array, Nobs, Ndraw, A_scale)
    """
    if use_tgr and memory_data is None:
        raise ValueError(
            "memory_data is required when use_tgr=True"
        )

    Nobs = len(event_posteriors)

    logger.info("Using %d events", Nobs)

    # Construct the event posterior arrays
    m1s = []
    qs = []
    cost1s = []
    cost2s = []
    a1s = []
    a2s = []
    zs = []
    log_pdraw = []
    A_hats = []
    A_sigmas = []
    log_weights = []

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

    # Apply event-level cuts to mirror those applied to injections in
    # read_injection_file, so numerator and denominator use the same boundary.

    # Pre-fetch IFARs for all events (always applied, mirroring the injection cut).
    if use_tgr:
        _ifar_names = [memory_data[i]["event_name"]
                       for i in range(len(event_posteriors))]
    elif event_names is not None:
        _ifar_names = list(event_names)
    else:
        raise ValueError(
            "use_tgr=False and event_names is None: cannot apply IFAR cut "
            "on observed events. Pass event_names explicitly."
        )
    _event_ifars = load_event_ifars(_ifar_names, ifar_cache_file)

    if (
        min_detector_frame_total_mass is not None
        or min_mass_ratio is not None
        or min_mass_2_source is not None
        or _event_ifars is not None
    ):
        filtered_posteriors = []
        filtered_names = []
        filtered_memory = [] if use_tgr else None
        for i, ep in enumerate(event_posteriors):
            if use_tgr:
                event_label_pre = memory_data[i]["event_name"]
            elif event_names is not None:
                event_label_pre = event_names[i]
            else:
                event_label_pre = f"event {i}"
            excluded = False

            if not excluded and _event_ifars is not None:
                ev_ifar = _event_ifars.get(event_label_pre, float("inf"))
                if ev_ifar < ifar_threshold:
                    logger.warning(
                        "Excluding observed event %s: IFAR %.3g yr is below "
                        "threshold %.3g yr",
                        event_label_pre, ev_ifar, ifar_threshold,
                    )
                    excluded = True

            if min_mass_ratio is not None:
                median_q = float(np.median(ep["mass_ratio"]))
                if median_q < min_mass_ratio:
                    logger.warning(
                        "Excluding observed event %s: median mass ratio %.4f "
                        "is below the threshold %.4f",
                        event_label_pre, median_q, min_mass_ratio,
                    )
                    excluded = True

            if not excluded and min_mass_2_source is not None:
                m1 = np.asarray(ep["mass_1_source"], dtype=float)
                q = np.asarray(ep["mass_ratio"], dtype=float)
                median_m2 = float(np.median(m1 * q))
                if median_m2 < min_mass_2_source:
                    logger.warning(
                        "Excluding observed event %s: median source-frame "
                        "secondary mass %.2f Msun is below the threshold "
                        "%.2f Msun",
                        event_label_pre, median_m2, min_mass_2_source,
                    )
                    excluded = True

            if not excluded and min_detector_frame_total_mass is not None:
                m1 = np.asarray(ep["mass_1_source"], dtype=float)
                q = np.asarray(ep["mass_ratio"], dtype=float)
                z = np.asarray(ep["redshift"], dtype=float)
                median_m_det = float(np.median(m1 * (1.0 + q) * (1.0 + z)))
                if median_m_det < min_detector_frame_total_mass:
                    logger.warning(
                        "Excluding observed event %s: median detector-frame "
                        "total mass %.2f Msun is below the threshold %.2f Msun",
                        event_label_pre, median_m_det,
                        min_detector_frame_total_mass,
                    )
                    excluded = True

            if not excluded:
                filtered_posteriors.append(ep)
                filtered_names.append(event_label_pre)
                if use_tgr:
                    filtered_memory.append(memory_data[i])

        n_before = len(event_posteriors)
        event_posteriors = filtered_posteriors
        if use_tgr:
            memory_data = filtered_memory
        n_after = len(event_posteriors)
        if n_before != n_after:
            logger.warning(
                "generate_data: %d/%d observed events retained after "
                "IFAR / mass cuts (%d excluded)",
                n_after, n_before, n_before - n_after,
            )
        else:
            logger.info(
                "generate_data: all %d observed events passed IFAR / mass cuts",
                n_after,
            )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "generate_data: final event list (%d): %s",
                len(filtered_names), ", ".join(filtered_names),
            )
        _write_analyzed_events(ifar_cache_file, filtered_names)

    A_scale = _compute_A_scale(memory_data, scale_tgr and use_tgr)

    min_available = min(len(ep) for ep in event_posteriors)
    if N_samples > min_available:
        logger.warning(
            "N_samples=%d exceeds the number of available samples in the "
            "smallest event (%d); reducing N_samples to %d to avoid "
            "excessive duplication.",
            N_samples, min_available, min_available,
        )
        N_samples = min_available

    for i_event, event_posterior in enumerate(tqdm(event_posteriors)):
        # instead of picking the first N_samples, pick N_samples randomly
        # use this already to apply the weights (should be more efficient
        # than applying the weights after the fact, after trimming the
        # samples)
        if "weights" in event_posterior.dtype.names:
            w = event_posterior["weights"]
        else:
            w = np.ones(len(event_posterior))

        if use_tgr:
            md = memory_data[i_event]
            if len(md["A_hat"]) != len(event_posterior):
                raise ValueError(
                    f"Memory data length ({len(md['A_hat'])}) does not "
                    f"match posterior length ({len(event_posterior)}) for "
                    f"event {md['event_name']}"
                )
            # log_weight is NOT used for resampling here; it is passed to the
            # model as an explicit per-sample additive term in the log
            # probability, avoiding degenerate resampling when the weights are
            # concentrated.

        neff = np.sum(w) ** 2 / np.sum(w**2)
        event_label = (
            memory_data[i_event]["event_name"] if use_tgr
            else f"event {i_event}"
        )
        d_full = 7
        if neff < d_full + 1:
            logger.warning(
                "Skipping event %s: effective sample size %.1f is too low "
                "to compute a non-singular covariance matrix (need > %d)",
                event_label, neff, d_full,
            )
            continue
        if neff < N_samples:
            logger.warning(
                "Effective sample size %.1f < %d requested samples for event %s",
                neff, N_samples, event_label,
            )

        idxs = prng.choice(len(event_posterior), size=N_samples,
                           replace=True, p=w/w.sum())

        m1s.append(event_posterior["mass_1_source"][idxs])
        qs.append(event_posterior["mass_ratio"][idxs])

        a1s.append(event_posterior["a_1"][idxs])
        a2s.append(event_posterior["a_2"][idxs])

        if use_tgr:
            a_hat_i, a_sig_i, lw_i = _sample_memory_event(
                memory_data[i_event], idxs, A_scale
            )
            if np.all(np.isnan(a_hat_i)):
                logger.warning(
                    "Skipping event %s: all A_hat samples are NaN "
                    "(broken memory results for waveform '%s')",
                    event_label, memory_data[i_event]["waveform_label"],
                )
                m1s.pop()
                qs.pop()
                a1s.pop()
                a2s.pop()
                continue
            if ignore_memory_weights:
                lw_i = np.zeros(N_samples)
        else:
            a_hat_i = a_sig_i = lw_i = np.zeros(N_samples)
        A_hats.append(a_hat_i)
        A_sigmas.append(a_sig_i)
        log_weights.append(lw_i)

        cost1s.append(event_posterior["cos_tilt_1"][idxs])
        cost2s.append(event_posterior["cos_tilt_2"][idxs])
        zs.append(event_posterior["redshift"][idxs])
        if "log_prior" in event_posterior.dtype.names:
            log_pdraw.append(event_posterior["log_prior"][idxs])
        elif "prior" in event_posterior.dtype.names:
            log_pdraw.append(np.log(np.clip(event_posterior["prior"][idxs], 1e-300, None)))
        else:
            raise KeyError(
                "Posterior samples must contain either 'log_prior' or 'prior' "
                "for reweighting."
            )


    Nobs = len(m1s)
    n_dropped_in_loop = len(event_posteriors) - Nobs
    if n_dropped_in_loop:
        logger.warning(
            "generate_data: %d/%d events dropped during sample assembly "
            "(low ESS, all-NaN A_hat, or singular covariance)",
            n_dropped_in_loop, len(event_posteriors),
        )
    logger.info("generate_data: %d events entering the likelihood", Nobs)

    event_data_array = np.array(
        [m1s, qs, cost1s, cost2s, a1s, a2s, A_hats, A_sigmas, zs, log_pdraw, log_weights]
    )

    injection_data = read_injection_file(
        injection_file,
        ifar_threshold=ifar_threshold,
        semianalytic_snr_threshold=semianalytic_snr_threshold,
        min_detector_frame_total_mass=min_detector_frame_total_mass,
        min_mass_ratio=min_mass_ratio,
        min_mass_2_source=min_mass_2_source,
    )
    Ndraw = int(injection_data["total_generated"])

    injection_data_array = np.array(
        [
            injection_data["mass_1_source"],
            injection_data["mass_ratio"],
            injection_data["cos_tilt_1"],
            injection_data["cos_tilt_2"],
            injection_data["a_1"],
            injection_data["a_2"],
            injection_data["redshift"],
            injection_data["log_prior"]
            - np.log(injection_data["analysis_time"]),  # p_draw/T_obs so exp(log_sel) = T_obs*beta(Λ) = VT
        ]
    )

    return (
        event_data_array,
        injection_data_array,
        Nobs,
        Ndraw,
        A_scale,
    )


def generate_tgr_only_data(event_posteriors, memory_data,
                           N_samples=N_SAMPLES_PER_EVENT, prng=None, scale_tgr=False,
                           ignore_memory_weights=False,
                           ifar_threshold=IFAR_THRESHOLD, ifar_cache_file=None):
    """Build simplified data arrays for the TGR-only model.

    Resamples posterior indices using memory importance weights and extracts
    per-sample ``A_hat`` and ``A_sigma`` for analytic Gaussian convolution.

    Parameters
    ----------
    event_posteriors : list of structured ndarray
        Per-event posterior sample arrays with named fields.
    memory_data : list of dict
        Per-event memory data from `load_memory_data`.  The ``A_hat``,
        ``A_sigma``, and importance weights (``log_weight``) are taken
        from these dicts.
    N_samples : int
        Number of posterior samples to draw per event.
    prng : None, int, or numpy.random.Generator
        Random state for reproducible resampling.
    scale_tgr : bool
        If True, divide TGR parameter values by their pooled standard
        deviation across all events.
    ifar_threshold : float
        Minimum IFAR (years) for an event to be included.  Always applied.
    ifar_cache_file : str or None
        Path passed to `load_event_ifars` as the cache file.  IFARs are
        always fetched (from the cache if it exists, otherwise from GWOSC)
        and events whose IFAR falls below *ifar_threshold* are excluded.
        When None, IFARs are fetched directly from GWOSC without caching.

    Returns
    -------
    tuple
        (A_hats, A_sigmas, log_weights, Nobs, A_scale) where A_hats,
        A_sigmas, and log_weights have shape (Nobs, N_samples).
        log_weights are the per-sample memory log-likelihood ratios to be
        included as additive terms in the model's log probability.
    """
    # Apply IFAR cut, filtering both memory_data and event_posteriors in
    # lockstep (always applied, mirroring the injection cut).
    names = [md["event_name"] for md in memory_data]
    event_ifars = load_event_ifars(names, ifar_cache_file)
    kept_memory = []
    kept_posteriors = []
    for md, ep in zip(memory_data, event_posteriors):
        ev_ifar = event_ifars.get(md["event_name"], float("inf"))
        if ev_ifar < ifar_threshold:
            logger.warning(
                "Excluding observed event %s: IFAR %.3g yr is below "
                "threshold %.3g yr",
                md["event_name"], ev_ifar, ifar_threshold,
            )
        else:
            kept_memory.append(md)
            kept_posteriors.append(ep)
    memory_data = kept_memory
    event_posteriors = kept_posteriors

    Nobs = len(event_posteriors)

    logger.info("Using %d events", Nobs)
    kept_names = [md["event_name"] for md in memory_data]
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "generate_tgr_only_data: final event list (%d): %s",
            Nobs, ", ".join(kept_names),
        )
    _write_analyzed_events(ifar_cache_file, kept_names)

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

    A_scale = _compute_A_scale(memory_data, scale_tgr)

    # Construct the event posterior arrays.
    # Resample uniformly — log_weight is passed to the model explicitly
    # rather than being baked into the resampling distribution.
    A_hats = []
    A_sigmas = []
    log_weights = []
    for md in memory_data:
        idxs = prng.choice(len(md["A_hat"]), size=N_samples, replace=True)
        a_hat_i, a_sig_i, lw_i = _sample_memory_event(md, idxs, A_scale)
        if np.all(np.isnan(a_hat_i)):
            logger.warning(
                "Skipping event %s: all A_hat samples are NaN "
                "(broken memory results for waveform '%s')",
                md["event_name"], md["waveform_label"],
            )
            continue
        if ignore_memory_weights:
            lw_i = np.zeros(N_samples)
        A_hats.append(a_hat_i)
        A_sigmas.append(a_sig_i)
        log_weights.append(lw_i)

    A_hats = np.array(A_hats)
    A_sigmas = np.array(A_sigmas)
    log_weights = np.array(log_weights)

    return A_hats, A_sigmas, log_weights, len(A_hats), A_scale
