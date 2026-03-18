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
N_SAMPLES_PER_EVENT = 10000
MIN_DETECTOR_FRAME_TOTAL_MASS = 66.0
MIN_MASS_RATIO = 1.0 / 6.0


def _waveform_sort_key(label):
    """Sort waveform labels by descending calibration, then alphabetically."""
    match = re.match(r"C(\d+):", label)
    calibration = int(match.group(1)) if match else -1
    return (-calibration, label)


def _pick_waveform_label(keys):
    """Pick the best available waveform label using the priority hierarchy.

    Priority: NRSur > SEOB > IMRPhenom (any remaining label).
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
    return keys_sorted[0]


def _resolve_waveform_label(keys, waveform=None):
    """Resolve *waveform* to the best matching label in *keys*.

    ``None`` or ``"auto"`` uses the default NRSur > SEOB > IMRPhenom
    hierarchy. A bare waveform name like ``"NRSur7dq4"`` selects the
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
    for event_file in event_files:
        basename = os.path.basename(event_file)
        match = re.search(r"(GW\d{6}_\d{6})", basename)
        if match is None:
            raise ValueError(
                f"Could not extract event name from filename: {basename}"
            )
        event_name = match.group(1)

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
                _log.info(
                    "%s: skipping event because waveform '%s' is not present in %s",
                    event_name, waveform_label, keys,
                )
                continue
            _log.info(
                "%s: selected memory waveform '%s'%s",
                event_name, chosen_label,
                f" (available: {keys})" if len(keys) > 1 else "",
            )

            grp = f[chosen_label]
            a_sample = grp["A_sample"][()].real
            a_hat = grp["A_hat"][()].real
            a_sigma = grp["A_sigma"][()].real
            log_weight = grp["log_weight"][()].real

        memory_data.append({
            "A_sample": np.asarray(a_sample),
            "A_hat": np.asarray(a_hat),
            "A_sigma": np.asarray(a_sigma),
            "log_weight": np.asarray(log_weight),
            "event_name": event_name,
            "waveform_label": chosen_label,
        })

    return memory_data


def read_injection_file(
    vt_file,
    ifar_threshold=IFAR_THRESHOLD,
    min_detector_frame_total_mass=MIN_DETECTOR_FRAME_TOTAL_MASS,
    min_mass_ratio=MIN_MASS_RATIO,
):
    """Read an HDF5 injection/selection file and extract relevant data.

    Applies IFAR and, when requested, detector-frame total-mass and
    mass-ratio cuts to determine which injections were "found", then extracts
    source-frame
    masses, spins, redshifts, and draw priors. Also computes derived
    spin quantities (chi_eff, chi_p) and converts the analysis time to
    years.

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
    min_detector_frame_total_mass : float or None
        Minimum detector-frame total mass threshold in solar masses.
        Injections with ``(m1_source + m2_source) * (1 + z)`` below this
        value are excluded. If None, this cut is disabled.
    min_mass_ratio : float or None
        Minimum mass-ratio threshold, where ``q = m2_source / m1_source``.
        Injections with ``q`` below this value are excluded. If None, this
        cut is disabled.
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
        detector_frame_total_mass = (
            (events["mass1_source"] + events["mass2_source"])
            * (1 + events["redshift"])
        )
        if min_detector_frame_total_mass is not None:
            found &= detector_frame_total_mass >= min_detector_frame_total_mass
        mass_ratio = events["mass2_source"] / events["mass1_source"]
        if min_mass_ratio is not None:
            found &= mass_ratio >= min_mass_ratio

        events = events[found]

        injections["mass_1_source"] = events["mass1_source"]
        injections["mass_ratio"] = (
            events["mass2_source"] / injections["mass_1_source"]
        )
        injections["redshift"] = events["redshift"]
        injections["a_1"] = (
            events["spin1x"] ** 2
            + events["spin1y"] ** 2
            + events["spin1z"] ** 2
        ) ** 0.5
        injections["a_2"] = (
            events["spin2x"] ** 2
            + events["spin2y"] ** 2
            + events["spin2z"] ** 2
        ) ** 0.5
        injections["cos_tilt_1"] = events["spin1z"] / injections["a_1"]
        injections["cos_tilt_2"] = events["spin2z"] / injections["a_2"]

        injections["spin1z"] = events["spin1z"]
        injections["spin2z"] = events["spin2z"]

        ln_prior = events[
            "lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
        ]

        # The draw prior density (ln_prior) is defined w.r.t. Cartesian spin
        # coordinates: p_draw(m1, m2, z, s1x, s1y, s1z, s2x, s2y, s2z).
        # The population model uses spherical spin parameters (a, cos_tilt, phi)
        # or aligned-spin (chi_z, cos_tilt, phi). To convert the draw prior to the
        # marginalized parameter space, we need to:
        #   The Jacobian for (sx, sy, sz) -> (a, cos_tilt, phi) is a^2,
        #   so we add 2*log(a) per spin to convert the draw prior to the
        #   spherical-spin parameter space used by the model.
        log_jacobian = (
            2 * np.log(np.clip(injections["a_1"], 1e-30, None)) +
            2 * np.log(np.clip(injections["a_2"], 1e-30, None))
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
            logger.warning("analysis_time not found in injection file")
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


def generate_data(
    event_posteriors,
    injection_file,
    memory_data=None,
    use_tgr=True,
    ifar_threshold=IFAR_THRESHOLD,
    min_detector_frame_total_mass=MIN_DETECTOR_FRAME_TOTAL_MASS,
    min_mass_ratio=MIN_MASS_RATIO,
    N_samples=N_SAMPLES_PER_EVENT,
    prng=None,
    scale_tgr=False,
    ignore_memory_weights=False,
):
    """Build per-event data arrays for the joint population model.

    Resamples posterior samples with importance weights, assembles arrays of
    (m1, q, spins, redshift, A_hat, A_sigma) per event, and computes KDE
    bandwidth matrices via the conditional covariance of the spin dimensions.
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
        Whether to include the TGR parameter in the KDE.
    ifar_threshold : float
        IFAR threshold passed to `read_injection_file`.
    min_detector_frame_total_mass : float or None
        Minimum detector-frame total mass cut passed to
        `read_injection_file`. If None, the cut is disabled.
    min_mass_ratio : float or None
        Minimum mass-ratio cut passed to `read_injection_file`. If None,
        the cut is disabled.
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

    Returns
    -------
    tuple
        (event_data_array, injection_data_array, BW_matrices,
        BW_matrices_sel, Nobs, Ndraw, A_scale)
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

    BW_matrices = []
    BW_matrices_sel = []

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

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

        # BW_matrices are always 2x2 (spins only); the TGR dimension
        # is handled analytically in the model.
        d = 2
        data_array = np.array(
            [
                a1s[-1],
                a2s[-1],
                m1s[-1],
                qs[-1],
                zs[-1],
                cost1s[-1],
                cost2s[-1],
            ]
        )

        full_cov_i = np.cov(data_array)
        try:
            prec_i = np.linalg.inv(full_cov_i)[:d, :d]
            cov_i = np.linalg.inv(prec_i)
        except np.linalg.LinAlgError:
            logger.warning(
                "Skipping event %s: singular covariance matrix despite ESS=%.1f",
                event_label, neff,
            )
            m1s.pop()
            qs.pop()
            a1s.pop()
            a2s.pop()
            A_hats.pop()
            A_sigmas.pop()
            log_weights.pop()
            cost1s.pop()
            cost2s.pop()
            zs.pop()
            log_pdraw.pop()
            continue

        BW_matrices.append(cov_i * N_samples ** (-2.0 / (4 + d)))
        BW_matrices_sel.append(cov_i[:2, :2] * N_samples ** (-2.0 / 6))

    BW_matrices = np.array(BW_matrices)
    BW_matrices_sel = np.array(BW_matrices_sel)

    Nobs = len(m1s)

    event_data_array = np.array(
        [m1s, qs, cost1s, cost2s, a1s, a2s, A_hats, A_sigmas, zs, log_pdraw, log_weights]
    )

    injection_data = read_injection_file(
        injection_file,
        ifar_threshold=ifar_threshold,
        min_detector_frame_total_mass=min_detector_frame_total_mass,
        min_mass_ratio=min_mass_ratio,
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
            injection_data["log_prior"],
        ]
    )

    return (
        event_data_array,
        injection_data_array,
        BW_matrices,
        BW_matrices_sel,
        Nobs,
        Ndraw,
        A_scale,
    )


def generate_tgr_only_data(event_posteriors, memory_data,
                           N_samples=N_SAMPLES_PER_EVENT, prng=None, scale_tgr=False,
                           ignore_memory_weights=False):
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

    Returns
    -------
    tuple
        (A_hats, A_sigmas, log_weights, Nobs, A_scale) where A_hats,
        A_sigmas, and log_weights have shape (Nobs, N_samples).
        log_weights are the per-sample memory log-likelihood ratios to be
        included as additive terms in the model's log probability.
    """
    Nobs = len(event_posteriors)

    logger.info("Using %d events", Nobs)

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
        if ignore_memory_weights:
            lw_i = np.zeros(N_samples)
        A_hats.append(a_hat_i)
        A_sigmas.append(a_sig_i)
        log_weights.append(lw_i)

    A_hats = np.array(A_hats)
    A_sigmas = np.array(A_sigmas)
    log_weights = np.array(log_weights)

    return A_hats, A_sigmas, log_weights, Nobs, A_scale
