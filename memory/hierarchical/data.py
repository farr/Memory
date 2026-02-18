"""Data loading and preparation for hierarchical TGR population analysis."""

import os
import re

import numpy as np
import h5py
import bilby
from tqdm import tqdm

align_spin_prior = bilby.gw.prior.AlignedSpin()


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
        HDF5 group name inside each memory file.  If None, the first
        available group is used.

    Returns
    -------
    list of dict
        One dict per event with keys ``'A_sample'`` (1-D float array),
        ``'log_weight'`` (1-D float array), and ``'event_name'`` (str).

    Raises
    ------
    FileNotFoundError
        If a memory results file cannot be found for an event.
    KeyError
        If the requested waveform label is not present in the file.
    """
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
            if waveform_label is not None:
                if waveform_label not in f:
                    raise KeyError(
                        f"Waveform label '{waveform_label}' not found in "
                        f"{mem_path}; available: {list(f.keys())}"
                    )
                grp = f[waveform_label]
            else:
                keys = list(f.keys())
                if not keys:
                    raise KeyError(f"No groups found in {mem_path}")
                grp = f[keys[0]]

            a_sample = grp["A_sample"][()]
            log_weight = grp["log_weight"][()]

        memory_data.append({
            "A_sample": np.asarray(a_sample),
            "log_weight": np.asarray(log_weight),
            "event_name": event_name,
        })

    return memory_data


def read_injection_file(
    vt_file, ifar_threshold=1000, use_tilts=False, snr_inspiral_cut=0, snr_cut=0
):
    """Read an HDF5 injection/selection file and extract relevant data.

    Applies IFAR and SNR cuts to determine which injections were "found",
    then extracts source-frame masses, spins, redshifts, and draw priors.
    Also computes derived spin quantities (chi_eff, chi_p) and converts
    the analysis time to years.

    Sources:
    - https://iopscience.iop.org/article/10.3847/2515-5172/ac2ba7
    - https://zenodo.org/records/16740117/preview/gwtc-4_o4a_sensitivity-estimates.md

    Parameters
    ----------
    vt_file : str
        Path to the HDF5 injection file.
    ifar_threshold : float
        Inverse false-alarm rate threshold (yr); injections with min FAR
        below 1/ifar_threshold are considered found.
    use_tilts : bool
        If False, reweight the draw prior by the aligned-spin prior
        (marginalizing over tilt angles).
    snr_inspiral_cut : float
        Inspiral SNR threshold; injections above this are also considered found.
    snr_cut : float
        Network optimal SNR threshold; injections above this are also
        considered found.

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

        snrs = events["estimated_optimal_snr_net"]

        if snr_cut > 0:
            found = found | (snrs > snr_cut)

        if snr_inspiral_cut > 0:
            snrs_inspiral = (
                1.1
                - 0.9
                * (events["mass1_source"] + events["mass2_source"])
                * (1 + events["redshift"])
                / 100
            ) * snrs
            found = found | (snrs_inspiral > snr_inspiral_cut)

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
        # The population model uses spherical spin parameters (a, cos_tilt)
        # or aligned-spin (chi_z). To convert the draw prior to the
        # marginalized parameter space, we need to:
        #   1. Factor out the Cartesian spin density, which for isotropic
        #      uniform spins is p_Cart(sx, sy, sz) = 1/(4*pi*a^2). This
        #      amounts to adding 2*log(a) per spin (the 4*pi is constant
        #      and cancels in importance-weight ratios).
        #   2. For use_tilts=False, add the aligned-spin marginal draw
        #      prior p_aligned(chi_z) = -log|chi_z|/2, since the model
        #      describes chi_z = a*cos(tilt) directly.
        log_jacobian = (
            2 * np.log(np.clip(injections["a_1"], 1e-30, None))
            + 2 * np.log(np.clip(injections["a_2"], 1e-30, None))
        )

        if use_tilts:
            log_prior = ln_prior + log_jacobian
        else:
            log_prior = (
                ln_prior
                + log_jacobian
                + np.log(align_spin_prior.prob(events["spin1z"]))
                + np.log(align_spin_prior.prob(events["spin2z"]))
            )

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
            print("analysis_time not found")
        else:
            injections["analysis_time"] /= 60 * 60 * 24 * 365.25

    for key in injections:
        injections[key] = np.asarray(injections[key])

    return injections


def generate_data(
    event_posteriors,
    injection_file,
    memory_data,
    use_tgr=True,
    use_tilts=False,
    ifar_threshold=1000,
    N_samples=2000,
    snr_cut=0,
    snr_inspiral_cut=0,
    prng=None,
    scale_tgr=False,
):
    """Build per-event data arrays for the joint population model.

    Resamples posterior samples with importance weights, assembles arrays of
    (m1, q, spins, redshift, TGR parameter) per event, and computes KDE
    bandwidth matrices via the conditional covariance of the spin/TGR
    dimensions. Also loads and processes the injection data for selection
    effects.

    Parameters
    ----------
    event_posteriors : list of structured ndarray
        Per-event posterior sample arrays with named fields.
    injection_file : str
        Path to the HDF5 injection/selection file.
    memory_data : list of dict
        Per-event memory data from `load_memory_data`.  The TGR parameter
        values (``A_sample``) and importance weights (``log_weight``) are
        taken from these dicts.
    use_tgr : bool
        Whether to include the TGR parameter in the KDE.
    use_tilts : bool
        Whether to include tilt angles in the data arrays.
    ifar_threshold : float
        IFAR threshold passed to `read_injection_file`.
    N_samples : int
        Number of posterior samples to draw per event.
    snr_cut : float
        Network SNR cut passed to `read_injection_file`.
    snr_inspiral_cut : float
        Inspiral SNR cut passed to `read_injection_file`.
    prng : None, int, or numpy.random.Generator
        Random state for reproducible resampling.
    scale_tgr : bool
        If True, divide TGR parameter values by their pooled standard
        deviation across all events.

    Returns
    -------
    tuple
        (event_data_array, injection_data_array, BW_matrices,
        BW_matrices_sel, Nobs, Ndraw, dphi_scale)
    """
    Nobs = len(event_posteriors)

    print(f"Using {Nobs} events!")

    # Construct the event posterior arrays
    m1s = []
    qs = []
    cost1s = []
    cost2s = []
    a1s = []
    a2s = []
    zs = []
    log_pdraw = []
    dphis = []
    kde_weights = []

    BW_matrices = []
    BW_matrices_sel = []

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

    if scale_tgr:
        pooled_phi = np.concatenate([
            md["A_sample"].ravel() for md in memory_data
        ])
        dphi_scale = max(np.nanstd(pooled_phi), 1e-12)
    else:
        dphi_scale = 1

    for i_event, event_posterior in enumerate(tqdm(event_posteriors)):
        md = memory_data[i_event]

        # instead of picking the first N_samples, pick N_samples randomly
        # use this already to apply the weights (should be more efficient
        # than applying the weights after the fact, after trimming the
        # samples)
        if "weights" in event_posterior.dtype.names:
            w = event_posterior["weights"]
        else:
            w = np.ones(len(event_posterior))

        if len(md["A_sample"]) != len(event_posterior):
            raise ValueError(
                f"Memory data length ({len(md['A_sample'])}) does not "
                f"match posterior length ({len(event_posterior)}) for "
                f"event {md['event_name']}"
            )
        log_w = np.log(np.clip(w, 1e-300, None)) + md["log_weight"]
        log_w -= log_w.max()
        w = np.exp(log_w)

        idxs = prng.choice(len(event_posterior), size=N_samples,
                           replace=True, p=w/w.sum())

        # TODO: check sampling efficiency by computing the effective sample size
        # and comparing it to the number of samples drawn
        neff = np.sum(w) ** 2 / np.sum(w**2)
        if neff < N_samples:
            # warn too few samples
            print(
                f"Warning: effective sample size {neff} is less than the number "
                f"of samples drawn {N_samples} for event "
                f"{event_posterior['event_name']}"
            )

        m1s.append(event_posterior["mass_1_source"][idxs])
        qs.append(event_posterior["mass_ratio"][idxs])

        a1s.append(event_posterior["a_1"][idxs])
        a2s.append(event_posterior["a_2"][idxs])

        dphis.append(md["A_sample"][idxs] / dphi_scale)

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

        if use_tgr:
            d = 3
            if use_tilts:
                data_array = np.array(
                    [
                        a1s[-1],
                        a2s[-1],
                        dphis[-1],
                        m1s[-1],
                        qs[-1],
                        zs[-1],
                        cost1s[-1],
                        cost2s[-1],
                    ]
                )
            else:
                data_array = np.array(
                    [a1s[-1], a2s[-1], dphis[-1], m1s[-1], qs[-1], zs[-1]]
                )
        else:
            d = 2
            if use_tilts:
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
            else:
                data_array = np.array(
                    [a1s[-1], a2s[-1], m1s[-1], qs[-1], zs[-1]]
                )

        # could have applied the weights here instead
        weights_i = np.ones(N_samples)

        kde_weights.append(weights_i)

        N_eff = np.sum(weights_i) ** 2 / np.sum(weights_i**2)

        full_cov_i = np.cov(data_array, aweights=weights_i)
        prec_i = np.linalg.inv(full_cov_i)[:d, :d]
        cov_i = np.linalg.inv(prec_i)

        BW_matrices.append(cov_i * N_eff ** (-2.0 / (4 + d)))
        BW_matrices_sel.append(cov_i[:2, :2] * N_eff ** (-2.0 / (6)))

    BW_matrices = np.array(BW_matrices)
    BW_matrices_sel = np.array(BW_matrices_sel)

    event_data_array = np.array(
        [m1s, qs, cost1s, cost2s, a1s, a2s, dphis, zs, log_pdraw, kde_weights]
    )

    injection_data = read_injection_file(
        injection_file,
        ifar_threshold=ifar_threshold,
        snr_cut=snr_cut,
        snr_inspiral_cut=snr_inspiral_cut,
        use_tilts=use_tilts,
    )
    Ndraw = int(injection_data["total_generated"])

    # Construct the injection arrays
    if use_tilts:
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
    else:
        injection_data_array = np.array(
            [
                injection_data["mass_1_source"],
                injection_data["mass_ratio"],
                injection_data["cos_tilt_1"],
                injection_data["cos_tilt_2"],
                injection_data["spin1z"],
                injection_data["spin2z"],
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
        dphi_scale,
    )


def generate_tgr_only_data(event_posteriors, memory_data,
                           N_samples=2000, prng=None, scale_tgr=False):
    """Build simplified data arrays for the TGR-only model.

    Resamples the memory amplitude from each event's memory results and
    computes per-event 1D KDE bandwidths using Silverman's rule.

    Parameters
    ----------
    event_posteriors : list of structured ndarray
        Per-event posterior sample arrays with named fields.
    memory_data : list of dict
        Per-event memory data from `load_memory_data`.  The TGR parameter
        values (``A_sample``) and importance weights (``log_weight``) are
        taken from these dicts.
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
        (dphis, bws_tgr, Nobs, dphi_scale) where dphis is shape
        (Nobs, N_samples), bws_tgr is shape (Nobs,).
    """
    Nobs = len(event_posteriors)

    print(f"Using {Nobs} events!")

    if prng is None:
        prng = np.random.default_rng(np.random.randint(1 << 32))
    elif isinstance(prng, int):
        prng = np.random.default_rng(prng)

    if scale_tgr:
        pooled_phi = np.concatenate([
            md["A_sample"].ravel() for md in memory_data
        ])
        dphi_scale = max(np.nanstd(pooled_phi), 1e-12)
    else:
        dphi_scale = 1

    # Construct the event posterior arrays
    dphis = []
    bws_tgr = []
    for md in memory_data:
        w = np.exp(md["log_weight"] - md["log_weight"].max())
        w /= w.sum()
        idxs = prng.choice(len(md["A_sample"]), size=N_samples,
                           replace=True, p=w)
        dphis.append(md["A_sample"][idxs] / dphi_scale)

        bws_tgr.append(
            np.std(dphis[-1]) * N_samples ** (-1.0 / 5)
        )

    bws_tgr = np.array(bws_tgr)
    dphis = np.array(dphis)

    return dphis, bws_tgr, Nobs, dphi_scale
