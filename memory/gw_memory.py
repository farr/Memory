import copy
import scipy
import numpy as np
import gwsurrogate
import spherical_functions
from scipy.integrate import cumulative_trapezoid as cumtrapz

import lal
import lalsimulation as lalsim

from memory.gw_residuals import _ensure_bilby_calibration_keys


def evaluate_surrogate_with_LAL(sample, res, approximant=lalsim.NRSur7dq4, ell_max=4):
    """Evaluate NRSur7dq4 using LALSimulation and return spherical-harmonic modes.

    This function calls `lalsimulation.SimInspiralChooseTDModes`
    with the NRSur7dq4 approximant and returns the time-domain
    spin-weighted spherical-harmonic modes.

    Parameters
    ----------
    sample : dict
        Dictionary of source parameters containing:
        - 'mass_1', 'mass_2' : float
            Component masses in solar masses.
        - 'spin_1x', 'spin_1y', 'spin_1z' : float
            Dimensionless spin components of the primary.
        - 'spin_2x', 'spin_2y', 'spin_2z' : float
            Dimensionless spin components of the secondary.
        - 'luminosity_distance' : float
            Luminosity distance in Mpc.
        - 'phase' : float
            Reference orbital phase in radians.
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - minimum_frequency : dict
            Dictionary of low-frequency cutoffs in Hz.
        - reference_frequency : float
            Reference frequency in Hz.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    ell_max : int, optional
        Maximum ℓ mode included in the waveform model.
        Default is 4.

    Returns
    -------
    h_modes : dict
        Dictionary of complex waveform modes {(ℓ, m): h_ℓm(t)}.
        Each value is a NumPy array.
    t : ndarray
        Time array in seconds corresponding to the modes.
    """
    mass_1 = sample["mass_1"] * lal.MSUN_SI
    mass_2 = sample["mass_2"] * lal.MSUN_SI

    s1x = sample["spin_1x"]
    s1y = sample["spin_1y"]
    s1z = sample["spin_1z"]

    s2x = sample["spin_2x"]
    s2y = sample["spin_2y"]
    s2z = sample["spin_2z"]

    distance = sample["luminosity_distance"] * 1e6 * lal.PC_SI

    phiRef = sample["phase"]

    fs = res["config"].sampling_frequency
    deltaT = 1 / fs

    f_low = 0.8 * res["config"].minimum_frequency[res["ifos"][0].name]
    f_ref = res["config"].reference_frequency

    if approximant == lalsim.NRSur7dq4:
        f_low = -1
        
    h_modes_lal = lalsim.SimInspiralChooseTDModes(
        phiRef,
        deltaT,
        mass_1,
        mass_2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        f_low,
        f_ref,
        distance,
        None,
        ell_max,
        approximant,
    )

    h_modes = {}

    for L in range(2, ell_max + 1):
        for M in range(-L, L + 1):
            h_modes[(L, M)] = lalsim.SphHarmTimeSeriesGetMode(
                h_modes_lal, L, M
            ).data.data

    t = np.arange(len(h_modes[(2, 2)])) * deltaT

    t += float(lalsim.SphHarmTimeSeriesGetMode(h_modes_lal, 2, 2).epoch)

    return h_modes, t


def evaluate_surrogate_with_LAL_as_polarizations(sample, res, approximant=lalsim.NRSur7dq4, ell_max=4):
    """Evaluate NRSur7dq4 using LALSimulation and return polarizations.

    This function calls `lalsimulation.SimInspiralChooseTDWaveform`
    with the NRSur7dq4 approximant and returns the plus and cross
    polarizations.

    Parameters
    ----------
    sample : dict
        Dictionary of source parameters containing masses (solar masses),
        spins (dimensionless), luminosity distance (Mpc),
        inclination angle `theta_jn` (radians), and phase (radians).
    res : dict
        Result dictionary containing:
        - 'ifos' : list
            Interferometer objects.
        - 'config' : object
            Configuration object.
        - 'samples' : list of dict
            Posterior samples.
        - 'fd' : dict
            Frequency-domain waveform data used for sizing arrays.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    ell_max : int, optional
        Maximum ℓ mode included internally in the waveform model.
        Default is 4.

    Returns
    -------
    hp : lal.REAL8TimeSeries
        Plus polarization time series.
    hc : lal.REAL8TimeSeries
        Cross polarization time series.
    """
    mass_1 = sample["mass_1"] * lal.MSUN_SI
    mass_2 = sample["mass_2"] * lal.MSUN_SI

    s1x = sample["spin_1x"]
    s1y = sample["spin_1y"]
    s1z = sample["spin_1z"]

    s2x = sample["spin_2x"]
    s2y = sample["spin_2y"]
    s2z = sample["spin_2z"]

    distance = sample["luminosity_distance"] * 1e6 * lal.PC_SI

    inclination = sample["theta_jn"]
    phiRef = sample["phase"]

    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0

    fs = res["config"].sampling_frequency
    deltaT = 1 / fs

    f_low = res["config"].minimum_frequency[res["ifos"][0].name]
    f_ref = res["config"].reference_frequency

    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        mass_1,
        mass_2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        distance,
        inclination,
        phiRef,
        longAscNodes,
        eccentricity,
        meanPerAno,
        deltaT,
        f_low,
        f_ref,
        lal.CreateDict(),
        approximant,
    )

    return hp, hc


def map_modes_to_polarizations(modes, t, sample, longAscNodes=0.0):
    """Project spherical-harmonic modes onto plus and cross polarizations.

    This function reconstructs the strain polarizations from a set of
    spin-weighted spherical-harmonic modes using the standard
    spin-weight -2 decomposition.

    Parameters
    ----------
    modes : dict
        Dictionary {(ℓ, m): h_ℓm(t)} of complex waveform modes.
    t : ndarray
        Time array in seconds.
    sample : dict
        Dictionary containing:
        - 'theta_jn' : float
            Inclination angle (radians).
        - 'phase' : float
            Reference orbital phase (radians).
    longAscNodes : float, optional
        Longitude of ascending nodes (radians). Default is 0.

    Returns
    -------
    h_plus : ndarray
        Plus polarization strain.
    h_cross : ndarray
        Cross polarization strain.
    """
    inclination = sample["theta_jn"]
    phiRef = sample["phase"]

    h_complex = np.zeros_like(t, dtype=complex)

    for (L, M), h_LM in modes.items():
        Y_LM = lal.SpinWeightedSphericalHarmonic(
            inclination, -phiRef + np.pi / 2, -2, L, M
        )
        h_complex += h_LM * Y_LM

    h_plus_temp = np.real(h_complex)
    h_cross_temp = -np.imag(h_complex)

    h_plus = h_plus_temp * np.cos(2 * longAscNodes) + h_cross_temp * np.sin(
        2 * longAscNodes
    )
    h_cross = -h_plus_temp * np.sin(2 * longAscNodes) + h_cross_temp * np.cos(
        2 * longAscNodes
    )

    return h_plus, h_cross


def compute_angular_integral_factor(tuple1, tuple2, tuple3=None):
    """Compute angular coupling using Wigner 3j symbols.

    Evaluates the integral of three spin-weighted spherical harmonics
    expressed in terms of Wigner 3j symbols.

    Parameters
    ----------
    tuple1 : tuple
        (spin_weight, ℓ, m) for the first mode.
    tuple2 : tuple
        (spin_weight, ℓ, m) for the second mode.
    tuple3 : tuple, optional
        (spin_weight, ℓ, m) for the third mode.
        If None, defaults to (0, 0, 0).

    Returns
    -------
    float
        Angular integral factor. Returns 0 if triangle or
        spin-weight selection rules are violated.
    """
    factor = 1.0
    if tuple3 is None:
        tuple3 = 0, 0, 0
        factor = np.sqrt(4 * np.pi)

    S1, L1, M1 = tuple1
    S2, L2, M2 = tuple2
    S3, L3, M3 = tuple3

    if abs(S1) > L1 or abs(S2) > L2 or abs(S3) > L3:
        return 0

    prefactor = factor * np.sqrt(
        (2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1) / (4 * np.pi)
    )
    w3j1 = spherical_functions.Wigner3j(L1, L2, L3, M1, M2, M3)
    w3j2 = spherical_functions.Wigner3j(L1, L2, L3, -S1, -S2, -S3)

    return prefactor * w3j1 * w3j2


def compute_angular_factors(ell_max=4, spin_weight=-2):
    """Precompute angular factors for memory calculations.

    Parameters
    ----------
    ell_max : int, optional
        Maximum ℓ mode included in the computation. Default is 4.
    spin_weight : int, optional
        Spin weight of the waveform modes. Default is -2.

    Returns
    -------
    dict
        Dictionary of angular factors keyed by
        (S1, L1, M1, S2, L2, M2, S3, L3, M3).
    """
    angular_factors = {}

    S1 = -spin_weight
    S2 = spin_weight
    S3 = 0
    for L1 in range(2, ell_max + 1):
        for M1 in range(-L1, L1 + 1):
            for L2 in range(2, ell_max + 1):
                for M2 in range(-L2, L2 + 1):
                    for L3 in range(2, ell_max + 1):
                        for M3 in range(-L3, L3 + 1):
                            ethbar_sq_factor = -np.sqrt(
                                (L3 + 0) * (L3 - 0 + 1)
                            ) * -np.sqrt((L3 + -1) * (L3 - -1 + 1))
                            mathfrak_D_inv_factor = 8.0 / (
                                (L3 + 2) * (L3 + 1) * L3 * (L3 - 1)
                            )
                            spin_weight_factor = (
                                0.125 * ethbar_sq_factor * mathfrak_D_inv_factor
                            )
                            angular_factors[(S1, L1, M1, S2, L2, M2, S3, L3, M3)] = (
                                spin_weight_factor
                                * (-1) ** (S1 - M1 + (-M1 + M2))
                                * compute_angular_integral_factor(
                                    (S1, L1, M1), (S2, L2, M2), (S3, L3, M3)
                                )
                            )

    return angular_factors


def compute_memory_correction(
    h, t, sample, ell_max=None, s=-2, angular_factors=None, return_memory_only=False
):
    """Compute the gravitational-wave memory correction.

    This function computes the Christodoulou memory contribution from
    a set of spherical-harmonic waveform modes and optionally adds it
    to the original waveform.

    Parameters
    ----------
    h : dict
        Dictionary {(ℓ, m): h_ℓm(t)} of complex waveform modes.
    t : ndarray
        Time array in seconds.
    sample : dict
        Dictionary containing component masses (solar masses) and
        luminosity distance (Mpc).
    ell_max : int, optional
        Maximum ℓ mode to include in the memory calculation.
        Defaults to the maximum ℓ present in `h`.
    s : int, optional
        Spin weight of the waveform modes. Default is -2.
    angular_factors : dict, optional
        Precomputed angular factors (from `compute_angular_factors`).
        If None, they are computed on the fly.
    return_memory_only : bool, optional
        If True, return only the memory contribution.
        If False, return the original waveform with memory added.
        Default is False.

    Returns
    -------
    dict
        Dictionary {(ℓ, m): h_ℓm(t)} containing either:
        - the waveform including memory, or
        - only the memory contribution if `return_memory_only=True`.
    """
    if ell_max is None:
        ell_max = max([mode[0] for mode in h.keys()])

    M_total = sample["mass_1"] + sample["mass_2"]
    M_in_sec = M_total * lal.MSUN_SI * lal.G_SI / lal.C_SI**3
    distance = sample["luminosity_distance"] * 1e6 * lal.PC_SI
    strain_scale = (M_total * lal.MSUN_SI * lal.G_SI / lal.C_SI**2) / distance

    h_dot = {}
    for mode in h:
        h_dot[mode] = np.gradient(h[mode], t)

    if return_memory_only:
        h_memory = {}
    else:
        h_memory = copy.deepcopy(h)

    modes = [mode for mode in h.keys() if mode[0] <= ell_max]

    for mode1 in modes:
        for mode2 in modes:
            m = -mode1[1] + mode2[1]
            if abs(m) > ell_max:
                continue

            mode_product_integral = cumtrapz(
                np.conjugate(h_dot[mode1]) * h_dot[mode2], t, initial=0
            )

            for ell in range(max(2, abs(m)), min(mode1[0] + mode2[0] + 1, ell_max)):
                if angular_factors is None:
                    ethbar_sq_factor = -np.sqrt((ell + 0) * (ell - 0 + 1)) * -np.sqrt(
                        (ell + -1) * (ell - -1 + 1)
                    )
                    mathfrak_D_inv_factor = 8.0 / (
                        (ell + 2) * (ell + 1) * ell * (ell - 1)
                    )
                    spin_weight_factor = (
                        0.125 * ethbar_sq_factor * mathfrak_D_inv_factor
                    )

                    angular_integral_factor = (-1) ** (
                        s + mode1[1] + m
                    ) * compute_angular_integral_factor(
                        (-s, mode1[0], -mode1[1]), (s, mode2[0], mode2[1]), (0, ell, -m)
                    )

                    angular_factor = spin_weight_factor * angular_integral_factor
                else:
                    angular_factor = angular_factors[
                        (-s, mode1[0], -mode1[1], s, mode2[0], mode2[1], 0, ell, -m)
                    ]

                if angular_factor == 0:
                    continue

                if (ell, m) in h_memory:
                    h_memory[(ell, m)] += (
                        angular_factor * mode_product_integral / strain_scale * M_in_sec
                    )
                else:
                    h_memory[(ell, m)] = (
                        angular_factor * mode_product_integral / strain_scale * M_in_sec
                    )

    return h_memory


def compute_memory_and_map_to_polarizations(
    sample, res, angular_factors=None, approximant=lalsim.NRSur7dq4, ell_max=4
):
    """Compute the memory and project it to plus/cross polarizations.

    This function evaluates the NRSur7dq4 surrogate using LALSimulation
    to obtain spherical-harmonic modes, computes the
    (Christodoulou) memory contribution, and maps the resulting
    memory modes onto the plus and cross strain polarizations.

    Parameters
    ----------
    sample : dict
        Dictionary of source parameters containing:
        - 'mass_1', 'mass_2' : float
            Component masses in solar masses.
        - 'spin_1x', 'spin_1y', 'spin_1z' : float
            Dimensionless spin components of the primary.
        - 'spin_2x', 'spin_2y', 'spin_2z' : float
            Dimensionless spin components of the secondary.
        - 'luminosity_distance' : float
            Luminosity distance in Mpc.
        - 'theta_jn' : float
            Inclination angle (radians).
        - 'phase' : float
            Reference orbital phase (radians).
    res : dict
        Result dictionary containing:
        - 'ifos' : list
            Interferometer objects.
        - 'config' : object
            Configuration object.
        - 'samples' : list of dict
            Posterior samples.
        - 'fd' : dict
            Frequency-domain waveform data used for sizing arrays.
    angular_factors : dict, optional
        Precomputed angular factors from `compute_angular_factors`.
        If None, angular factors are computed internally.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    ell_max : int, optional
        Maximum ℓ mode included in the memory calculation.
        Default is 4.

    Returns
    -------
    hp_memory : ndarray
        Plus polarization of the memory contribution.
    hc_memory : ndarray
        Cross polarization of the memory contribution.
    t : ndarray
        Time array in seconds corresponding to the memory waveform.
    """
    # evaluate surrogate
    h, t = evaluate_surrogate_with_LAL(sample, res, approximant=approximant, ell_max=ell_max)

    # memory
    h_memory = compute_memory_correction(
        h, t, sample, angular_factors=angular_factors, return_memory_only=True
    )

    hp_memory, hc_memory = map_modes_to_polarizations(h_memory, t, sample)

    return hp_memory, hc_memory, t


def insert_waveform(h, t_target, input_idx, target_idx):
    """Insert a waveform into a target time array with constant extension.

    This function returns an array with the same length as `t_target`
    in which the waveform `h` is inserted such that
    `h[input_idx]` aligns with the element at `target_idx`
    in the output array. Outside the region where `h` overlaps
    the target array, the output is filled by constant extension
    using the nearest boundary value of `h`.

    Parameters
    ----------
    h : array_like
        One-dimensional waveform array to be inserted.
    t_target : array_like
        Target time array defining the desired output length.
        Only its length is used.
    input_idx : int
        Index in `h` that should align with `target_idx`
        in the output array.
    target_idx : int
        Index in the output array at which `h[input_idx]`
        will be placed.

    Returns
    -------
    ndarray
        Array of length `len(t_target)` containing the inserted
        waveform with constant boundary extension outside the
        overlap region.

    Raises
    ------
    IndexError
        If `input_idx` is out of bounds for the input waveform `h`.
    """
    h = np.asarray(h).ravel()
    n = h.size
    m = np.asarray(t_target).size

    if n == 0:
        return np.zeros(m)
    if m == 0:
        return np.zeros(0, dtype=h.dtype)

    input_idx = int(input_idx)
    target_idx = int(target_idx)

    if not (0 <= input_idx < n):
        raise IndexError(f"input_idx={input_idx} is out of bounds for h of length {n}")

    # Where h[0] would land in the target
    start = target_idx - input_idx  # can be negative
    end = start + n  # EXCLUSIVE end in target coordinates

    out = np.empty(m, dtype=h.dtype)

    # No overlap: h entirely before target
    if end <= 0:
        out.fill(h[-1])
        return out

    # No overlap: h entirely after target
    if start >= m:
        out.fill(h[0])
        return out

    # Overlap region in target (exclusive end, Python slicing style)
    ov_start = max(start, 0)
    ov_end = min(end, m)

    # Corresponding overlap region in h
    h_start = ov_start - start  # >= 0
    h_end = h_start + (ov_end - ov_start)  # exclusive

    # Insert overlap
    out[ov_start:ov_end] = h[h_start:h_end]

    # Constant-fill left side with first valid inserted value
    if ov_start > 0:
        out[:ov_start] = h[h_start]

    # Constant-fill right side with last valid inserted value
    if ov_end < m:
        out[ov_end:] = h[h_end - 1]

    return out


def insert_memory_into_time_array(hp, hc, t, sample, config, fd):
    """Insert memory polarizations into the analysis time array.

    This function aligns the time-domain memory waveform with the
    detector data time grid by matching the memory peak to the
    geocentric coalescence time. The waveform is inserted into a
    target array of appropriate length and padded using constant
    boundary extension outside the overlap region.

    Parameters
    ----------
    hp : ndarray
        Plus polarization memory strain in the time domain.
    hc : ndarray
        Cross polarization memory strain in the time domain.
    t : ndarray
        Time array (seconds) corresponding to `hp` and `hc`.
    sample : dict
        Dictionary containing:
        - 'geocent_time' : float
            Geocentric coalescence time (seconds).
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - start_time : float
            Start time of the analysis segment (seconds).
    fd : ndarray
        Frequency-domain waveform array for a given detector.
        Used to determine the target time-series length.

    Returns
    -------
    hp_inserted : ndarray
        Plus polarization memory strain aligned to the analysis grid.
    hc_inserted : ndarray
        Cross polarization memory strain aligned to the analysis grid.
    delta_t : float
        Effective time shift applied when aligning the waveform.
    """
    fs = config.sampling_frequency
    deltaT = 1 / fs

    t_data = np.arange(2 * (fd.size - 1)) * deltaT + config.start_time

    input_idx = np.argmin(abs(t))
    target_idx = np.argmin(abs(t_data - sample["geocent_time"]))

    delta_t = sample["geocent_time"] - (t_data[target_idx] - t[input_idx])

    hp_inserted = insert_waveform(hp, t_data, input_idx, target_idx)
    hc_inserted = insert_waveform(hc, t_data, input_idx, target_idx)

    return hp_inserted, hc_inserted, delta_t


def polarizations_to_FD(hp_memory, hc_memory, delta_t, config, roll_on=0.2):
    """Convert time-domain memory polarizations to the frequency domain.

    This function applies a Tukey window to the time-domain memory
    polarizations, performs a real FFT, and applies the appropriate
    time-shift phase factor.

    Parameters
    ----------
    hp_memory : ndarray
        Plus polarization memory strain in the time domain.
    hc_memory : ndarray
        Cross polarization memory strain in the time domain.
    delta_t : float
        Time shift (seconds) applied when aligning the waveform.
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - duration : float
            Duration of the analysis segment in seconds.
    roll_on : float, optional
        Duration (seconds) of the Tukey window roll-on region.
        Default is 0.2.

    Returns
    -------
    hp_memory_FD : ndarray
        Frequency-domain plus polarization memory strain.
    hc_memory_FD : ndarray
        Frequency-domain cross polarization memory strain.
    """
    # window
    alpha = 2 * roll_on / config.duration
    window = scipy.signal.windows.tukey(hp_memory.size, alpha)

    # fft
    fs = config.sampling_frequency
    deltaT = 1 / fs

    freqs = np.fft.rfftfreq(hp_memory.size, deltaT)
    hp_memory_FD = (
        np.fft.rfft(window * hp_memory)
        * deltaT
        * np.exp(-1j * 2 * np.pi * freqs * delta_t)
    )
    hc_memory_FD = (
        np.fft.rfft(window * hc_memory)
        * deltaT
        * np.exp(-1j * 2 * np.pi * freqs * delta_t)
    )

    return hp_memory_FD, hc_memory_FD


def project_to_detectors(hp, hc, sample, ifos, is_SEOB):
    """Project polarizations onto detector responses.

    This function computes the detector response for each interferometer
    given plus and cross polarizations in the frequency domain. It
    applies antenna pattern factors and calibration corrections using
    Bilby-compatible calibration parameters.

    Parameters
    ----------
    hp : ndarray
        Frequency-domain plus polarization strain.
    hc : ndarray
        Frequency-domain cross polarization strain.
    sample : dict
        Dictionary of source parameters including sky location,
        polarization angle, geocentric time, and calibration parameters.
    ifos : list
        List of interferometer objects with:
        - name : str
        - calibration_model : object
        - get_detector_response(...) method
    is_SEOB: bool
        Whether or not the sample comes from SEOB. If True, ignore parts
        of the code responsible for calibration information.

    Returns
    -------
    dict
        Dictionary mapping interferometer name to complex
        frequency-domain strain:
        {ifo_name: h_fd}.
    """
    pols = {
        "plus": hp,
        "cross": hc,
    }

    if not is_SEOB:
        n_points = int(ifos[0].calibration_model.n_points)
        sample_normalized = _ensure_bilby_calibration_keys(
            sample, tuple(ifo.name for ifo in ifos), n_points
        )
    else:
        sample_normalized = sample
        
    out: Dict[str, np.ndarray] = {}
    for ifo in ifos:
        model_fd = ifo.get_detector_response(pols, sample_normalized)
        out[ifo.name] = model_fd

    return out


def make_memories(res, angular_factors=None, approximant=lalsim.NRSur7dq4, ell_max=4):
    """Compute memory waveforms for a set of posterior samples.

    This function computes the gravitational-wave memory
    contribution for each sample in a result dictionary, projects the
    memory onto detector responses, and returns the detector-frame
    frequency-domain memory waveforms.

    Parameters
    ----------
    res : dict
        Result dictionary containing:
        - 'ifos' : list
            Interferometer objects.
        - 'config' : object
            Configuration object.
        - 'samples' : list of dict
            Posterior samples.
        - 'fd' : dict
            Frequency-domain waveform data used for sizing arrays.
    angular_factors : dict, optional
        Precomputed angular factors (from `compute_angular_factors`).
        If None, they are computed on the fly.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    ell_max : int, optional
        Maximum ℓ mode included in the memory calculation.
        Default is 4.

    Returns
    -------
    list of dict
        List of detector-frame memory waveforms, one per sample.
        Each entry is a dictionary:
        {ifo_name: h_memory_fd}.
    """
    ifos = res["ifos"]
    config = res["config"]
    samples = res["samples"]

    if angular_factors is None:
        angular_factors = compute_angular_factors(ell_max)

    h_memories_in_det = []
    for i, sample in enumerate(samples):
        hp, hc, t = compute_memory_and_map_to_polarizations(
            sample, res, angular_factors=angular_factors, approximant=approximant, ell_max=ell_max
        )

        hp_inserted, hc_inserted, delta_t = insert_memory_into_time_array(
            hp, hc, t, sample, config, res["fd"][ifos[0].name]["model"][i]
        )

        hp_FD, hc_FD = polarizations_to_FD(hp_inserted, hc_inserted, delta_t, config)

        is_SEOB = approximant == lalsim.SEOBNRv4PHM
        h_memory_in_det = project_to_detectors(hp_FD, hc_FD, sample, ifos, is_SEOB)

        h_memories_in_det.append(h_memory_in_det)

    return h_memories_in_det


def compute_memory_variables_likelihoods_and_weights(res, h_memories_in_dets):
    """Compute Gaussian amplitude parameters and importance weights for memory.

    This function evaluates, for each posterior sample, the optimal
    Gaussian amplitude estimator for the memory contribution.
    It computes the maximum-likelihood amplitude, its uncertainty,
    a random draw from the corresponding Gaussian distribution,
    and the associated log-weight and log-likelihood contribution.

    The calculation is performed by combining inner products across
    all detectors.

    Parameters
    ----------
    res : dict
        Result dictionary containing:
        - 'ifos' : list
            List of interferometer objects with a
            `template_template_inner_product` method.
        - 'fd' : dict
            Frequency-domain data products, including:
            res['fd'][ifo_name]['residual']
            which contains residual frequency-domain data for each sample.
    h_memories_in_dets : list of dict
        List (indexed by sample) of detector-frame memory waveforms.
        Each entry is a dictionary:
        {ifo_name: h_memory_fd}.

    Returns
    -------
    ndarray
        Array of shape (n_samples, 5) where each row contains:
        - A_hat : float
            Maximum-likelihood amplitude estimator.
        - A_sigma : float
            Standard deviation of the Gaussian amplitude posterior.
        - A_sample : float
            Random draw from the Gaussian posterior.
        - log_weight : float
            Logarithmic importance weight associated with the
            marginalization over amplitude.
        - log_likelihood : float
            Log-likelihood contribution including the memory term.
    """
    memory_variables_likelihoods_and_weights = []
    for k in range(len(res["fd"][res["ifos"][0].name]["residual"])):
        hrs = 0.0
        hhs = 0.0
        rrs = 0.0
        for det in res["ifos"]:
            hrs += det.template_template_inner_product(
                h_memories_in_dets[k][det.name], res["fd"][det.name]["residual"][k]
            )
            hhs += det.template_template_inner_product(
                h_memories_in_dets[k][det.name], h_memories_in_dets[k][det.name]
            )
            rrs += det.template_template_inner_product(
                res["fd"][det.name]["residual"][k], res["fd"][det.name]["residual"][k]
            )
        A_hat = np.real(hrs / hhs)
        A_sigma = 1 / np.sqrt(np.real(hhs))
        A_sample = np.random.normal(loc=A_hat, scale=A_sigma)
        log_weight = 0.5 * A_hat * np.conjugate(hrs) - 0.5 * np.log(2 * np.pi * hhs)
        log_likelihood = -0.5 * rrs + log_weight

        memory_variables_likelihoods_and_weights.append(
            [A_hat, A_sigma, A_sample, log_weight, log_likelihood]
        )
    memory_variables_likelihoods_and_weights = np.array(
        memory_variables_likelihoods_and_weights
    )

    return memory_variables_likelihoods_and_weights
