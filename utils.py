import copy
import scipy
import numpy as np
import gwsurrogate
import spherical_functions
from scipy.integrate import cumulative_trapezoid as cumtrapz

import lal
import lalsimulation as lalsim

from gw_residuals import _ensure_bilby_calibration_keys

def evaluate_surrogate(
    path_to_surrogate, sample, config, return_dynamics=False, ellMax=4
):
    """Evaluate the NRSur7dq4 surrogate model and return waveform modes.

    This function evaluates the NRSur7dq4 time-domain surrogate using
    `gwsurrogate` and returns inertial-frame gravitational-wave modes.

    Parameters
    ----------
    path_to_surrogate : str
        Path to the NRSur7dq4 surrogate HDF5 file.
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
    return_dynamics : bool, optional
        If True, also return the surrogate precessing dynamics.
        Default is False.
    ellMax : int, optional
        Maximum spherical-harmonic ℓ mode to evaluate. Default is 4.

    Returns
    -------
    h_modes : dict
        Dictionary of complex waveform modes {(ℓ, m): h_ℓm(t)}.
        Each value is a NumPy array.
    t : ndarray
        Time array in seconds.
    dyn : dict, optional
        Dictionary of precessing dynamics (only if
        `return_dynamics=True`).
    """
    # Extract masses and spins
    mass_1 = sample["mass_1"]
    mass_2 = sample["mass_2"]
    M_total = mass_1 + mass_2

    chiA = np.array([sample["spin_1x"], sample["spin_1y"], sample["spin_1z"]])
    chiB = np.array([sample["spin_2x"], sample["spin_2y"], sample["spin_2z"]])

    # Enforce m1 >= m2 for NRSur7dq4
    if mass_2 > mass_1:
        mass_1, mass_2 = mass_2, mass_1
        chiA, chiB = chiB, chiA

    q = mass_1 / mass_2

    # Get frequencies, time spacing, and phi_ref
    fs = config.sampling_frequency
    dt = 1 / fs

    f_low = config.minimum_frequency["H1"]
    f_ref = config.reference_frequency

    phi_ref = sample["phase"]

    # Load the surrogate model
    sur = gwsurrogate.LoadSurrogate(path_to_surrogate)

    # Evaluate the surrogate waveform
    precessing_opts = {"return_dynamics": return_dynamics}

    t, h_modes, dyn = sur(
        q,
        chiA,
        chiB,
        M=M_total,
        dist_mpc=sample["luminosity_distance"],
        units="mks",
        dt=dt,
        f_low=f_low,
        f_ref=f_ref,
        phi_ref=phi_ref,
        precessing_opts=precessing_opts,
        ellMax=ellMax,
    )

    if return_dynamics:
        return h_modes, t, dyn
    else:
        return h_modes, t


def evaluate_surrogate_with_LAL(
        sample, config, lmax=4
):
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
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - minimum_frequency : dict
            Dictionary of low-frequency cutoffs in Hz.
        - reference_frequency : float
            Reference frequency in Hz.
    lmax : int, optional
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

    phiRef = sample["phase"]

    fs = config.sampling_frequency
    deltaT = 1 / fs

    f_low = config.minimum_frequency["H1"]
    f_ref = config.reference_frequency

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
        -1, #f_low,
        f_ref,
        distance,
        None,
        lmax,
        lalsim.NRSur7dq4,
    )

    h_modes = {}

    for L in range(2, lmax + 1):
        for M in range(-L, L + 1):
            h_modes[(L, M)] = lalsim.SphHarmTimeSeriesGetMode(
                h_modes_lal, L, M
            ).data.data

    t = np.arange(len(h_modes[(2, 2)])) * deltaT
    
    t += float(
        lalsim.SphHarmTimeSeriesGetMode(
            h_modes_lal, 2, 2
        ).epoch
    )
    
    return h_modes, t


def evaluate_surrogate_with_LAL_as_polarizations(
        sample, config, lmax=4
):
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
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - minimum_frequency : dict
            Dictionary of low-frequency cutoffs in Hz.
        - reference_frequency : float
            Reference frequency in Hz.
    lmax : int, optional
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

    fs = config.sampling_frequency
    deltaT = 1 / fs

    f_low = config.minimum_frequency["H1"]
    f_ref = config.reference_frequency

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
        lalsim.NRSur7dq4,
    )

    return hp, hc


def map_modes_to_polarizations(
        modes, t, sample, longAscNodes=0.0
):
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


def compute_angular_integral_factor(
        tuple1, tuple2, tuple3=None
):
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


def compute_angular_factors(
        ell_max=6, spin_weight=-2
):
    """Precompute angular factors for nonlinear memory calculations.

    Parameters
    ----------
    ell_max : int, optional
        Maximum ℓ mode included in the computation. Default is 6.
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
    """Compute the nonlinear gravitational-wave memory correction.

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
    if ell_max is None:
        ell_max = max([mode[0] for mode in modes])

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
        sample, config, angular_factors=None, lmax=4
):
    """
    """
    # evaluate surrogate
    h, t = evaluate_surrogate_with_LAL(sample, config, lmax=4)
    
    # memory
    h_memory = compute_memory_correction(
        h, t, sample, angular_factors=angular_factors, return_memory_only=True
    )
    
    hp_memory, hc_memory = map_modes_to_polarizations(h_memory, t, sample)

    return hp_memory, hc_memory, t

def insert_waveform(h, t_target, input_idx, target_idx):
    """
    Return an array the same length as t_target where waveform h is 'inserted'
    so that h[input_idx] aligns with output[target_idx].

    Outside the overlap region, the output is filled with the nearest boundary
    value of h (constant extension), matching the intent of your original code.
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
    start = target_idx - input_idx          # can be negative
    end   = start + n                       # EXCLUSIVE end in target coordinates

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
    ov_end   = min(end, m)

    # Corresponding overlap region in h
    h_start = ov_start - start              # >= 0
    h_end   = h_start + (ov_end - ov_start) # exclusive

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
    """
    """
    fs = config.sampling_frequency
    deltaT = 1 / fs

    t_data = np.arange(2*(fd.size - 1)) * deltaT + config.start_time
    
    input_idx = np.argmin(abs(t))
    target_idx = np.argmin(abs(t_data - sample['geocent_time']))

    delta_t = sample['geocent_time'] - (t_data[target_idx] - t[input_idx])
    
    hp_inserted = insert_waveform(hp, t_data, input_idx, target_idx)
    hc_inserted = insert_waveform(hc, t_data, input_idx, target_idx)
    
    return hp_inserted, hc_inserted, delta_t

def polarizations_to_FD(hp_memory, hc_memory, delta_t, config, roll_on=0.2):
    """
    """
    # window
    alpha = 2 * roll_on / config.duration
    window = scipy.signal.windows.tukey(hp_memory.size, alpha)
    
    # fft
    fs = config.sampling_frequency
    deltaT = 1 / fs

    freqs = np.fft.rfftfreq(hp_memory.size, delta_t)
    hp_memory_FD = np.fft.rfft(window * hp_memory) * deltaT * np.exp(-1j * 2 * np.pi * freqs * delta_t)
    hc_memory_FD = np.fft.rfft(window * hc_memory) * deltaT * np.exp(-1j * 2 * np.pi * freqs * delta_t)
    
    return hp_memory_FD, hc_memory_FD

def project_to_detectors(hp, hc, sample, ifos):
    """
    """
    pols = {
        "plus": hp,
        "cross": hc,
    }
    
    n_points = int(ifos[0].calibration_model.n_points)
    sample_normalized = _ensure_bilby_calibration_keys(
        sample, tuple(ifo.name for ifo in ifos), n_points
    )

    out: Dict[str, np.ndarray] = {}
    for ifo in ifos:
        model_fd = ifo.get_detector_response(pols, sample_normalized)
        out[ifo.name] = model_fd
        
    return out

def compute_memories(res, ell_max=4):
    """
    """
    ifos = res['ifos']
    config = res['config']
    samples = res['samples']
    
    angular_factors = compute_angular_factors(ell_max)

    h_memories_in_det = []
    for i, sample in enumerate(samples):
        hp, hc, t = compute_memory_and_map_to_polarizations(
            sample, config, angular_factors=angular_factors, lmax=ell_max
        )

        hp_inserted, hc_inserted, delta_t = insert_memory_into_time_array(
            hp, hc, t, sample, config, res['fd']['H1']['model'][i]
        )

        hp_FD, hc_FD = polarizations_to_FD(hp_inserted, hc_inserted, delta_t, config)
        
        h_memory_in_det = project_to_detectors(hp_FD, hc_FD, sample, ifos)

        h_memories_in_det.append(h_memory_in_det)

    return h_memories_in_det
