import os
import copy
import scipy
import numpy as np
import gwsurrogate
import spherical_functions
import multiprocessing as mp
from scipy.integrate import cumulative_trapezoid as cumtrapz

import lal
import lalsimulation as lalsim
import lalsimulation.gwsignal as gwsignal
import astropy.units as u
from lalsimulation.gwsignal.core.waveform import CompactBinaryCoalescenceGenerator

from memory.gw_residuals import _ensure_bilby_calibration_keys

from scipy.special import log_ndtr
from scipy.stats import truncnorm


def lal_mode_to_numpy_fft_bins(fs_mode, n_pos, dt):
    """Convert a LAL COMPLEX16FrequencySeries mode to a NumPy-ifft-ordered spectrum.

    LAL's SphHarmFrequencySeries stores frequencies as a single contiguous array
    containing negative and positive frequencies with DC in the middle.
    For the standard LVK grid, the full array length is typically:
        len_full = 2 * (n_pos - 1) + 1 = N + 1,
    where N = 2*(n_pos-1) is the time-series length.

    NumPy's complex FFT uses length-N bins with a single Nyquist bin at k=N/2
    (corresponding to -f_Nyq). Therefore we:
      - keep the negative-Nyquist bin from LAL,
      - drop the positive-Nyquist bin from LAL.

    Parameters
    ----------
    fs_mode : lal.COMPLEX16FrequencySeries
        Mode frequency series from `lalsimulation.SphHarmFrequencySeriesGetMode`.
    n_pos : int
        Number of nonnegative-frequency bins on the rfft grid (N/2 + 1).
    dt : float
        Sampling time step in seconds.

    Returns
    -------
    Hfull : ndarray
        Complex array of length N in NumPy FFT ordering (DC at index 0).
    """
    data = np.asarray(fs_mode.data.data, dtype=complex)
    len_full = int(data.size)

    N = 2 * (n_pos - 1)

    # For the standard LAL layout we expect len_full == N + 1
    # (both -Nyquist and +Nyquist are present in LAL's storage).
    if len_full != N + 1:
        raise ValueError(
            f"Unexpected LAL mode length: got len_full={len_full}, expected N+1={N+1}. "
            "This can happen if f_max/deltaF do not match your LVK grid."
        )

    # DC index in LAL storage
    # If len_full = 2*(n_pos-1)+1, then DC index is (n_pos-1)
    dc_idx = n_pos - 1

    Hfull = np.zeros(N, dtype=complex)

    # NumPy bins:
    #   k=0               -> f=0
    #   k=1..N/2-1        -> +df .. +(f_Nyq-df)
    #   k=N/2             -> -f_Nyq (Nyquist)
    #   k=N/2+1..N-1      -> -(f_Nyq-df) .. -df
    #
    # LAL bins (conceptually):
    #   data[0]           -> -f_Nyq
    #   data[1:dc_idx]    -> -f_Nyq+df .. -df
    #   data[dc_idx]      -> 0
    #   data[dc_idx+1:-1] -> +df .. +f_Nyq-df
    #   data[-1]          -> +f_Nyq  (drop this to fit NumPy length-N)
    Hfull[0] = data[dc_idx]
    Hfull[1 : n_pos - 1] = data[dc_idx + 1 : dc_idx + (n_pos - 1)]
    Hfull[N // 2] = data[0]                    # keep negative Nyquist
    Hfull[N // 2 + 1 :] = data[1:dc_idx]       # remaining negative freqs

    # Apply epoch shift if present
    t0 = float(fs_mode.epoch)
    if t0 != 0.0:
        freqs_full = np.fft.fftfreq(N, dt)
        Hfull *= np.exp(+2j * np.pi * freqs_full * t0)

    return Hfull


def ifft_modes_from_ChooseFDModes(h_modes_lal, config, ell_max=4, n_pos=None, center_peak=True):
    """Inverse Fourier transform LAL FD modes to complex time-domain modes (precession-safe).

    This function converts the output of `lalsimulation.SimInspiralChooseFDModes`
    into complex time-domain spin-weighted spherical-harmonic modes on the LVK
    analysis grid *without assuming aligned-spin half-axis support or ±m symmetry*.

    It uses the full negative+positive frequency content returned by LAL for each
    (ℓ,m) mode, maps it to NumPy FFT bin ordering, and performs an inverse FFT with
    LAL scaling:
        h(t_n) = IFFT[H_k] / dt.

    Optionally, it circularly shifts all modes so that the peak of |h_22| is placed
    at the center of the segment, preventing merger/ringdown from wrapping to the
    start of the array.

    Parameters
    ----------
    h_modes_lal : lal.SphHarmFrequencySeries
        Output of `lalsimulation.SimInspiralChooseFDModes`.
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
        - duration : float
    ell_max : int, optional
        Maximum ℓ mode to extract and transform. Default is 4.
    n_pos : int, optional
        Number of nonnegative-frequency bins. If None, inferred from
        N = duration * sampling_frequency as N/2 + 1.
        For LVK analyses you typically want to pass `n_pos = fd_model.size`.
    center_peak : bool, optional
        If True, shift so that peak(|h_22|) is at t=0 (center of array).

    Returns
    -------
    h_modes_td : dict
        Dictionary {(ℓ, m): h_ℓm(t)} of complex time-domain modes.
    t : ndarray
        Time array in seconds (centered if `center_peak=True`).
    """
    fs = config.sampling_frequency
    dt = 1.0 / fs

    if n_pos is None:
        N_float = config.duration * fs
        N = int(round(N_float))
        if abs(N - N_float) > 1e-6:
            raise ValueError(
                "config.duration * config.sampling_frequency must be an integer. "
                f"Got {N_float}."
            )
        n_pos = N // 2 + 1
    else:
        n_pos = int(n_pos)
        N = 2 * (n_pos - 1)

    h_modes_td = {}

    for L in range(2, ell_max + 1):
        for M in range(-L, L + 1):
            fs_LM = lalsim.SphHarmFrequencySeriesGetMode(h_modes_lal, L, M)
            Hfull = lal_mode_to_numpy_fft_bins(fs_LM, n_pos, dt)
            h_modes_td[(L, M)] = np.fft.ifft(Hfull) / dt

    # Build time array
    if center_peak and (2, 2) in h_modes_td:
        peak_idx = int(np.argmax(np.abs(h_modes_td[(2, 2)])))
        center_idx = N // 2
        shift = center_idx - peak_idx
        for k in list(h_modes_td.keys()):
            h_modes_td[k] = np.roll(h_modes_td[k], shift)
        t = (np.arange(N) - center_idx) * dt
    else:
        t = np.arange(N) * dt

    return h_modes_td, t

def evaluate_surrogate_with_LAL(sample, config, ifos, approximant=lalsim.NRSur7dq4, ell_max=4, FD=False):
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
    ifos : object
        IFOs.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    ell_max : int, optional
        Maximum ℓ mode included in the waveform model.
        Default is 4.
    FD : bool, optional
        Whether or not to call the FDModes function instead.

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
    inclination = sample['theta_jn']

    phiRef = sample["phase"]

    fs = config.sampling_frequency
    deltaT = 1.0 / fs
    
    duration = config.duration
    deltaF = 1 / duration

    f_low = config.minimum_frequency[ifos[0].name]
    f_max = fs/2
    f_ref = config.reference_frequency

    if approximant == lalsim.NRSur7dq4:
        f_low = -1
    else:
        # Some approximants (e.g. IMRPhenomXO4a, NRSur7dq4 FD) require f_low <= f_ref.
        # When the analysis config has f_ref < f_low (e.g. 9 Hz vs 20 Hz), lower f_low
        # so the waveform starts at or below the reference frequency.
        # We must NOT change f_ref — it sets the spin angle convention.
        f_low = min(f_low, f_ref)

    # --- gwsignal path (e.g. SEOBNRv5PHM via pyseobnr) ---
    if isinstance(approximant, CompactBinaryCoalescenceGenerator):
        # Read lmax_nyquist from the PE config's waveform_arguments_dict if
        # present (e.g. {'lmax_nyquist': 1} for low-mass NSBH events like
        # GW230529 where even the (2,2) ringdown exceeds Nyquist at 4096 Hz).
        # Default to 2 so higher modes don't abort for typical BBH events.
        flat_cfg = config.config_dict.get("config", {})
        raw_wf_args = flat_cfg.get("waveform_arguments_dict") or flat_cfg.get("waveform-arguments-dict")
        cfg_lmax_nyquist = 2
        if raw_wf_args is not None:
            try:
                import ast
                extra = ast.literal_eval(str(raw_wf_args)) if isinstance(raw_wf_args, str) else raw_wf_args
                if isinstance(extra, dict) and "lmax_nyquist" in extra:
                    cfg_lmax_nyquist = int(extra["lmax_nyquist"])
            except Exception:
                pass
        params = {
            "mass1":        mass_1 * u.kg,
            "mass2":        mass_2 * u.kg,
            "spin1x":       s1x * u.dimensionless_unscaled,
            "spin1y":       s1y * u.dimensionless_unscaled,
            "spin1z":       s1z * u.dimensionless_unscaled,
            "spin2x":       s2x * u.dimensionless_unscaled,
            "spin2y":       s2y * u.dimensionless_unscaled,
            "spin2z":       s2z * u.dimensionless_unscaled,
            "distance":     distance * u.m,
            "deltaT":       deltaT * u.s,
            "f22_start":    max(f_low, 1.0) * u.Hz,
            "f22_ref":      f_ref * u.Hz,
            "phi_ref":      phiRef * u.rad,
            "inclination":  inclination * u.rad,
            "lmax":         ell_max,
            "lmax_nyquist": cfg_lmax_nyquist,
        }
        modes_gw = gwsignal.GenerateTDModes(params, approximant)
        h_modes = {}
        t = None
        for key, ts in modes_gw.items():
            if isinstance(key, str):
                continue  # skip 'time_array' or metadata entries
            l, m = int(key[0]), int(key[1])
            if l > ell_max:
                continue
            h_modes[(l, m)] = np.asarray(ts.value)
            if t is None:
                t = ts.times.value
        if t is None:
            raise ValueError("gwsignal returned no modes.")
        return h_modes, t

    if not FD:
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
    else:
        h_modes_lal = lalsim.SimInspiralChooseFDModes(
            mass_1,
            mass_2,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
            deltaF,
            f_low,
            f_max,
            f_ref,
            phiRef + np.pi/4 + np.pi,
            distance,
            inclination,
            None,
            approximant,
        )

        return ifft_modes_from_ChooseFDModes(h_modes_lal, config, ell_max=4, n_pos=None)
    

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
    sample, config, ifos, angular_factors=None, ell_max=4, approximant=lalsim.NRSur7dq4, is_TD=True
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
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - minimum_frequency : dict
            Dictionary of low-frequency cutoffs in Hz.
        - reference_frequency : float
            Reference frequency in Hz.
    ifos : object
        IFOs.
    angular_factors : dict, optional
        Precomputed angular factors from `compute_angular_factors`.
        If None, angular factors are computed internally.
    ell_max : int, optional
        Maximum ℓ mode included in the memory calculation.
        Default is 4.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    is_TD : bool
        Whether or not the approximant is TD.

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
    try:
        if is_TD:
            h, t = evaluate_surrogate_with_LAL(sample, config, ifos, approximant=approximant, ell_max=ell_max)
        else:
            h, t = evaluate_surrogate_with_LAL(sample, config, ifos, approximant=approximant, ell_max=ell_max, FD=True)
    except:
        raise ValueError(f"Can't evaluate approximant {approximant}.")

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


def polarizations_to_FD(hp_memory, hc_memory, delta_t, config, roll_on=1.0):
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
        Default is 1.0 to match ifo.strain_data.roll_off in gw_residuals.py.

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


def project_to_detectors(hp, hc, sample, ifos):
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

    n_points = int(ifos[0].calibration_model.n_points)
    sample_normalized = _ensure_bilby_calibration_keys(
        sample, tuple(ifo.name for ifo in ifos), n_points
    )
        
    out: Dict[str, np.ndarray] = {}
    for ifo in ifos:
        model_fd = ifo.get_detector_response(pols, sample_normalized)
        out[ifo.name] = model_fd

    return out


_G_CONFIG = None
_G_IFOs = None
_G_ANG = None
_G_ELL_MAX = None
_G_APPROX = None
_G_IS_TD = None


def init_worker(config, ifos, angular_factors, ell_max, approximant, is_TD):
    global _G_CONFIG, _G_IFOs, _G_ANG, _G_ELL_MAX, _G_APPROX, _G_IS_TD
    _G_CONFIG = config
    _G_IFOs = ifos
    _G_ANG = angular_factors
    _G_ELL_MAX = ell_max
    _G_APPROX = approximant
    _G_IS_TD = is_TD


def process_sample_small(args):
    i, sample_and_fd_residual_in_dets = args
    return process_sample((i, sample_and_fd_residual_in_dets, _G_CONFIG, _G_IFOs, _G_ANG, _G_ELL_MAX, _G_APPROX, _G_IS_TD))


def process_sample(args):
    """Trivial wrapper for multiprocessing in function below.
    """
    i, sample_and_fd_residual_in_dets, config, ifos, angular_factors, ell_max, approximant, is_TD = args
    sample, fd_residual_in_dets = sample_and_fd_residual_in_dets

    hp, hc, t = compute_memory_and_map_to_polarizations(
        sample,
        config,
        ifos,
        angular_factors=angular_factors,
        ell_max=ell_max,
        approximant=approximant,
        is_TD=is_TD
    )

    hp_inserted, hc_inserted, delta_t = insert_memory_into_time_array(
        hp,
        hc,
        t,
        sample,
        config,
        fd_residual_in_dets[ifos[0].name],
    )

    hp_FD, hc_FD = polarizations_to_FD(
        hp_inserted,
        hc_inserted,
        delta_t,
        config,
    )

    h_memory_in_dets = project_to_detectors(
        hp_FD,
        hc_FD,
        sample,
        ifos,
    )

    return compute_memory_variables_likelihoods_and_weights(fd_residual_in_dets, h_memory_in_dets, ifos)


def make_memories(samples, fd_residuals_in_dets, config, ifos, angular_factors=None, approximant=lalsim.NRSur7dq4, ell_max=4, multiprocess=False):
    """Compute memory waveforms for a set of posterior samples.

    This function computes the gravitational-wave memory
    contribution for each sample in a result dictionary, projects the
    memory onto detector responses, and returns the detector-frame
    frequency-domain memory waveforms.

    Parameters
    ----------
    samples : list
        Event samples.
    fd_model : list
        Event frequency domain residuals.
    config : object
        Configuration object with attributes:
        - sampling_frequency : float
            Sampling frequency in Hz.
        - minimum_frequency : dict
            Dictionary of low-frequency cutoffs in Hz.
        - reference_frequency : float
            Reference frequency in Hz.
    ifos : list
        Detector IFOs.
    angular_factors : dict, optional
        Precomputed angular factors (from `compute_angular_factors`).
        If None, they are computed on the fly.
    approximant : lalsim.Approximant
        Waveform approximant with which to generate modes.
        Default is lalsim.NRSur7dq4.
    ell_max : int, optional
        Maximum ℓ mode included in the memory calculation.
        Default is 4.
    multiprocess : bool, optional
        Whether or not to multiprocess over sample calculations.
        Default is False.

    Returns
    -------
    memory_variables_likelihoods_and_weights : ndarray
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
    if angular_factors is None:
        angular_factors = compute_angular_factors(ell_max)

    try:
        h, t = evaluate_surrogate_with_LAL(samples[0], config, ifos, approximant=approximant, ell_max=ell_max)
        is_TD = True
    except:
        try:
            h, t = evaluate_surrogate_with_LAL(samples[0], config, ifos, approximant=approximant, ell_max=ell_max, FD=True)
            is_TD = False
        except:
            raise ValueError(f"Can't evaluate approximant {approximant}.")

    print("Analyzing", len(samples), "samples.", flush=True)
    # gwsignal generators (e.g. SEOBNRv5PHM via pyseobnr) cannot be pickled
    # across spawn-based multiprocessing workers; fall back to serial.
    if multiprocess and isinstance(approximant, CompactBinaryCoalescenceGenerator):
        print("gwsignal approximant: disabling multiprocessing (not picklable).", flush=True)
        multiprocess = False
    if multiprocess:
        nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        print("Using", nproc, "processes.", flush=True)
        ctx = mp.get_context("spawn")
        args = [(samples[i], fd_residuals_in_dets[i]) for i in range(len(samples))]
        with ctx.Pool(
            processes=nproc,
            initializer=init_worker,
            initargs=(config, ifos, angular_factors, ell_max, approximant, is_TD),
        ) as pool:
            memory_variables_likelihoods_and_weights = pool.map(
                process_sample_small,
                enumerate(args),
                chunksize=max(100, len(samples) // (nproc * 200))
            )
    else:
        memory_variables_likelihoods_and_weights = []
        for i, sample in enumerate(samples):
            args = [sample, fd_residuals_in_dets[i]]
            memory_variables_likelihoods_and_weights.append(process_sample((i, args, config, ifos, angular_factors, ell_max, approximant, is_TD)))
            
    return np.array(memory_variables_likelihoods_and_weights)


def compute_memory_variables_likelihoods_and_weights(fd_residual_in_dets, h_memory_in_dets, ifos):
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
    fd_residual_in_dets : dict
        Dictionary containing the ifo names as keys and storing
        the frequency domain residual data.
    h_memory_in_dets : dict
        Dictionary containing the ifo names as keys and storing
        the frequency domain memory data.
    ifos : list
        Detector IFOs.

    Returns
    -------
    memory_variables_likelihoods_and_weights : ndarray
        Array of shape (1, 5) where each row contains:
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
    hrs = 0.0
    hhs = 0.0
    rrs = 0.0
    for det in ifos:
        hrs += det.template_template_inner_product(
            h_memory_in_dets[det.name], fd_residual_in_dets[det.name]
        ).real
        hhs += det.template_template_inner_product(
            h_memory_in_dets[det.name], h_memory_in_dets[det.name]
        ).real
        rrs += det.template_template_inner_product(
            fd_residual_in_dets[det.name], fd_residual_in_dets[det.name]
        ).real
            
    A_hat = np.real(hrs / hhs)
    A_sigma = 1 / np.sqrt(np.real(hhs))
    A_sample = np.random.normal(loc=A_hat, scale=A_sigma)
    log_weight = 0.5 * A_hat * np.real(np.conjugate(hrs)) - 0.5 * np.log(2 * np.pi * hhs)
    log_likelihood = -0.5 * rrs + log_weight
        
    memory_variables_likelihoods_and_weights = np.array(
        [A_hat, A_sigma, A_sample, log_weight, log_likelihood]
    )

    return memory_variables_likelihoods_and_weights
