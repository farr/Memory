import copy
import numpy as np
import gwsurrogate
import spherical_functions
from scipy.integrate import cumulative_trapezoid as cumtrapz


def evaluate_surrogate(
    path_to_surrogate, sample, dt=0.1, f_low=0.0, return_dynamics=False, ell_max=4
):
    """Evaluate NRSur7dq4 gravitational waveform surrogate for some sample.

    This function computes the inertial-frame gravitational-wave modes for
    a given posterior sample using the NRSur7dq4 surrogate model.

    Parameters
    ----------
    path_to_surrogate : str
        Path to the NRSur7dq4 surrogate H5 file.
    sample : dict
        Dictionary of posterior parameters with keys:
        - 'mass_1', 'mass_2' (float): Component masses in solar masses.
        - 'spin_1x', 'spin_1y', 'spin_1z' (float): Spin vector of object 1.
        - 'spin_2x', 'spin_2y', 'spin_2z' (float): Spin vector of object 2.
    dt : float, optional
        Time spacing in seconds. Default is 0.1.
    f_low : float, optional
        Low-frequency cutoff in Hz. Default is lowest frequency available.
    return_dynamics : bool, optional
        If True, also return precessing dynamics. Default is False.
    ell_max : int, optional
        Maximum ℓ mode to evaluate. Default is 4.

    Returns
    -------
    h_modes : dict
        Dictionary of complex waveform modes {(ℓ, m): h_ℓm(t)}.
    t : np.ndarray
        Array of times corresponding to the waveform modes.
    dyn : dict, optional
        Dictionary of precessing dynamics if `return_dynamics=True`.

    """
    # Extract masses and spins
    mass_1 = sample["mass_1"]
    mass_2 = sample["mass_2"]
    chiA = np.array([sample["spin_1x"], sample["spin_1y"], sample["spin_1z"]])
    chiB = np.array([sample["spin_2x"], sample["spin_2y"], sample["spin_2z"]])

    # Enforce m1 >= m2 for NRSur7dq4
    if mass_2 > mass_1:
        mass_1, mass_2 = mass_2, mass_1
        chiA, chiB = chiB, chiA

    q = mass_1 / mass_2
    Mtot = mass_1 + mass_2  # Total mass

    # Load the surrogate model
    sur = gwsurrogate.LoadSurrogate(path_to_surrogate)

    # Evaluate the surrogate waveform
    precessing_opts = {"return_dynamics": return_dynamics}

    t, h_modes, dyn = sur(
        q,
        chiA,
        chiB,
        dt=dt,
        f_low=f_low,
        precessing_opts=precessing_opts,
        ellMax=ell_max,
    )

    if return_dynamics:
        return h_modes, t, dyn
    else:
        return h_modes, t


def compute_angular_integral_factor(tuple1, tuple2, tuple3=None):
    """Compute the angular integral factor using Wigner 3j symbols.

    Parameters
    ----------
    tuple1 : tuple
        (spin_weight, ℓ, m) for first mode.
    tuple2 : tuple
        (spin_weight, ℓ, m) for second mode.
    tuple3 : tuple, optional
        (spin_weight, ℓ, m) for third mode. Defaults to (0, 0, 0).

    Returns
    -------
    float
        Angular integral factor. Returns 0 if triangle conditions are not satisfied.

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


def compute_angular_factors(ell_max=6, spin_weight=-2):
    """Precompute angular factors appearing in gravitational wave memory formula.

    Parameters
    ----------
    ell_max : int
        Maximum ℓ mode to include.
    spin_weight : int
        Spin weight of the waveform. Default is -2.

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
    h, t, ell_max=None, s=-2, angular_factors=None, return_memory_only=False
):
    """Compute the gravitational-wave memory correction for a waveform.

    Parameters
    ----------
    h : dict
        Dictionary of waveform modes {(ℓ, m): h_ℓm(t)}.
    t : np.ndarray
        Time array corresponding to the waveform.
    ell_max : int, optional
        Maximum ℓ to include in memory computation. Defaults to max ℓ in `h`.
    s : int, optional
        Spin weight of waveform. Default is -2.
    angular_factors : dict, optional
        Precomputed angular factors. If None, computed on the fly.
    return_memory_only : bool, optional
        If True, return only the memory modes, otherwise return h with memory added.

    Returns
    -------
    dict
        Waveform with memory correction applied, or just memory correction if
        return_memory_only=True. Keys are modes (ℓ, m) and values are arrays
        representing h_ℓm(t) with memory included, or just of the memory
        if return_memory_only=True.

    """
    if ell_max is None:
        ell_max = max([mode[0] for mode in h.keys()])

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
                    h_memory[(ell, m)] += angular_factor * mode_product_integral
                else:
                    h_memory[(ell, m)] = angular_factor * mode_product_integral

    return h_memory
