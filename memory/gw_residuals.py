"""
Gravitational wave residual computation with calibration.

This module provides tools for computing data-model residuals for gravitational
wave events using PESummary posterior files with spline calibration.
"""

from __future__ import annotations

import ast
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from pesummary.io import read as pesummary_read
from gwosc.datasets import event_gps, event_detectors, find_datasets
from gwpy.timeseries import TimeSeries

import bilby
from bilby.gw.detector.psd import PowerSpectralDensity
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator


LOGGER = logging.getLogger("gw_residuals")


@dataclass
class AnalysisConfig:
    """Configuration for gravitational wave analysis.
    
    Stores both parsed analysis parameters and the complete raw config dictionary
    from the PESummary file, allowing access to any settings not explicitly parsed.
    """
    label: str
    event: str
    detectors: Tuple[str, ...]
    trigger_time: float
    start_time: float
    end_time: float
    duration: float
    sampling_frequency: float
    minimum_frequency: Dict[str, float]
    maximum_frequency: Optional[Dict[str, float]]
    reference_frequency: float
    waveform_approximant: str
    config_dict: Dict[str, Dict[str, Any]]
    """Complete config.ini settings as section -> key -> value dict."""


# ----------------------------
# Generic parsing helpers
# ----------------------------

def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert value to float with fallback."""
    if x is None:
        return default
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except ValueError:
        return default


def _maybe_literal(x: Any) -> Any:
    """Try to parse strings that are dict/list/numeric literals."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return x


def _get_cfg(cfg: Dict[str, Dict[str, Any]], section: str, key: str, default=None):
    """Get config value from section/key with fallback."""
    if section not in cfg or key not in cfg[section]:
        return default
    return _maybe_literal(cfg[section][key])


def _cfg_first_match(cfg: Dict[str, Dict[str, Any]], section: str, keys: List[str], default=None):
    """Find first matching key in config section."""
    sec = cfg.get(section, {})
    for k in keys:
        if k in sec:
            return _maybe_literal(sec[k])
    return default


def _parse_ifo_freq_dict(value: Any, detectors: Iterable[str], default: float) -> Dict[str, float]:
    """
    Parse a frequency specification that might be:
      - a float / int
      - a numeric string
      - a dict-like string: "{'H1': 20, 'L1': 20}" or "{H1: 20, L1: 20}"
      - a dict already
    Returns per-IFO dict with fallback default.
    """
    out = {d: float(default) for d in detectors}
    if value is None:
        return out

    value = _maybe_literal(value)

    if isinstance(value, (float, int, np.floating, np.integer)):
        return {d: float(value) for d in detectors}

    if isinstance(value, str):
        f = _as_float(value, None)
        if f is not None:
            return {d: float(f) for d in detectors}

        # Try to handle dict-like strings with bare keys
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            for d in detectors:
                s = s.replace(f"{d}:", f"'{d}':")
            try:
                dct = ast.literal_eval(s)
                if isinstance(dct, dict):
                    for d in detectors:
                        if d in dct:
                            out[d] = float(dct[d])
                    return out
            except Exception:
                return out

        return out

    if isinstance(value, dict):
        for d in detectors:
            if d in value:
                out[d] = float(value[d])
        return out

    return out


def _choose_label(data, label: Optional[str]) -> str:
    """Choose analysis label from data or default to first."""
    if label is not None:
        return label
    if len(data.labels) == 1:
        return data.labels[0]
    return data.labels[0]


# ----------------------------
# GWOSC strain helpers
# ----------------------------

# GWF filename pattern: {prefix}-{IFO}_HOFT_C00_[AR_]BAYESWAVE_S00-{GPS}-{DUR}.gwf
_GWF_FILENAME_RE = re.compile(
    r"^[A-Z]-([A-Z0-9]+)_HOFT_C00_(?:AR_)?BAYESWAVE_S\d+-(\d+)-(\d+)\.gwf$"
)

# Default channel name for glitch-subtracted strain in BayesWave output frames.
# Confirmed from Zenodo 16857060 (GWTC-4.0 glitch frames): H1:GDS-CALIB_STRAIN_CLEAN
# is the finalized PE-ready clean channel.  H1:GDS-CALIB_STRAIN_CLEAN_BAYESWAVE_S00
# is an intermediate product; H1:GDS-CALIB_STRAIN_CLEAN_glitch is the subtracted model.
# Override via glitch_channel_format if your frames use a different convention.
GLITCH_SUBTRACTED_CHANNEL_FORMAT = "{ifo}:GDS-CALIB_STRAIN_CLEAN"


def _find_frame_file(frame_dir: str, ifo: str, start: float, end: float) -> Optional[str]:
    """Find a BayesWave GWF frame file in *frame_dir* covering [start, end).

    Scans all ``*.gwf`` files in *frame_dir* whose filename encodes the IFO
    name and a GPS segment that fully contains ``[start, end)``.

    Parameters
    ----------
    frame_dir : str
        Directory containing ``.gwf`` files.
    ifo : str
        Interferometer name, e.g. ``"H1"``.
    start, end : float
        Requested GPS time range.

    Returns
    -------
    str or None
        Full path to the first matching file, or ``None`` if none found.
    """
    for path in glob.glob(os.path.join(frame_dir, "*.gwf")):
        fname = os.path.basename(path)
        m = _GWF_FILENAME_RE.match(fname)
        if m is None:
            continue
        file_ifo, gps_start, duration = m.group(1), int(m.group(2)), int(m.group(3))
        if file_ifo != ifo:
            continue
        if gps_start <= start and gps_start + duration >= end:
            return path
    return None


def _read_frame_strain(
    frame_file: str,
    ifo: str,
    start: float,
    end: float,
    channel_format: str = GLITCH_SUBTRACTED_CHANNEL_FORMAT,
) -> TimeSeries:
    """Read glitch-subtracted strain from a local GWF frame file.

    Parameters
    ----------
    frame_file : str
        Path to the ``.gwf`` file.
    ifo : str
        Interferometer name, e.g. ``"H1"``.
    start, end : float
        GPS time range to extract.
    channel_format : str
        Python format string for the channel name; ``{ifo}`` is replaced by
        *ifo*.  Default: ``"{ifo}:DCS-CALIB_STRAIN_CLEAN_C00"``.

    Returns
    -------
    TimeSeries
    """
    channel = channel_format.format(ifo=ifo)
    LOGGER.info("Reading frame strain: %s  channel=%s  [%.1f, %.1f)", frame_file, channel, start, end)
    return TimeSeries.read(frame_file, channel, start=start, end=end)


def _select_gwosc_dataset(start: float, end: float, fs: float) -> Optional[str]:
    """Select appropriate GWOSC dataset based on time segment and sampling frequency."""
    seg = (int(np.floor(start)), int(np.ceil(end)))
    runs = find_datasets(type="run", segment=seg)
    if not runs:
        return None
    want_4k = fs <= 4096 + 1e-6
    for r in runs:
        ru = r.upper()
        if want_4k and "4KHZ" in ru:
            return r
        if (not want_4k) and "16KHZ" in ru:
            return r
    return runs[0]


def _download_gwosc_strain(
    detectors: Iterable[str],
    start: float,
    end: float,
    fs: float,
    frame_dir: Optional[str] = None,
    glitch_channel_format: str = GLITCH_SUBTRACTED_CHANNEL_FORMAT,
    *,
    skip_missing: bool = True,
) -> Dict[str, TimeSeries]:
    """
    Obtain strain data for specified detectors and time range.

    If a detector's strain cannot be obtained (e.g., not observing / not in GWOSC),
    skip it when skip_missing=True.
    """
    dataset = None  # lazily queried only when needed
    out: Dict[str, TimeSeries] = {}

    for det in detectors:
        # 1) Try local frames first
        gwf = None
        if frame_dir is not None:
            gwf = _find_frame_file(frame_dir, det, start, end)

        try:
            if gwf is not None:
                ts = _read_frame_strain(gwf, det, start, end, glitch_channel_format)
            else:
                if dataset is None:
                    dataset = _select_gwosc_dataset(start, end, fs)
                LOGGER.info("Fetching GWOSC open strain: %s [%s, %s), dataset=%s", det, start, end, dataset)
                if dataset is None:
                    ts = TimeSeries.fetch_open_data(det, start, end, cache=True, verbose=True)
                else:
                    ts = TimeSeries.fetch_open_data(det, start, end, cache=True, verbose=True, dataset=dataset)

            # resample if needed
            if ts.sample_rate.value != fs:
                LOGGER.info("Resampling %s %g Hz -> %g Hz", det, ts.sample_rate.value, fs)
                ts = ts.resample(fs)

            out[det] = ts

        except Exception as e:
            msg = f"Could not obtain strain for {det} covering [{start}, {end}): {e!r}"
            if skip_missing:
                LOGGER.warning("%s. Dropping detector %s.", msg, det)
                continue
            raise

    if not out:
        raise RuntimeError(
            "Could not obtain strain for any detector. "
            "Provide frame_dir with local frames or choose a different time segment."
        )

    return out


def _infer_waveform_approximant_from_config(cfg_dict: Dict[str, Dict[str, Any]], default: str = "IMRPhenomPv2") -> str:
    """Infer waveform approximant from config, trying multiple naming conventions."""
    # Try flattened bilby_pipe style
    val = _cfg_first_match(cfg_dict, "config", ["waveform-approximant", "waveform_approximant", "approximant"], default=None)
    if val is not None:
        return str(val)

    # Try older sectioned style
    val = _cfg_first_match(cfg_dict, "waveform", ["waveform-approximant", "waveform_approximant", "approximant"], default=None)
    if val is not None:
        return str(val)

    # Fallback: scan any section for approximant key
    for section, sec in cfg_dict.items():
        if not isinstance(sec, dict):
            continue
        for k, v in sec.items():
            if "approximant" in str(k).lower() and "injection" not in str(k).lower():
                return str(_maybe_literal(v))

    return str(default)


# ----------------------------
# Config -> AnalysisConfig
# ----------------------------

def _parse_analysis_config(data, label: str, event: str) -> AnalysisConfig:
    """Parse PESummary config into AnalysisConfig dataclass."""
    cfg = data.config[label]

    # --- 1) Start with detectors from config (if present) ---
    dets = _get_cfg(cfg, "data", "detectors", None)
    if dets is None:
        detectors_cfg = tuple(sorted(event_detectors(event)))
    else:
        dets = _maybe_literal(dets)
        if isinstance(dets, str):
            try:
                detectors_cfg = tuple(ast.literal_eval(dets))
            except Exception:
                detectors_cfg = tuple(dets.replace(",", " ").split())
        else:
            detectors_cfg = tuple(dets)

    detectors_cfg = tuple(str(d).strip() for d in detectors_cfg if str(d).strip())

    # --- 2) OVERRIDE using detectors actually present in PESummary PSD for this label ---
    # This prevents "only L1" config entries from dropping H1 even though the run has H1 PSD/samples.
    psd_for_label = None
    try:
        psd_for_label = data.psd[label]
    except Exception:
        psd_for_label = None

    if isinstance(psd_for_label, dict) and len(psd_for_label) > 0:
        detectors_psd = tuple(sorted(psd_for_label.keys()))
        detectors = detectors_psd
    else:
        detectors = detectors_cfg

    trig = float(event_gps(event))

    duration = _get_cfg(cfg, "data", "duration", None)
    if duration is None:
        duration = _get_cfg(cfg, "engine", "seglen", 4.0)
    duration = float(duration)
    
    sampling_frequency = _get_cfg(cfg, "data", "sampling_frequency", None)
    if sampling_frequency is None:
        sampling_frequency = _get_cfg(cfg, "engine", "srate", 4096)
    sampling_frequency = float(sampling_frequency)

    start_time_cfg = _as_float(_get_cfg(cfg, "data", "start_time", None), None)
    end_time_cfg = _as_float(_get_cfg(cfg, "data", "end_time", None), None)
    if start_time_cfg is not None and end_time_cfg is not None:
        start_time, end_time = start_time_cfg, end_time_cfg
    else:
        start_time = trig - duration / 2.0
        end_time = trig + duration / 2.0

    # IMPORTANT: use the corrected `detectors` when building per-IFO dicts
    min_freq_val = _get_cfg(cfg, "analysis", "minimum_frequency", None)
    if min_freq_val is None:
        min_freq_val = _get_cfg(cfg, "lalinference", "flow", 20)
    min_freq = _parse_ifo_freq_dict(min_freq_val, detectors, default=20.0)

    max_freq_val = _get_cfg(cfg, "analysis", "maximum_frequency", None)
    if max_freq_val is None:
        max_freq_val = _get_cfg(cfg, "lalinference", "fhigh", None)
    
    max_freq = None
    if max_freq_val is not None:
        max_freq = _parse_ifo_freq_dict(
            max_freq_val,
            detectors,
            default=float(sampling_frequency / 2.0),
        )

    reference_frequency = _get_cfg(cfg, "waveform", "reference_frequency", None)
    if reference_frequency is None:
        reference_frequency = _get_cfg(cfg, "engine", "fref", 50.0)
    reference_frequency = float(reference_frequency)
    
    waveform_approximant = _get_cfg(cfg, "engine", "approx", None)
    if waveform_approximant is None:
        waveform_approximant = _infer_waveform_approximant_from_config(
            cfg, default="IMRPhenomPv2"
        )

    return AnalysisConfig(
        label=label,
        event=event,
        detectors=detectors,
        trigger_time=trig,
        start_time=float(start_time),
        end_time=float(end_time),
        duration=float(duration),
        sampling_frequency=float(sampling_frequency),
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        reference_frequency=float(reference_frequency),
        waveform_approximant=waveform_approximant,
        config_dict=cfg,
    )


# ----------------------------
# Spline calibration settings from config
# ----------------------------

def _infer_spline_npoints_from_config(cfg_dict: Dict[str, Dict[str, Any]], default: int = 10) -> int:
    """Infer number of spline calibration points from config."""
    candidates = [
        "spline_calibration_npoints",
        "spline_calibration_n_points",
        "spline_calibration_points",
        "calibration_npoints",
        "calibration_n_points",
        "npoints",
        "n_points",
        "recalib_npoints",
        "recalib_n_points",
        "spline_npoints",
        "spline_n_points",
    ]

    for section in ("calibration", "analysis", "likelihood", "data"):
        val = _cfg_first_match(cfg_dict, section, candidates, default=None)
        if val is not None:
            try:
                return int(val)
            except Exception:
                pass

    # Last resort: look for any "*npoint*" key in [calibration]
    for k, v in cfg_dict.get("calibration", {}).items():
        if "npoint" in str(k).lower():
            try:
                return int(_maybe_literal(v))
            except Exception:
                continue

    return int(default)


def _infer_calib_minmax_freq_from_config(
    cfg_dict: Dict[str, Dict[str, Any]],
    detectors: Tuple[str, ...],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Infer per-IFO calibration spline min/max frequencies from config."""
    min_keys = [
        "calibration_minimum_frequency",
        "calibration_minimum_frequency_dict",
        "spline_calibration_minimum_frequency",
        "spline_calibration_minimum_frequency_dict",
        "minimum_frequency_calibration",
        "minimum_frequency_calibration_dict",
        "spline_minimum_frequency",
    ]
    max_keys = [
        "calibration_maximum_frequency",
        "calibration_maximum_frequency_dict",
        "spline_calibration_maximum_frequency",
        "spline_calibration_maximum_frequency_dict",
        "maximum_frequency_calibration",
        "maximum_frequency_calibration_dict",
        "spline_maximum_frequency",
    ]

    min_val = _cfg_first_match(cfg_dict, "calibration", min_keys, default=None)
    max_val = _cfg_first_match(cfg_dict, "calibration", max_keys, default=None)

    min_dict: Dict[str, float] = {}
    max_dict: Dict[str, float] = {}

    if min_val is not None:
        parsed = _parse_ifo_freq_dict(min_val, detectors, default=np.nan)
        min_dict = {k: float(v) for k, v in parsed.items() if np.isfinite(v)}

    if max_val is not None:
        parsed = _parse_ifo_freq_dict(max_val, detectors, default=np.nan)
        max_dict = {k: float(v) for k, v in parsed.items() if np.isfinite(v)}

    return min_dict, max_dict


def _attach_spline_calibration_from_config(
    ifos: InterferometerList,
    cfg_dict: Dict[str, Dict[str, Any]],
    detectors: Tuple[str, ...],
    *,
    base_prefix: str = "recalib_",
    default_n_points: int = 10,
) -> Dict[str, Any]:
    """
    Attach bilby CubicSpline calibration models to interferometers.
    
    Uses per-IFO prefix like 'recalib_H1_' to match parameter naming conventions.
    """
    n_points = _infer_spline_npoints_from_config(cfg_dict, default=default_n_points)
    cal_min_dict, cal_max_dict = _infer_calib_minmax_freq_from_config(cfg_dict, detectors)

    info = {"n_points": int(n_points), "per_ifo": {}}

    for ifo in ifos:
        min_f = float(cal_min_dict.get(ifo.name, ifo.minimum_frequency))

        if getattr(ifo, "maximum_frequency", None) is None:
            ifo_max_default = float(ifo.strain_data.sampling_frequency / 2.0)
        else:
            ifo_max_default = float(ifo.maximum_frequency)

        max_f = float(cal_max_dict.get(ifo.name, ifo_max_default))
        if max_f <= min_f:
            raise ValueError(
                f"Invalid calibration spline frequency bounds for {ifo.name}: "
                f"min_f={min_f}, max_f={max_f}."
            )

        per_ifo_prefix = f"{base_prefix}{ifo.name}_"

        ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
            prefix=per_ifo_prefix,
            minimum_frequency=min_f,
            maximum_frequency=max_f,
            n_points=int(n_points),
        )

        info["per_ifo"][ifo.name] = {"prefix": per_ifo_prefix, "min_f": min_f, "max_f": max_f}

    return info


def _check_expected_spline_keys(sample: Dict[str, Any], ifos: InterferometerList) -> None:
    """Validate that posterior sample contains expected calibration parameter keys."""
    n = int(ifos[0].calibration_model.n_points)
    for ifo in ifos:
        pref = ifo.calibration_model.prefix
        for i in range(n):
            for kind in ("amplitude", "phase"):
                k = f"{pref}{kind}_{i}"
                if k not in sample:
                    near = [kk for kk in sample.keys() if ifo.name in kk and 
                           ("amp" in kk.lower() or "phase" in kk.lower() or 
                            "phi" in kk.lower() or "recalib" in kk.lower())]
                    near = sorted(near)[:60]
                    raise KeyError(
                        f"Missing calibration key '{k}' in posterior sample.\n"
                        f"Expected bilby spline keys like '{pref}amplitude_0' and '{pref}phase_0'.\n"
                        f"Some nearby keys (first 60):\n" + "\n".join(near)
                    )


# ----------------------------
# bilby objects
# ----------------------------

def _build_waveform_generator_bbh(cfg: AnalysisConfig) -> WaveformGenerator:
    """Build bilby WaveformGenerator for binary black hole signals."""
    waveform_arguments = dict(
        waveform_approximant=cfg.waveform_approximant,
        reference_frequency=cfg.reference_frequency,
    )
    return WaveformGenerator(
        duration=cfg.duration,
        sampling_frequency=cfg.sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )


def _build_ifos_with_psd_and_strain(
    data,
    cfg: AnalysisConfig,
    frame_dir: Optional[str] = None,
    glitch_channel_format: str = GLITCH_SUBTRACTED_CHANNEL_FORMAT,
) -> InterferometerList:
    """Build interferometer list with PSDs and strain data."""
    strain = _download_gwosc_strain(
        cfg.detectors, cfg.start_time, cfg.end_time, cfg.sampling_frequency,
        frame_dir=frame_dir, glitch_channel_format=glitch_channel_format,
        skip_missing=True,
    )

    available = tuple(sorted(strain.keys()))
    if set(available) != set(cfg.detectors):
        LOGGER.warning(
            "Requested detectors=%s but strain available only for %s. Proceeding with available detectors.",
            cfg.detectors, available
        )

    # Build IFO objects and set strain (ONLY available)
    ifos = InterferometerList([])
    for det in available:
        ifo = bilby.gw.detector.get_empty_interferometer(det)

        ifo.minimum_frequency = float(cfg.minimum_frequency.get(det, 20.0))
        if cfg.maximum_frequency is not None and det in cfg.maximum_frequency:
            ifo.maximum_frequency = float(cfg.maximum_frequency[det])

        ifo.strain_data.set_from_gwpy_timeseries(strain[det])
        ifo.strain_data.roll_off = 0.25 * cfg.duration
        ifos.append(ifo)

    # Attach PSDs embedded in PESummary (ONLY for available)
    psd_dict = data.psd[cfg.label]
    for ifo in ifos:
        if ifo.name not in psd_dict:
            LOGGER.warning("No PSD in PESummary for %s; leaving PSD unset.", ifo.name)
            continue
        psd = psd_dict[ifo.name]
        freq = np.asarray(psd.frequencies)
        psd_vals = np.asarray(psd.strains)
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=freq,
            psd_array=psd_vals,
        )

    return ifos


# ----------------------------
# Calibration key handling
# ----------------------------

_CAL_KIND_SYNONYMS = {
    "amplitude": ["amplitude", "amp", "dA", "deltaA", "delta_amplitude"],
    "phase": ["phase", "phi", "dphi", "deltaPhi", "delta_phase"],
}

_SPCAL_PATTERNS = {
    "amplitude": [
        "{ifo}_spcal_amp_{i}",
        "{ifo}_spcal_amplitude_{i}",
        "spcal_{ifo}_amp_{i}",
        "spcal_{ifo}_amplitude_{i}",
    ],
    "phase": [
        "{ifo}_spcal_phi_{i}",
        "{ifo}_spcal_phase_{i}",
        "spcal_{ifo}_phi_{i}",
        "spcal_{ifo}_phase_{i}",
    ],
}


class IdentityCalibration:
    """Calibration model that applies no calibration (factor = 1)."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.n_points = 0  # for compatibility with code that inspects n_points

    def get_calibration_factor(self, frequency_array, prefix: str = "", **parameters):
        # bilby expects a complex array of same length as frequency_array
        return np.ones_like(np.asarray(frequency_array), dtype=np.complex128)


def _find_cal_key(sample_keys: List[str], ifo: str, kind: str, i: int) -> Optional[str]:
    syns = _CAL_KIND_SYNONYMS[kind]
    keys_lut = {k.lower(): k for k in sample_keys}  # lower -> original

    # 0) First: exact known spcal patterns
    for pat in _SPCAL_PATTERNS.get(kind, []):
        want_l = pat.format(ifo=ifo.lower(), i=i)
        # sample keys might have original case like "H1_spcal_amp_0"
        # so try both lower(ifo) and raw ifo just in case:
        for want in (
            pat.format(ifo=ifo, i=i),
            pat.format(ifo=ifo.lower(), i=i),
            pat.format(ifo=ifo.upper(), i=i),
        ):
            hit = keys_lut.get(want.lower())
            if hit is not None:
                return hit

    # 1) Your existing fuzzy logic (recalib_* etc.)
    idx_pat = rf"(?:_{i}\b|{i}\b)"
    candidates = []
    for k in sample_keys:
        kl = k.lower()
        if ifo.lower() not in kl:
            continue
        if not any(s.lower() in kl for s in syns):
            continue
        if re.search(idx_pat, kl) is None:
            continue
        candidates.append(k)

    if not candidates:
        return None

    want = f"recalib_{ifo}_{kind}_{i}"
    if want in candidates:
        return want

    pref = f"recalib_{ifo}_"
    starts = [k for k in candidates if k.startswith(pref)]
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:
        full_kind = [k for k in starts if kind in k.lower()]
        if len(full_kind) == 1:
            return full_kind[0]
        return sorted(starts)[0]

    return sorted(candidates)[0]


def _posterior_has_any_calibration_keys(
    samples: List[Dict[str, Any]],
    ifo_names: Tuple[str, ...],
    *,
    calibration_prefix: str = "recalib_",
    n_points_probe: int = 3,
) -> bool:
    """
    Return True if any of the first few samples appears to contain calibration parameters.

    We check:
      - keys starting with the expected prefix (e.g. 'recalib_')
      - spcal-style patterns handled by _find_cal_key
    """
    probe = samples[: min(10, len(samples))]
    for s in probe:
        keys = list(s.keys())

        # quick prefix check
        if any(str(k).startswith(calibration_prefix) for k in keys):
            return True

        # try a small number of indices with your existing matcher
        for ifo in ifo_names:
            for kind in ("amplitude", "phase"):
                for i in range(n_points_probe):
                    if _find_cal_key(keys, ifo=ifo, kind=kind, i=i) is not None:
                        return True

    return False


def _ensure_bilby_calibration_keys(
    sample: Dict[str, Any],
    ifo_names: Tuple[str, ...],
    n_points: int,
) -> Dict[str, Any]:
    """
    Return a copy of sample with calibration keys normalized to bilby conventions.
    
    Adds aliases so bilby CubicSpline can find:
      recalib_{IFO}_amplitude_i and recalib_{IFO}_phase_i
    """
    keys = list(sample.keys())
    out = dict(sample)

    for ifo in ifo_names:
        for kind in ("amplitude", "phase"):
            for i in range(int(n_points)):
                expected = f"recalib_{ifo}_{kind}_{i}"
                if expected in out:
                    continue

                found = _find_cal_key(keys, ifo=ifo, kind=kind, i=i)
                if found is None:
                    nearby = [k for k in keys if ifo in k and 
                             ("cal" in k.lower() or "amp" in k.lower() or 
                              "phase" in k.lower() or "phi" in k.lower())]
                    nearby = sorted(nearby)[:40]
                    raise KeyError(
                        f"Could not find a posterior key for {expected}.\n"
                        f"bilby expects keys like 'recalib_{ifo}_{kind}_{i}'.\n"
                        f"Here are some calibration-ish keys for {ifo} I can see (first 40):\n"
                        + "\n".join(nearby)
                    )

                out[expected] = out[found]

    return out


# ----------------------------
# Core computation function
# ----------------------------

def compute_one_sample_fd(
    ifos: InterferometerList,
    waveform_generator: WaveformGenerator,
    sample: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    pols = waveform_generator.frequency_domain_strain(sample)

    cal_model = getattr(ifos[0], "calibration_model", None) if len(ifos) else None
    use_spline = cal_model is not None and getattr(cal_model, "n_points", 0) and int(cal_model.n_points) > 0

    if use_spline:
        n_points = int(ifos[0].calibration_model.n_points)
        sample_used = _ensure_bilby_calibration_keys(sample, tuple(ifo.name for ifo in ifos), n_points)
    else:
        sample_used = sample

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ifo in ifos:
        model_fd = np.asarray(ifo.get_detector_response(pols, sample_used))
        data_fd = np.asarray(ifo.strain_data.frequency_domain_strain)
        out[ifo.name] = {"model_fd": model_fd, "residual_fd": data_fd - model_fd}
    return out


# ----------------------------
# Sample iteration helpers
# ----------------------------

def _iter_samples_as_dicts(data, label: str, max_samples: Optional[int], thin: int) -> Iterator[Dict[str, Any]]:
    """Iterate over posterior samples as dictionaries."""
    df = data.samples_dict[label].to_pandas()
    if thin and thin > 1:
        df = df.iloc[::thin]
    if max_samples is not None:
        df = df.iloc[:max_samples]
    for _, row in df.iterrows():
        yield row.to_dict()


def _fd_to_td(x_fd: np.ndarray, sampling_frequency: float, duration: float) -> np.ndarray:
    """Convert frequency-domain to time-domain signal."""
    n = int(round(sampling_frequency * duration))
    return np.fft.irfft(x_fd, n=n)


# ----------------------------
# High-level convenience function
# ----------------------------

def compute_bbh_residuals_with_spline_calibration(
    pesummary_h5: str,
    event: str,
    *,
    label: Optional[str] = None,
    max_samples: Optional[int] = 200,
    thin: int = 1,
    return_time_domain: bool = False,
    sanity_check_calibration_params: bool = True,
    calibration_prefix: str = "recalib_",
    default_spline_n_points: int = 10,
    loglevel: str = "INFO",
    frame_dir: Optional[str] = None,
    glitch_channel_format: str = GLITCH_SUBTRACTED_CHANNEL_FORMAT,
) -> Dict[str, Any]:
    """
    Compute data-model residuals for BBH posterior samples with spline calibration.

    If the PESummary posterior does not contain calibration parameters, this
    function will automatically disable calibration (identity response) and
    compute residuals without attempting to normalize/check calibration keys.
    """
    logging.basicConfig(level=getattr(logging, loglevel.upper(), logging.INFO))

    data = pesummary_read(pesummary_h5)
    use_label = _choose_label(data, label)
    cfg = _parse_analysis_config(data, use_label, event)

    ifos = _build_ifos_with_psd_and_strain(
        data, cfg,
        frame_dir=frame_dir,
        glitch_channel_format=glitch_channel_format,
    )
    used_detectors = tuple(ifo.name for ifo in ifos)
    wfgen = _build_waveform_generator_bbh(cfg)

    # ---- Load samples early so we can decide whether calibration is available ----
    samples: List[Dict[str, Any]] = list(_iter_samples_as_dicts(data, use_label, max_samples, thin))
    if len(samples) == 0:
        raise RuntimeError("No posterior samples selected (check max_samples/thin).")

    # ---- Decide whether to enable calibration based on posterior contents ----
    enable_calibration = _posterior_has_any_calibration_keys(
        samples,
        used_detectors,
        calibration_prefix=calibration_prefix,
    )

    cfg_dict = data.config[cfg.label]
    if enable_calibration:
        calibration_info = _attach_spline_calibration_from_config(
            ifos,
            cfg_dict,
            used_detectors,
            base_prefix=calibration_prefix,
            default_n_points=default_spline_n_points,
        )
        LOGGER.info("Calibration enabled. Spline settings: %s", calibration_info)

        if sanity_check_calibration_params:
            probe = samples[: min(10, len(samples))]
            any_pref = any(any(str(k).startswith(calibration_prefix) for k in s.keys()) for s in probe)
            if not any_pref:
                LOGGER.warning(
                    "Calibration enabled, but did not see any keys starting with '%s' in the first few samples. "
                    "Key mapping will rely on fuzzy matching (spcal patterns etc.).",
                    calibration_prefix,
                )

        # Validate expected spline keys (only when enabled)
        n_points = int(ifos[0].calibration_model.n_points)
        s0_norm = _ensure_bilby_calibration_keys(samples[0], used_detectors, n_points)
        _check_expected_spline_keys(s0_norm, ifos)
    else:
        # No calibration keys: attach identity calibration model so bilby doesn't crash
        for ifo in ifos:
            ifo.calibration_model = IdentityCalibration(prefix=f"{calibration_prefix}{ifo.name}_")

        calibration_info = {"enabled": False, "reason": "No calibration keys found in posterior samples."}
        LOGGER.info("Calibration disabled: no calibration keys found in posterior samples.")

    # ---- Pre-allocate output arrays ----
    first = compute_one_sample_fd(ifos, wfgen, samples[0])

    fd_out: Dict[str, Dict[str, np.ndarray]] = {}
    td_out: Dict[str, Dict[str, np.ndarray]] = {}

    for ifo_name, d in first.items():
        nfreq = d["model_fd"].shape[0]
        fd_out[ifo_name] = {
            "model": np.empty((len(samples), nfreq), dtype=np.complex128),
            "residual": np.empty((len(samples), nfreq), dtype=np.complex128),
        }
        if return_time_domain:
            nt = int(round(cfg.sampling_frequency * cfg.duration))
            td_out[ifo_name] = {"residual": np.empty((len(samples), nt), dtype=np.float64)}

    # ---- Main computation loop ----
    for i, s in enumerate(samples):
        r = compute_one_sample_fd(ifos, wfgen, s)
        for ifo_name, d in r.items():
            fd_out[ifo_name]["model"][i, :] = d["model_fd"]
            fd_out[ifo_name]["residual"][i, :] = d["residual_fd"]
            if return_time_domain:
                td_out[ifo_name]["residual"][i, :] = _fd_to_td(
                    d["residual_fd"], cfg.sampling_frequency, cfg.duration
                )

    out: Dict[str, Any] = {
        "config": cfg,
        "used_detectors": used_detectors,
        "ifos": ifos,
        "waveform_generator": wfgen,
        "calibration_info": calibration_info,
        "samples": samples,
        "fd": fd_out,
    }
    if return_time_domain:
        out["td"] = td_out
    return out
