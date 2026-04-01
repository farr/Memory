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
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# Ensure the bundled waveform data directory is on LAL_DATA_PATH so that
# LALSimulation can find NRSur7dq4_v1.0.h5 and similar data files.
_PACKAGE_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
if os.path.isdir(_PACKAGE_DATA_DIR):
    _lal_path = os.environ.get("LAL_DATA_PATH", "")
    _lal_dirs = _lal_path.split(":") if _lal_path else []
    if _PACKAGE_DATA_DIR not in _lal_dirs:
        os.environ["LAL_DATA_PATH"] = (
            _PACKAGE_DATA_DIR + (":" + _lal_path if _lal_path else "")
        )

from pesummary.io import read as pesummary_read
from gwosc.datasets import event_gps, event_detectors, find_datasets
from gwpy.timeseries import TimeSeries

import bilby
from bilby.gw.detector.psd import PowerSpectralDensity
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator, GWSignalWaveformGenerator
import lalsimulation as _lalsim


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
    waveform_minimum_frequency: Optional[float] = None
    """Waveform-generation starting frequency (the 'waveform' key in
    bilby_pipe's per-IFO minimum-frequency dict).  When present this is the
    frequency passed to the WaveformGenerator, which may differ from the IFO
    noise-floor minimum (e.g. GW230704 SpinTaylor: IFO=20 Hz, waveform=9 Hz)."""


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
    """fres
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

        # Try to handle dict-like strings with bare keys, e.g.:
        #   "{H1: 20, L1: 20}"  or  "{H1: 20, L1: 20, waveform: 9.0}"
        # The "waveform" key sets the waveform-generation start frequency only;
        # it is NOT a fallback for per-IFO likelihood-integral lower limits.
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            for d in detectors:
                s = s.replace(f"{d}:", f"'{d}':")
            # Quote any remaining bare word keys (e.g. "waveform:") so
            # ast.literal_eval doesn't choke on them.
            s = re.sub(r"(?<!['\"])\b([a-zA-Z_]\w*)\s*:", r"'\1':", s)
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


def _parse_waveform_fmin(value: Any) -> Optional[float]:
    """Extract the 'waveform' key from a bilby_pipe minimum-frequency string.

    bilby_pipe allows ``minimum-frequency = "{H1:20, L1:20, waveform:9.0}"``
    where ``waveform`` sets the starting frequency for waveform generation
    independently of the per-IFO noise-floor minimum.  Returns that value,
    or None if the key is absent or the value is not a dict-like string.
    """
    if value is None:
        return None
    value = _maybe_literal(value)
    if isinstance(value, dict):
        v = value.get("waveform")
        return float(v) if v is not None else None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    s = re.sub(r"(?<!['\"])\b([a-zA-Z_]\w*)\s*:", r"'\1':", s)
    try:
        dct = ast.literal_eval(s)
        v = dct.get("waveform") if isinstance(dct, dict) else None
        return float(v) if v is not None else None
    except Exception:
        return None


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
    """Infer waveform approximant from config.

    GWTC files store this as ``waveform-approximant`` in the flat ``[config]``
    section (bilby_pipe convention).
    """
    flat = cfg_dict.get("config", {})
    # bilby_pipe: hyphen or underscore; LALInference: [engine] approx
    val = (flat.get("waveform-approximant") or flat.get("waveform_approximant")
           or cfg_dict.get("engine", {}).get("approx"))
    if val is not None:
        return str(_maybe_literal(val))
    return str(default)


# ----------------------------
# Config -> AnalysisConfig
# ----------------------------

def _parse_analysis_config(data, label: str, event: str) -> AnalysisConfig:
    """Parse PESummary config into AnalysisConfig dataclass.

    All GWTC files (GWTC-2.1, GWTC-3, GWTC-4) use a single flat ``[config]``
    section with bilby_pipe-style hyphenated keys.  We read directly from there
    and fall back to GWOSC lookups only when a key is absent.
    """
    cfg = data.config[label]

    # Three config conventions in the GWTC catalog:
    #   bilby_pipe hyphen  (GWTC-3/4 BBH, most GWTC-2.1): flat [config] with
    #                       hyphenated keys, e.g. "sampling-frequency".
    #   bilby_pipe underscore (GWTC-4 NSBH): flat [config] with underscore
    #                       keys, e.g. "sampling_frequency".
    #   LALInference (older GWTC-2.1): sectioned INI — [engine], [analysis],
    #                       [lalinference].
    flat = cfg.get("config", {})          # bilby_pipe (both hyphen & underscore)
    engine = cfg.get("engine", {})        # LALInference
    analysis = cfg.get("analysis", {})    # LALInference
    lalinf = cfg.get("lalinference", {})  # LALInference

    def _flat(hyphen_key):
        """Look up a bilby_pipe key in the flat [config] section, trying both
        the hyphenated form (e.g. 'sampling-frequency') and the underscore
        form (e.g. 'sampling_frequency')."""
        v = flat.get(hyphen_key)
        if v is None:
            v = flat.get(hyphen_key.replace("-", "_"))
        return v

    def _first(*vals):
        """Return first non-None value after _maybe_literal conversion."""
        for v in vals:
            if v is not None:
                return _maybe_literal(v)
        return None

    # Accept either a bare GWOSC event name ("GW230608_205047") or a
    # filename that contains one ("IGWN-...-GW230608_205047-....hdf5").
    gw_name_match = re.search(r"(GW\d{6}_\d{6})", event)
    gw_name = gw_name_match.group(1) if gw_name_match else event

    # --- 1) Detectors ---
    # bilby_pipe: [config] detectors; LALInference: [analysis] ifos
    dets_raw = _first(_flat("detectors"), analysis.get("ifos"))
    if dets_raw is None:
        detectors_cfg = tuple(sorted(event_detectors(gw_name)))
    elif isinstance(dets_raw, (list, tuple)):
        detectors_cfg = tuple(str(d).strip() for d in dets_raw if str(d).strip())
    else:
        s = str(dets_raw).strip()
        try:
            detectors_cfg = tuple(str(d).strip() for d in ast.literal_eval(s) if str(d).strip())
        except Exception:
            detectors_cfg = tuple(p.strip() for p in s.replace(",", " ").split() if p.strip())

    # --- 2) OVERRIDE with detectors present in PESummary PSDs ---
    # This handles cases where the config lists fewer IFOs than the run used.
    psd_for_label = None
    try:
        psd_for_label = data.psd[label]
    except Exception:
        pass
    if isinstance(psd_for_label, dict) and len(psd_for_label) > 0:
        detectors = tuple(sorted(psd_for_label.keys()))
    else:
        detectors = detectors_cfg

    # --- 3) Trigger time ---
    # bilby_pipe: [config] trigger-time
    # LALInference: not in config; read median geocent_time from posterior samples,
    # falling back to GWOSC (which may require a shorter event name like "GW170608").
    trig_raw = _flat("trigger-time")
    if trig_raw is not None:
        trig = float(trig_raw)
    else:
        try:
            df_t = data.samples_dict[label].to_pandas()
            trig = float(df_t["geocent_time"].median())
        except Exception:
            trig = float(event_gps(gw_name))

    def _warn_default(name, value):
        LOGGER.warning(
            "%s/%s: '%s' not found in config; using default %r. "
            "Config sections: %s",
            event, label, name, value, sorted(cfg.keys()),
        )
        return value

    # --- 4) Segment duration and start/end times ---
    # bilby_pipe: [config] duration + post-trigger-duration
    # LALInference: [engine] seglen; post-trigger always 2 s by convention
    duration_raw = _first(_flat("duration"), engine.get("seglen"))
    duration = float(duration_raw) if duration_raw is not None else float(_warn_default("duration", 4.0))
    post_trigger_raw = _flat("post-trigger-duration")
    post_trigger = float(post_trigger_raw) if post_trigger_raw is not None else float(_warn_default("post-trigger-duration", 2.0))
    start_time = trig - (duration - post_trigger)
    end_time = trig + post_trigger

    # --- 5) Sampling frequency ---
    # bilby_pipe: [config] sampling-frequency; LALInference: [engine] srate
    fs_raw = _first(_flat("sampling-frequency"), engine.get("srate"))
    sampling_frequency = float(fs_raw) if fs_raw is not None else float(_warn_default("sampling-frequency", 4096.0))

    # --- 6) Per-IFO frequency bounds ---
    # bilby_pipe: [config] minimum/maximum-frequency
    # LALInference: [lalinference] flow / fhigh
    min_freq_raw = _first(_flat("minimum-frequency"), lalinf.get("flow"))
    if min_freq_raw is None:
        min_freq_raw = _warn_default("minimum-frequency", 20.0)
    min_freq = _parse_ifo_freq_dict(min_freq_raw, detectors, default=20.0)
    # Extract the optional "waveform:" key from the minimum-frequency dict.
    # bilby_pipe uses this to set a lower waveform-generation start frequency
    # that differs from the per-IFO noise-floor minimum (e.g. GW230704 SpinTaylor:
    # IFO noise floor = 20 Hz, waveform generation starts at 9 Hz = f_ref).
    waveform_fmin = _parse_waveform_fmin(min_freq_raw)
    max_freq_raw = _first(_flat("maximum-frequency"), lalinf.get("fhigh"))
    max_freq = (
        _parse_ifo_freq_dict(max_freq_raw, detectors, default=sampling_frequency / 2.0)
        if max_freq_raw is not None
        else None
    )

    # --- 7) Reference frequency ---
    # bilby_pipe: [config] reference-frequency; LALInference: [engine] fref
    fref_raw = _first(_flat("reference-frequency"), engine.get("fref"))
    reference_frequency = float(fref_raw) if fref_raw is not None else float(_warn_default("reference-frequency", 50.0))

    # --- 8) Waveform approximant ---
    # bilby_pipe: [config] waveform-approximant; LALInference: [engine] approx
    approx_raw = _first(_flat("waveform-approximant"), engine.get("approx"))
    waveform_approximant = str(approx_raw) if approx_raw is not None else str(_warn_default("waveform-approximant", "IMRPhenomPv2"))

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
        waveform_minimum_frequency=waveform_fmin,
        config_dict=cfg,
    )


# ----------------------------
# Spline calibration settings from config
# ----------------------------

def _infer_spline_npoints_from_config(cfg_dict: Dict[str, Dict[str, Any]], default: int = 10) -> int:
    """Infer number of spline calibration points from config.

    GWTC files store this as ``spline-calibration-nodes`` in the flat ``[config]``
    section.
    """
    flat = cfg_dict.get("config", {})
    # bilby_pipe: hyphen or underscore form
    val = flat.get("spline-calibration-nodes") or flat.get("spline_calibration_nodes")
    # LALInference: [engine] spcal-nodes
    if val is None:
        val = cfg_dict.get("engine", {}).get("spcal-nodes", None)
    if val is not None:
        try:
            return int(_maybe_literal(val))
        except Exception:
            pass
    return int(default)


def _infer_calib_minmax_freq_from_config(
    cfg_dict: Dict[str, Dict[str, Any]],
    detectors: Tuple[str, ...],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Infer per-IFO calibration spline min/max frequencies from config.

    GWTC files do not store explicit calibration frequency bounds; the spline
    covers the full analysis band.  Return empty dicts so the caller falls back
    to the IFO min/max frequency.
    """
    return {}, {}


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

def _is_gwsignal_only_approximant(name: str) -> bool:
    """Return True if this approximant is not available as a native LAL integer enum."""
    try:
        _lalsim.SimInspiralGetApproximantFromString(name)
        return False
    except Exception:
        return True


def _build_waveform_generator_bbh(cfg: AnalysisConfig) -> WaveformGenerator:
    """Build bilby WaveformGenerator for binary black hole signals.

    Uses GWSignalWaveformGenerator for approximants only available via gwsignal
    (e.g. SEOBNRv5PHM via pyseobnr), and standard WaveformGenerator otherwise.
    """
    waveform_arguments = dict(
        waveform_approximant=cfg.waveform_approximant,
        reference_frequency=cfg.reference_frequency,
    )
    # If the PE config specifies a separate waveform-generation minimum frequency
    # (the "waveform:" key in bilby_pipe's per-IFO minimum-frequency dict), pass
    # it explicitly so the WaveformGenerator uses it rather than bilby's 20 Hz
    # default.  This matters when f_ref < IFO noise-floor minimum (e.g. GW230704
    # SpinTaylor: IFO=20 Hz, waveform=f_ref=9 Hz).
    if cfg.waveform_minimum_frequency is not None:
        waveform_arguments["minimum_frequency"] = cfg.waveform_minimum_frequency
    # Merge any extra waveform arguments from the PE config
    # (e.g. waveform_arguments_dict = {'lmax_nyquist': 1} for SEOBNRv5PHM on
    # low-mass NSBH events where the ringdown exceeds Nyquist at 4096 Hz).
    flat = cfg.config_dict.get("config", {})
    raw_wf_args = flat.get("waveform_arguments_dict") or flat.get("waveform-arguments-dict")
    if raw_wf_args is not None:
        extra = _maybe_literal(raw_wf_args)
        if isinstance(extra, dict):
            waveform_arguments.update(extra)
    if _is_gwsignal_only_approximant(cfg.waveform_approximant):
        return GWSignalWaveformGenerator(
            duration=cfg.duration,
            sampling_frequency=cfg.sampling_frequency,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
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
# C-level stderr capture helper
# ----------------------------

class _CaptureCStderr:
    """Context manager that captures C-level stderr (fd 2).

    LAL/XLAL error messages are written directly to the C file descriptor,
    not to Python's sys.stderr, so they never appear in Python exception
    strings.  Capturing fd 2 lets us parse XLAL messages (e.g. the ISCO
    frequency limit from SEOBNRv4PHM) after the call raises.

    After __exit__ the captured text is re-emitted to real stderr so that
    messages are not silently swallowed.
    """

    def __init__(self):
        self.captured = ""

    def __enter__(self):
        self._saved_fd = os.dup(2)
        self._r, w = os.pipe()
        os.dup2(w, 2)
        os.close(w)
        return self

    def __exit__(self, *_):
        # Restore real stderr (closes the pipe write end held by fd 2).
        os.dup2(self._saved_fd, 2)
        os.close(self._saved_fd)
        # Drain pipe — write end is now closed so os.read returns b"" at EOF.
        chunks = []
        while True:
            try:
                data = os.read(self._r, 65536)
                if not data:
                    break
                chunks.append(data)
            except OSError:
                break
        os.close(self._r)
        self.captured = b"".join(chunks).decode("utf-8", errors="replace")
        if self.captured:
            os.write(2, self.captured.encode())  # fd 2 is real stderr again


# ----------------------------
# Core computation function
# ----------------------------

def compute_one_sample_fd(
    ifos: InterferometerList,
    waveform_generator: WaveformGenerator,
    sample: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    f_ref_orig = waveform_generator.waveform_arguments.get("reference_frequency", 20.0)
    tried_fmin_reduction = False          # for "fRef < f_min"
    tried_lmax_nyquist = False            # for "ringdown freq > Nyquist" (SEOBNRv5PHM via lmax_nyquist)
    tried_eob_nyquist_check = False       # for "ringdown freq > Nyquist" (LAL SEOBNRv4PHM via EOBEllMaxForNyquistCheck)
    _FMIN_FLOOR = 1.0             # Hz — never go below this for ISCO retries
    _seen_isco_limit = None       # track the highest ISCO limit we've encountered
    sample_try = sample
    while True:
        _stderr_cap = _CaptureCStderr()
        try:
            with _stderr_cap:
                pols = waveform_generator.frequency_domain_strain(sample_try)
            break
        except Exception as exc:
            msg = str(exc).lower()
            f_ref_curr = waveform_generator.waveform_arguments.get("reference_frequency", f_ref_orig)

            # SEOBNRv4PHM raises "Initial frequency is too high, the limit is X Hz"
            # when minimum_frequency > the ISCO frequency for a particular sample's
            # mass/spin parameters.  Parse X from the message and lower
            # minimum_frequency to just below it.  This is distinct from the
            # f_ref < f_min case below; both produce "internal function call failed".
            #
            # XLAL writes the detail ("the limit is X") only to C-level stderr,
            # NOT to the Python exception string.  We capture C stderr via
            # _CaptureCStderr and search there as well as in the exception text.
            _combined = str(exc) + _stderr_cap.captured
            _combined_lower = _combined.lower()
            isco_limit = None
            isco_match = re.search(r"the limit is ([\d.]+)", _combined, re.IGNORECASE)
            if isco_match and ("initial frequency is too high" in _combined_lower
                               or "intitial frequency is too high" in _combined_lower):
                isco_limit = float(isco_match.group(1))

            # "internal function call failed" without nyquist/ringdown text signals
            # fRef < f_min (IMRPhenomXO4a, NRSur7dq4, etc.).  Fix by lowering
            # minimum_frequency to f_ref — never change f_ref itself, as that
            # would alter the spin-angle convention for the sample.
            # Guard: don't fire this on the ISCO-limit error (which is unrelated
            # to f_ref and produces the same "internal function call failed" text).
            is_freq_too_low = (
                "internal function call failed" in msg
                and "nyquist" not in msg
                and "ringdown" not in msg
                and isco_limit is None
                and f_ref_curr < 21.0
            )
            # pyseobnr (SEOBNRv5PHM via gwsignal) raises "ringdown frequency of
            # (N,N) mode greater than maximum frequency from Nyquist theorem"
            # as a Python exception — detectable in `msg`.
            # LAL SEOBNRv4PHM raises "XLALEOBCheckNyquistFrequency: Ringdown
            # frequency > Nyquist frequency!" only to C-level stderr — detectable
            # in `_combined_lower` but NOT in `msg`.
            # For pyseobnr: retry with lmax_nyquist=2 then lmax_nyquist=1.
            # For LAL: retry with mode_array=[[2,2],[2,-2]] (dominant mode only)
            # to avoid the higher-mode Nyquist check entirely.
            is_nyquist_ringdown = "nyquist" in msg and "ringdown" in msg
            is_nyquist_ringdown_lal = (
                not is_nyquist_ringdown
                and "nyquist" in _combined_lower
                and "ringdown" in _combined_lower
            )
            curr_lmax = waveform_generator.waveform_arguments.get("lmax_nyquist", 4)
            curr_fmin_explicit = waveform_generator.waveform_arguments.get("minimum_frequency")
            if isco_limit is not None:
                new_fmin = 0.99 * isco_limit
                if new_fmin < _FMIN_FLOOR:
                    raise  # hit floor — give up
                if curr_fmin_explicit is not None and new_fmin >= curr_fmin_explicit:
                    raise  # already tried this or lower, no progress
                _seen_isco_limit = max(_seen_isco_limit or 0.0, isco_limit)
                LOGGER.warning(
                    "SEOB initial frequency too high (limit=%.4g Hz); "
                    "retrying with minimum_frequency=%.4g Hz",
                    isco_limit, new_fmin,
                )
                waveform_generator.waveform_arguments["minimum_frequency"] = new_fmin
                sample_try = {**sample_try, "minimum_frequency": new_fmin}
            elif is_freq_too_low and not tried_fmin_reduction and (
                _seen_isco_limit is None or f_ref_curr >= _seen_isco_limit
            ):
                # Don't lower f_min below a previously-seen ISCO limit: doing so
                # would immediately re-trigger the ISCO error, creating a deadlock
                # (e.g. GW200308 SEOBNRv4PHM: f_ref=3 Hz, ISCO=9.7 Hz).
                tried_fmin_reduction = True
                LOGGER.warning(
                    "reference_frequency=%.4g < minimum_frequency (%s); "
                    "lowering minimum_frequency to match",
                    f_ref_curr, exc,
                )
                waveform_generator.waveform_arguments["minimum_frequency"] = f_ref_curr
                sample_try = {**sample_try, "minimum_frequency": f_ref_curr}
            elif is_nyquist_ringdown and not tried_lmax_nyquist:
                tried_lmax_nyquist = True
                next_lmax = 2 if curr_lmax > 2 else 1
                LOGGER.warning(
                    "Ringdown Nyquist error (%s); retrying with lmax_nyquist=%d",
                    exc, next_lmax,
                )
                waveform_generator.waveform_arguments["lmax_nyquist"] = next_lmax
                sample_try = {**sample_try, "lmax_nyquist": next_lmax}
            elif is_nyquist_ringdown and tried_lmax_nyquist and curr_lmax > 1:
                # lmax_nyquist=2 also failed; disable check entirely
                LOGGER.warning(
                    "Ringdown Nyquist error with lmax_nyquist=2 (%s); retrying with lmax_nyquist=1",
                    exc,
                )
                waveform_generator.waveform_arguments["lmax_nyquist"] = 1
                sample_try = {**sample_try, "lmax_nyquist": 1}
            elif is_nyquist_ringdown_lal and not tried_eob_nyquist_check:
                # LAL SEOBNRv4PHM: "Ringdown frequency > Nyquist frequency!"
                # written to C-level stderr.  EOBEllMaxForNyquistCheck is a
                # LALDict key (SimInspiralWaveformParamsInsertEOBEllMaxForNyquistCheck)
                # that limits which modes are tested — same idea as lmax_nyquist=2
                # for pyseobnr.  First try ell≤2 check, then ell≤1 (disable).
                tried_eob_nyquist_check = True
                curr_eob = waveform_generator.waveform_arguments.get(
                    "EOBEllMaxForNyquistCheck", 5
                )
                next_eob = 2 if curr_eob > 2 else 1
                LOGGER.warning(
                    "LAL Nyquist ringdown error (C-stderr); retrying with "
                    "EOBEllMaxForNyquistCheck=%d",
                    next_eob,
                )
                waveform_generator.waveform_arguments["EOBEllMaxForNyquistCheck"] = next_eob
                sample_try = {**sample_try, "EOBEllMaxForNyquistCheck": next_eob}
            elif is_nyquist_ringdown_lal and tried_eob_nyquist_check:
                curr_eob = waveform_generator.waveform_arguments.get(
                    "EOBEllMaxForNyquistCheck", 5
                )
                if curr_eob > 1:
                    LOGGER.warning(
                        "LAL Nyquist error with EOBEllMaxForNyquistCheck=2; "
                        "retrying with EOBEllMaxForNyquistCheck=1 (disable check)",
                    )
                    waveform_generator.waveform_arguments["EOBEllMaxForNyquistCheck"] = 1
                    sample_try = {**sample_try, "EOBEllMaxForNyquistCheck": 1}
                else:
                    raise  # check already disabled, still failing — give up
            else:
                raise

    cal_model = getattr(ifos[0], "calibration_model", None) if len(ifos) else None
    use_spline = cal_model is not None and getattr(cal_model, "n_points", 0) and int(cal_model.n_points) > 0

    if use_spline:
        n_points = int(ifos[0].calibration_model.n_points)
        try:
            sample_used = _ensure_bilby_calibration_keys(
                sample_try, tuple(ifo.name for ifo in ifos), n_points
            )
        except KeyError:
            sample_used = sample_try
    else:
        sample_used = sample_try

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ifo in ifos:
        model_fd = np.asarray(ifo.get_detector_response(pols, sample_used))
        data_fd = np.asarray(ifo.strain_data.frequency_domain_strain)
        out[ifo.name] = {"model_fd": model_fd, "residual_fd": data_fd - model_fd}
    return out


# ----------------------------
# Sample iteration helpers
# ----------------------------

def _iter_samples_as_dicts(data, label: str, max_samples: Optional[int], thin: int, load_specific_sample: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Iterate over posterior samples as dictionaries."""
    df = data.samples_dict[label].to_pandas()
    if thin and thin > 1:
        df = df.iloc[::thin]
    if max_samples is not None:
        df = df.iloc[:max_samples]
    if load_specific_sample is not None:
        df = df.iloc[load_specific_sample:load_specific_sample + 1]
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
    load_specific_sample = None,
) -> Dict[str, Any]:
    """
    Compute data-model residuals for BBH posterior samples with spline calibration.

    If the PESummary posterior does not contain calibration parameters, this
    function will automatically disable calibration (identity response) and
    compute residuals without attempting to normalize/check calibration keys.
    """
    logging.basicConfig(level=getattr(logging, loglevel.upper(), logging.INFO))

    data = pesummary_read(pesummary_h5)
    if label is not None and label not in data.labels:
        raise ValueError(
            f"Label '{label}' not found in file. Available labels: {data.labels}"
        )
    use_label = _choose_label(data, label)
    cfg = _parse_analysis_config(data, use_label, event)

    raw_ifos = _build_ifos_with_psd_and_strain(
        data, cfg,
        frame_dir=frame_dir,
        glitch_channel_format=glitch_channel_format,
    )
    # Note: added to skip Virgo when NaNs are present
    ifos = []
    for ifo in raw_ifos:
        if not np.any(np.isnan(ifo.strain_data.frequency_domain_strain)):
            ifos.append(ifo)
            
    used_detectors = tuple(ifo.name for ifo in ifos)
    wfgen = _build_waveform_generator_bbh(cfg)

    # ---- Load samples early so we can decide whether calibration is available ----
    samples: List[Dict[str, Any]] = list(_iter_samples_as_dicts(data, use_label, max_samples, thin, load_specific_sample))
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
        try:
            s0_norm = _ensure_bilby_calibration_keys(samples[0], used_detectors, n_points)
            _check_expected_spline_keys(s0_norm, ifos)
        except KeyError as e:
            LOGGER.warning(
                "Calibration key lookup failed (%s); disabling calibration.", e
            )
            for ifo in ifos:
                ifo.calibration_model = IdentityCalibration(
                    prefix=f"{calibration_prefix}{ifo.name}_"
                )
            calibration_info = {"enabled": False, "reason": f"Key lookup failed: {e}"}
    else:
        # No calibration keys: attach identity calibration model so bilby doesn't crash
        for ifo in ifos:
            ifo.calibration_model = IdentityCalibration(prefix=f"{calibration_prefix}{ifo.name}_")

        calibration_info = {"enabled": False, "reason": "No calibration keys found in posterior samples."}
        LOGGER.info("Calibration disabled: no calibration keys found in posterior samples.")

    # ---- Ensure geocent_time is present in every sample ----
    # Some old LALInference posteriors omit this column (or use "time").  bilby
    # requires it, so inject from cfg.trigger_time when missing.
    _geocent_time_aliases = ("geocent_time", "time", "tc")
    for s in samples:
        if "geocent_time" not in s:
            found = next((s[k] for k in _geocent_time_aliases if k in s), None)
            s["geocent_time"] = found if found is not None else cfg.trigger_time

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
    # Save the waveform_arguments keys that compute_one_sample_fd may modify
    # (minimum_frequency, lmax_nyquist) so we can reset them before each
    # sample.  Without this, an ISCO retry that lowers minimum_frequency
    # would bleed into subsequent samples that have no ISCO issue, causing
    # a mismatch with the make_memories path (which reads config.minimum_frequency
    # fresh for every sample).
    _wfargs_orig = {
        k: wfgen.waveform_arguments[k]
        for k in ("minimum_frequency", "lmax_nyquist", "EOBEllMaxForNyquistCheck")
        if k in wfgen.waveform_arguments
    }
    for i, s in enumerate(samples):
        # Reset per-sample modifiable waveform arguments to their original state.
        for k in ("minimum_frequency", "lmax_nyquist", "EOBEllMaxForNyquistCheck"):
            if k in _wfargs_orig:
                wfgen.waveform_arguments[k] = _wfargs_orig[k]
            else:
                wfgen.waveform_arguments.pop(k, None)
        r = compute_one_sample_fd(ifos, wfgen, s)
        # Write the effective minimum_frequency used back to the sample dict so
        # that evaluate_surrogate_with_LAL (make_memories path) can use the same
        # value instead of always reading from config.
        eff_fmin = wfgen.waveform_arguments.get("minimum_frequency")
        if eff_fmin is not None:
            s["minimum_frequency"] = eff_fmin
        else:
            s.pop("minimum_frequency", None)
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
