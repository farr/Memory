#!/usr/bin/env python3
"""Whitened time-domain strain for GW230606_004305.

Whitening procedure:
  1. Load a 128-second segment (same as for the spectrogram) for both
     H1 (BayesWave cleaned frame) and L1 (GWOSC).
  2. Compute a Welch/median ASD from the full 128-second segment
     (fftlength=4 s, 50 % overlap, Hann window).
  3. Whiten using gwpy TimeSeries.whiten(asd=...) — applies a time-domain
     FIR whitening filter.
  4. Crop the whitened series to the 4-second analysis window for display.

Outputs: results/gw230606_whitened_td.png
"""

import os
import sys
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory.gw_residuals import (
    _parse_analysis_config,
    _download_gwosc_strain,
    _find_frame_file,
    _read_frame_strain,
    _choose_label,
    GLITCH_SUBTRACTED_CHANNEL_FORMAT,
)
from pesummary.io import read as pesummary_read

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

EVENT        = "GW230606_004305"
PESUMMARY_H5 = (
    "/mnt/home/ccalvk/ceph/GWTC-4/"
    "IGWN-GWTC4p0-1a206db3d_721-GW230606_004305-combined_PEDataRelease.hdf5"
)
PARAM_KEY = "C00:NRSur7dq4"

# Padding for Welch ASD estimation and FIR filter settling
QTRANSFORM_PAD = 64.0   # same as spectrogram — 128 s total

# Welch ASD parameters
FFTLENGTH = 4.0    # seconds — matches analysis segment, df = 0.25 Hz
OVERLAP   = 2.0    # 50 % overlap

REPO_DIR  = os.path.join(os.path.dirname(__file__), "..")
FRAME_DIR = os.path.join(REPO_DIR, "data", "frames")
OUTFILE   = os.path.join(REPO_DIR, "results", "gw230606_whitened_td.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    # --- Parse analysis config ---
    print("Reading PESummary config ...")
    data  = pesummary_read(PESUMMARY_H5)
    label = _choose_label(data, PARAM_KEY)
    cfg   = _parse_analysis_config(data, label, EVENT)
    gps   = cfg.trigger_time
    print(f"  segment: [{cfg.start_time:.2f}, {cfg.end_time:.2f}]  ({cfg.duration} s)")

    t_long_start = gps - QTRANSFORM_PAD
    t_long_end   = gps + QTRANSFORM_PAD

    # --- Load H1 BayesWave frame (128 s) ---
    print("Loading H1 BayesWave frame ...")
    gwf = _find_frame_file(FRAME_DIR, "H1", t_long_start, t_long_end)
    if gwf is None:
        raise FileNotFoundError(
            f"No H1 BayesWave frame in {FRAME_DIR}. "
            "Run data/download_glitch_frames.sh first."
        )
    h1_long = _read_frame_strain(gwf, "H1", t_long_start, t_long_end,
                                 channel_format=GLITCH_SUBTRACTED_CHANNEL_FORMAT)
    if h1_long.sample_rate.value != 4096.0:
        print(f"  Resampling H1 {h1_long.sample_rate} -> 4096 Hz ...")
        h1_long = h1_long.resample(4096.0)

    # --- Load L1 GWOSC (128 s) ---
    print("Loading L1 GWOSC ...")
    l1_long = _download_gwosc_strain(
        ["L1"], t_long_start, t_long_end, fs=4096.0, frame_dir=None,
    )["L1"]

    # --- Welch ASD from the 128-second segment ---
    asd_kwargs = dict(fftlength=FFTLENGTH, overlap=OVERLAP,
                      window="hann", method="median")
    print("Computing Welch ASDs ...")
    asd_h1 = h1_long.asd(**asd_kwargs)
    asd_l1 = l1_long.asd(**asd_kwargs)
    print(f"  H1 ASD: {len(asd_h1)} frequency bins, "
          f"df={asd_h1.df.value:.3f} Hz")
    print(f"  L1 ASD: {len(asd_l1)} frequency bins, "
          f"df={asd_l1.df.value:.3f} Hz")

    # --- Whiten the full 128-second segment, then crop ---
    print("Whitening ...")
    h1_white = h1_long.whiten(asd=asd_h1)
    l1_white = l1_long.whiten(asd=asd_l1)

    h1_plot = h1_white.crop(cfg.start_time, cfg.end_time)
    l1_plot = l1_white.crop(cfg.start_time, cfg.end_time)
    times_h1 = h1_plot.times.value - gps
    times_l1 = l1_plot.times.value - gps

    # --- Plot ---
    plt.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    axes[0].plot(times_h1, h1_plot.value, lw=0.7, color="C1",
                 label=f"H1  BayesWave ({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo='H1')})")
    axes[0].axhline(0, color="k", lw=0.4, ls="--")
    axes[0].set_ylabel("Whitened strain")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(
        f"{EVENT}  |  GPS {gps:.2f}  |  gwpy whitened time-domain strain\n"
        f"Welch ASD: {2*QTRANSFORM_PAD:.0f}-s segment, "
        f"fftlength={FFTLENGTH:.0f} s, overlap={OVERLAP:.0f} s, Hann, median"
    )

    axes[1].plot(times_l1, l1_plot.value, lw=0.7, color="C0",
                 label="L1  GWOSC (fetch_open_data)")
    axes[1].axhline(0, color="k", lw=0.4, ls="--")
    axes[1].set_ylabel("Whitened strain")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel(f"Time from GPS {gps:.2f} [s]")

    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.4)

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
