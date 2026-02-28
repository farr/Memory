#!/usr/bin/env python3
"""Frequency-domain comparison of GWOSC and frame strain for GW230606_004305.

Applies exactly the same conditioning as gw_residuals.py / bilby:
  - segment duration, start/end time from the PESummary analysis config
  - Tukey window (roll_off=0.2 s, alpha=2*roll_off/duration)
  - rfft normalised to strain/Hz (bilby nfft convention)

The ASD shown is simply |h(f)| — the Fourier norm of the single windowed
FFT, with no Welch averaging.

Requires: frames already downloaded by plot_gw230606_data_comparison.py.

Outputs: results/gw230606_fd_comparison.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory.gw_residuals import (
    _parse_analysis_config,
    _download_gwosc_strain,
    _read_frame_strain,
    _find_frame_file,
    _choose_label,
    GLITCH_SUBTRACTED_CHANNEL_FORMAT,
)
from pesummary.io import read as pesummary_read

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVENT        = "GW230606_004305"
PESUMMARY_H5 = (
    "/mnt/home/ccalvk/ceph/GWTC-4/"
    "IGWN-GWTC4p0-1a206db3d_721-GW230606_004305-combined_PEDataRelease.hdf5"
)
PARAM_KEY    = "C00:NRSur7dq4"

REPO_DIR  = os.path.join(os.path.dirname(__file__), "..")
FRAME_DIR = os.path.join(REPO_DIR, "data", "frames")
OUTFILE   = os.path.join(REPO_DIR, "results", "gw230606_fd_comparison.png")

ROLL_OFF  = 0.2   # bilby default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bilby_window_and_fft(strain_array, duration, sampling_frequency):
    """Apply Tukey window and rfft exactly as bilby does.

    Returns (freq_array, h_tilde) where h_tilde is in strain/Hz.
    """
    alpha  = 2 * ROLL_OFF / duration
    window = tukey(len(strain_array), alpha=alpha)
    h_tilde = np.fft.rfft(strain_array * window) / sampling_frequency
    freq    = np.linspace(0, sampling_frequency / 2, len(h_tilde))
    return freq, h_tilde


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    # --- Parse analysis config from PESummary ------------------------------
    print("Reading PESummary config ...")
    data  = pesummary_read(PESUMMARY_H5)
    label = _choose_label(data, PARAM_KEY)
    cfg   = _parse_analysis_config(data, label, EVENT)

    print(f"  label:    {label}")
    print(f"  detectors: {cfg.detectors}")
    print(f"  duration:  {cfg.duration} s")
    print(f"  start:     {cfg.start_time}")
    print(f"  end:       {cfg.end_time}")
    print(f"  fs:        {cfg.sampling_frequency} Hz")
    alpha = 2 * ROLL_OFF / cfg.duration
    print(f"  Tukey alpha: {alpha:.4f}  (roll_off={ROLL_OFF} s)")

    # --- Load GWOSC strain -------------------------------------------------
    print("Loading GWOSC strain via fetch_open_data ...")
    gwosc_strain = _download_gwosc_strain(
        cfg.detectors, cfg.start_time, cfg.end_time, cfg.sampling_frequency,
        frame_dir=None,
    )

    # --- Load frame strain -------------------------------------------------
    print("Loading frame strain ...")
    frame_strain = {}
    for det in cfg.detectors:
        gwf = _find_frame_file(FRAME_DIR, det, cfg.start_time, cfg.end_time)
        if gwf is not None:
            print(f"  {det}: reading {os.path.basename(gwf)}")
            frame_strain[det] = _read_frame_strain(
                gwf, det, cfg.start_time, cfg.end_time,
                channel_format=GLITCH_SUBTRACTED_CHANNEL_FORMAT,
            )
        else:
            print(f"  {det}: no frame file found, reusing GWOSC")
            frame_strain[det] = gwosc_strain[det]

    # --- Window + FFT ------------------------------------------------------
    print("Windowing and FFT-ing ...")
    gwosc_fd = {}
    frame_fd = {}
    for det in cfg.detectors:
        ts_g = gwosc_strain[det]
        ts_f = frame_strain[det]

        # Resample frame to match GWOSC rate if needed
        if ts_f.sample_rate.value != ts_g.sample_rate.value:
            print(f"  Resampling {det} frame {ts_f.sample_rate} → {ts_g.sample_rate}")
            ts_f = ts_f.resample(ts_g.sample_rate.value)

        freqs_g, h_g = bilby_window_and_fft(
            ts_g.value, cfg.duration, cfg.sampling_frequency
        )
        freqs_f, h_f = bilby_window_and_fft(
            ts_f.value, cfg.duration, cfg.sampling_frequency
        )
        gwosc_fd[det] = (freqs_g, h_g)
        frame_fd[det] = (freqs_f, h_f)

    # --- Plot --------------------------------------------------------------
    print("Plotting ...")
    plt.rcParams["text.usetex"] = False
    dets = list(cfg.detectors)
    n    = len(dets)
    fig, axes = plt.subplots(
        2, n,
        figsize=(7 * n, 9),
        gridspec_kw={"hspace": 0.35, "wspace": 0.25},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    colors = {"gwosc": "C0", "frame_H1": "C1", "frame_L1": "C3"}

    for col, det in enumerate(dets):
        freqs_g, h_g = gwosc_fd[det]
        freqs_f, h_f = frame_fd[det]
        asd_g = np.abs(h_g)
        asd_f = np.abs(h_f)

        frame_label = (
            f"BayesWave frame\n({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo=det)})"
            if det in frame_strain and frame_strain[det] is not gwosc_strain[det]
            else "GWOSC frame"
        )
        fc = colors.get(f"frame_{det}", "C2")

        # Top: ASD log-log
        ax = axes[0, col]
        ax.loglog(freqs_g, asd_g, lw=0.7, color="C0",
                  label="GWOSC (fetch_open_data)", alpha=0.9)
        ax.loglog(freqs_f, asd_f, lw=0.7, color=fc,
                  label=frame_label, alpha=0.9)
        ax.set_xlim(cfg.minimum_frequency.get(det, 20), cfg.sampling_frequency / 2)
        ax.set_ylabel(r"$|\tilde{h}(f)|$  [strain / Hz]")
        ax.set_title(
            f"{det}  ---  Tukey-windowed single FFT\n"
            f"duration={cfg.duration} s,  $\\alpha$={alpha:.3f}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)

        # Bottom: ratio
        ax = axes[1, col]
        ratio = np.interp(freqs_g, freqs_f, asd_f) / np.where(asd_g > 0, asd_g, np.nan)
        ax.semilogx(freqs_g, ratio, lw=0.6, color=fc)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xlim(cfg.minimum_frequency.get(det, 20), cfg.sampling_frequency / 2)
        ax.set_ylim(0, 3)
        ax.set_ylabel("Ratio  (frame / GWOSC)")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title(f"{det}  frame / GWOSC  |h(f)| ratio")
        ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        f"{EVENT}  |  GPS {cfg.trigger_time:.2f}  |  {label}\n"
        f"Conditioning: Tukey window (roll\\_off={ROLL_OFF} s) + rfft / fs  "
        f"[identical to bilby / gw\\_residuals.py]",
        fontsize=10,
    )

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
