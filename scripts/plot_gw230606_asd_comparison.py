#!/usr/bin/env python3
"""ASD comparison of GWOSC and frame-based strain for GW230606_004305.

Uses the same cached frame files as plot_gw230606_data_comparison.py
(run that script first to download the frames).  Computes ASDs from a
~64-second segment around the merger and plots them in log-log scale.

Outputs: results/gw230606_asd_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT = "GW230606_004305"
GPS   = event_gps(EVENT)      # 1370047403.8

# Segment for ASD estimation — long enough for good frequency resolution
ASD_DT   = 64.0               # seconds either side of merger
FFT_DT   = 4.0                # FFT segment length (sets freq resolution = 0.25 Hz)
OVERLAP  = 2.0                # overlap between FFT segments

FRAME_GPS_START = 1370046464
FRAME_DURATION  = 4096

H1_FRAME_FNAME = f"H-H1_HOFT_C00_BAYESWAVE_S00-{FRAME_GPS_START}-{FRAME_DURATION}.gwf"
L1_FRAME_FNAME = f"L-L1_GWOSC_O4a_4KHZ_R1-{FRAME_GPS_START}-{FRAME_DURATION}.gwf"

H1_CHAN = "H1:GDS-CALIB_STRAIN_CLEAN"
L1_CHAN = "L1:GWOSC-4KHZ_R1_STRAIN"

REPO_DIR  = os.path.join(os.path.dirname(__file__), "..")
FRAME_DIR = os.path.join(REPO_DIR, "data", "frames")
OUTFILE   = os.path.join(REPO_DIR, "results", "gw230606_asd_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    h1_frame_path = os.path.join(FRAME_DIR, H1_FRAME_FNAME)
    l1_frame_path = os.path.join(FRAME_DIR, L1_FRAME_FNAME)

    for p in (h1_frame_path, l1_frame_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{os.path.basename(p)} not found in {FRAME_DIR}. "
                "Run plot_gw230606_data_comparison.py first to download the frames."
            )

    t_start = GPS - ASD_DT
    t_end   = GPS + ASD_DT

    # --- Load data -----------------------------------------------------------
    print("Loading strain data ...")
    h1_gwosc = TimeSeries.fetch_open_data("H1", t_start, t_end, cache=True, verbose=False)
    h1_frame = TimeSeries.read(h1_frame_path, H1_CHAN, start=t_start, end=t_end)
    l1_gwosc = TimeSeries.fetch_open_data("L1", t_start, t_end, cache=True, verbose=False)
    l1_frame = TimeSeries.read(l1_frame_path, L1_CHAN, start=t_start, end=t_end)

    # Resample H1 frame to match GWOSC rate if needed (frame may be 16 kHz)
    if h1_frame.sample_rate.value != h1_gwosc.sample_rate.value:
        print(f"  Resampling H1 frame {h1_frame.sample_rate} → {h1_gwosc.sample_rate}")
        h1_frame = h1_frame.resample(h1_gwosc.sample_rate.value)

    # --- Compute ASDs --------------------------------------------------------
    print("Computing ASDs ...")
    kwargs = dict(fftlength=FFT_DT, overlap=OVERLAP, method="median", window="hann")
    h1_gwosc_asd = h1_gwosc.asd(**kwargs)
    h1_frame_asd = h1_frame.asd(**kwargs)
    l1_gwosc_asd = l1_gwosc.asd(**kwargs)
    l1_frame_asd = l1_frame.asd(**kwargs)

    # Ratio: frame / GWOSC (deviations from 1 indicate differences)
    # Interpolate onto common frequency grid first
    freqs = h1_gwosc_asd.frequencies.value
    h1_frame_interp = np.interp(freqs, h1_frame_asd.frequencies.value, h1_frame_asd.value)
    h1_ratio = h1_frame_interp / h1_gwosc_asd.value

    freqs_l = l1_gwosc_asd.frequencies.value
    l1_frame_interp = np.interp(freqs_l, l1_frame_asd.frequencies.value, l1_frame_asd.value)
    l1_ratio = l1_frame_interp / l1_gwosc_asd.value

    # --- Plot ----------------------------------------------------------------
    print("Plotting ...")
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        gridspec_kw={"hspace": 0.35, "wspace": 0.25},
    )

    df = 1.0 / FFT_DT   # frequency resolution = lowest non-zero frequency bin

    # H1 ASD
    ax = axes[0, 0]
    ax.loglog(h1_gwosc_asd.frequencies.value, h1_gwosc_asd.value,
              lw=0.8, color="C0", label="GWOSC (fetch_open_data)", alpha=0.9)
    ax.loglog(h1_frame_asd.frequencies.value, h1_frame_asd.value,
              lw=0.8, color="C1", label=f"BayesWave frame\n({H1_CHAN})", alpha=0.9)
    ax.set_xlim(df, h1_gwosc.sample_rate.value / 2)
    ax.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
    ax.set_title(f"H1  —  {EVENT}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # H1 ratio
    ax = axes[1, 0]
    ax.semilogx(freqs, h1_ratio, lw=0.7, color="C2")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlim(df, h1_gwosc.sample_rate.value / 2)
    ax.set_ylim(0.5, 2.0)
    ax.set_ylabel("ASD ratio  (frame / GWOSC)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("H1  frame / GWOSC ratio")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # L1 ASD
    ax = axes[0, 1]
    ax.loglog(l1_gwosc_asd.frequencies.value, l1_gwosc_asd.value,
              lw=0.8, color="C0", label="GWOSC (fetch_open_data)", alpha=0.9)
    ax.loglog(l1_frame_asd.frequencies.value, l1_frame_asd.value,
              lw=0.8, color="C3", label=f"GWOSC frame\n({L1_CHAN})", alpha=0.9)
    ax.set_xlim(df, l1_gwosc.sample_rate.value / 2)
    ax.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
    ax.set_title(f"L1  —  {EVENT}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # L1 ratio
    ax = axes[1, 1]
    ax.semilogx(freqs_l, l1_ratio, lw=0.7, color="C4")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlim(df, l1_gwosc.sample_rate.value / 2)
    ax.set_ylim(0.5, 2.0)
    ax.set_ylabel("ASD ratio  (frame / GWOSC)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("L1  frame / GWOSC ratio")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        f"{EVENT}  |  GPS {GPS:.2f}  |  ASD from ±{ASD_DT:.0f} s around merger\n"
        f"FFT length {FFT_DT} s, {OVERLAP} s overlap, Hann window, median averaging",
        fontsize=10,
    )

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
