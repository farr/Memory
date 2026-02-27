#!/usr/bin/env python3
"""Time-domain comparison of GWOSC and frame-based strain for GW230606_004305.

Downloads the H1 BayesWave glitch-subtracted frame (Zenodo 16857060) and the
L1 GWOSC GWF frame, then plots raw and bandpassed strain from both sources
alongside the data obtained via fetch_open_data.  The H1 difference panel
reveals the glitch model removed by BayesWave.

Outputs: results/gw230606_data_comparison.png
"""

import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT = "GW230606_004305"
GPS = event_gps(EVENT)          # 1370047403.8

# Wider load window for PSD estimation; plot window is tighter
LOAD_DT = 64.0                  # seconds either side of merger to load
PLOT_DT = 2.0                   # seconds either side of merger to display
BP_LOW, BP_HIGH = 20, 500       # bandpass range [Hz]

FRAME_GPS_START = 1370046464
FRAME_DURATION  = 4096

H1_FRAME_FNAME = f"H-H1_HOFT_C00_BAYESWAVE_S00-{FRAME_GPS_START}-{FRAME_DURATION}.gwf"
H1_FRAME_URL   = f"https://zenodo.org/records/16857060/files/{H1_FRAME_FNAME}?download=1"

L1_FRAME_FNAME = f"L-L1_GWOSC_O4a_4KHZ_R1-{FRAME_GPS_START}-{FRAME_DURATION}.gwf"
L1_FRAME_URL   = (
    f"https://gwosc.org/archive/data/O4a_4KHZ_R1/1369440256/{L1_FRAME_FNAME}"
)

REPO_DIR   = os.path.join(os.path.dirname(__file__), "..")
FRAME_DIR  = os.path.join(REPO_DIR, "data", "frames")
OUTFILE    = os.path.join(REPO_DIR, "results", "gw230606_data_comparison.png")

# Default channel name guesses (tried in order; first match wins)
H1_CHANNEL_CANDIDATES = [
    "H1:GDS-CALIB_STRAIN_CLEAN",           # confirmed in Zenodo 16857060 frames
    "H1:GDS-CALIB_STRAIN_CLEAN_BAYESWAVE_S00",
    "H1:DCS-CALIB_STRAIN_CLEAN_C00",
    "H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C00",
]
L1_CHANNEL_CANDIDATES = [
    "L1:GWOSC-4KHZ_R1_STRAIN",
    "L1:GWOSC-16KHZ_R1_STRAIN",
    "L1:DCS-CALIB_STRAIN_CLEAN_C00",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_frame(fname, url, cache_dir):
    """Download *fname* from *url* into *cache_dir* if not already present."""
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        print(f"  Using cached {fname}")
        return path
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Downloading {fname} (~1 GB, this may take a while) ...")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--fail", "-o", path, url],
        check=True,
    )
    return path


def list_channels(path):
    """Return channel names in *path*, or [] if the frame library is unavailable."""
    try:
        from gwpy.io.gwf import iter_channel_names
        return list(iter_channel_names(path))
    except Exception as exc:
        print(f"  Warning: could not list channels in {os.path.basename(path)}: {exc}")
        return []


def pick_channel(channels, candidates, ifo):
    """Return the first candidate present in *channels*, or the first strain channel."""
    for c in candidates:
        if c in channels:
            return c
    # fall back: first channel containing IFO name and 'STRAIN'
    for ch in channels:
        if ifo in ch and "STRAIN" in ch.upper():
            return ch
    if channels:
        print(f"  Warning: none of {candidates} found; using first channel: {channels[0]}")
        return channels[0]
    return candidates[0]  # last resort: try the first candidate blindly


def load_ts(source, channel, start, end, label):
    """Load a TimeSeries from *source* (path or 'gwosc') and crop to [start, end]."""
    if source == "gwosc":
        print(f"  fetch_open_data {channel[:2]} [{start:.0f}, {end:.0f})")
        ts = TimeSeries.fetch_open_data(
            channel[:2], start - LOAD_DT, end + LOAD_DT, cache=True, verbose=False
        )
    else:
        print(f"  read frame {os.path.basename(source)}  channel={channel}")
        ts = TimeSeries.read(source, channel, start=start - LOAD_DT, end=end + LOAD_DT)
    return ts.crop(start, end)


def bandpass(ts, flow=BP_LOW, fhigh=BP_HIGH):
    return ts.bandpass(flow, fhigh)


def plot_row(ax, ts, t0, color, label, lw=0.6, alpha=1.0):
    t = ts.times.value - t0
    ax.plot(t, ts.value, lw=lw, color=color, label=label, alpha=alpha)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    t_start = GPS - PLOT_DT
    t_end   = GPS + PLOT_DT

    # --- Download frames ---------------------------------------------------
    print("Downloading frame files ...")
    h1_frame_path = download_frame(H1_FRAME_FNAME, H1_FRAME_URL, FRAME_DIR)
    l1_frame_path = download_frame(L1_FRAME_FNAME, L1_FRAME_URL, FRAME_DIR)

    # --- Discover channels -------------------------------------------------
    print("Inspecting channel names ...")
    h1_channels = list_channels(h1_frame_path)
    l1_channels = list_channels(l1_frame_path)
    if h1_channels:
        print(f"  H1 frame channels: {h1_channels}")
    if l1_channels:
        print(f"  L1 frame channels: {l1_channels}")

    h1_chan = pick_channel(h1_channels, H1_CHANNEL_CANDIDATES, "H1")
    l1_chan = pick_channel(l1_channels, L1_CHANNEL_CANDIDATES, "L1")
    print(f"  Using H1 channel: {h1_chan}")
    print(f"  Using L1 channel: {l1_chan}")

    # --- Load data ---------------------------------------------------------
    print("Loading strain data ...")
    h1_gwosc = load_ts("gwosc",       "H1",    t_start, t_end, "H1 GWOSC")
    h1_frame = load_ts(h1_frame_path, h1_chan, t_start, t_end, "H1 frame")
    l1_gwosc = load_ts("gwosc",       "L1",    t_start, t_end, "L1 GWOSC")
    l1_frame = load_ts(l1_frame_path, l1_chan, t_start, t_end, "L1 frame")

    # --- Resample to common rate for H1 difference -------------------------
    # BayesWave frame may be at 16384 Hz; GWOSC open data at 4096 Hz.
    h1_gwosc_r = h1_gwosc.resample(h1_frame.sample_rate.value)

    # --- Bandpass ----------------------------------------------------------
    print("Bandpassing ...")
    h1_gwosc_bp = bandpass(h1_gwosc)
    h1_frame_bp = bandpass(h1_frame)
    l1_gwosc_bp = bandpass(l1_gwosc)
    l1_frame_bp = bandpass(l1_frame)

    # H1 difference (frame − GWOSC): reveals the BayesWave glitch model
    h1_diff    = h1_frame    - h1_gwosc_r
    h1_diff_bp = bandpass(h1_diff)

    # --- Plot --------------------------------------------------------------
    print("Plotting ...")
    fig, axes = plt.subplots(
        5, 1,
        figsize=(13, 14),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    t0 = GPS

    # Row 0 — H1 raw
    plot_row(axes[0], h1_gwosc,    t0, "C0", "GWOSC (fetch_open_data)")
    plot_row(axes[0], h1_frame,    t0, "C1", f"BayesWave frame ({h1_chan})", alpha=0.85)
    axes[0].set_ylabel("H1 raw strain")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(
        f"{EVENT}  |  GPS {GPS:.2f}  |  t=0 at merger\n"
        f"H1: GWOSC vs BayesWave-cleaned frame;  L1: GWOSC vs GWOSC frame"
    )

    # Row 1 — H1 bandpassed
    plot_row(axes[1], h1_gwosc_bp, t0, "C0", f"GWOSC  ({BP_LOW}–{BP_HIGH} Hz)")
    plot_row(axes[1], h1_frame_bp, t0, "C1", f"BayesWave frame", alpha=0.85)
    axes[1].set_ylabel(f"H1 bandpassed")
    axes[1].legend(fontsize=7, loc="upper right")

    # Row 2 — H1 difference (BayesWave glitch model)
    plot_row(axes[2], h1_diff,    t0, "C2", "raw",      lw=0.5, alpha=0.6)
    plot_row(axes[2], h1_diff_bp, t0, "C3", f"{BP_LOW}–{BP_HIGH} Hz")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel("H1  frame − GWOSC\n(BayesWave glitch model)")
    axes[2].legend(fontsize=7, loc="upper right")

    # Row 3 — L1 raw
    plot_row(axes[3], l1_gwosc, t0, "C0", "GWOSC (fetch_open_data)")
    plot_row(axes[3], l1_frame, t0, "C4", f"GWOSC frame ({l1_chan})", alpha=0.85)
    axes[3].set_ylabel("L1 raw strain")
    axes[3].legend(fontsize=7, loc="upper right")

    # Row 4 — L1 bandpassed
    plot_row(axes[4], l1_gwosc_bp, t0, "C0", f"GWOSC  ({BP_LOW}–{BP_HIGH} Hz)")
    plot_row(axes[4], l1_frame_bp, t0, "C4", "GWOSC frame", alpha=0.85)
    axes[4].set_ylabel(f"L1 bandpassed")
    axes[4].legend(fontsize=7, loc="upper right")
    axes[4].set_xlabel(f"Time from GPS {GPS:.2f} [s]")

    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.4)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
