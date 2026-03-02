#!/usr/bin/env python3
"""Sliding 4-second FFT of L1 strain from the GWOSC 16 KHz frame.

Same conditioning and layout as plot_gw230606_l1_fft_sliding.py (which uses
fetch_open_data at 4096 Hz) but reads from the GWOSC 16 KHz frame
(L-L1_GWOSC_O4a_16KHZ_R1-...) and resamples to 4096 Hz before the FFT.
Sanity check: if the low-frequency oscillations are present here too they are
in the raw detector data, not an artefact of the 4 KHz download pipeline.

Outputs: results/gw230606_l1_16khz_fft_sliding.png
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from gwpy.timeseries import TimeSeries

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory.gw_residuals import (
    _parse_analysis_config,
    _choose_label,
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
ROLL_OFF  = 0.2

SEGMENT_DURATION = 4.0
OVERLAP          = 0.25
STEP             = SEGMENT_DURATION - OVERLAP   # 3.75 s
SEARCH_PAD       = 30.0

FMIN_PLOT = 10.0
FMAX_PLOT = 200.0

FRAME_GPS_START = 1370046464
FRAME_DURATION  = 4096
L1_16KHZ_FNAME  = f"L-L1_GWOSC_O4a_16KHZ_R1-{FRAME_GPS_START}-{FRAME_DURATION}.gwf"
L1_16KHZ_URL    = (
    f"https://gwosc.org/archive/data/O4a_16KHZ_R1/1369440256/{L1_16KHZ_FNAME}"
)
L1_16KHZ_CHAN   = "L1:GWOSC-16KHZ_R1_STRAIN"

REPO_DIR  = os.path.join(os.path.dirname(__file__), "..")
FRAME_DIR = os.path.join(REPO_DIR, "data", "frames")
OUTFILE   = os.path.join(REPO_DIR, "results", "gw230606_l1_16khz_fft_sliding.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_frame(fname, url, cache_dir):
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        print(f"  Using cached {fname}")
        return path
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Downloading {fname} ...")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--fail", "-o", path, url],
        check=True,
    )
    return path


def bilby_fft(strain_array, duration, sampling_frequency):
    alpha   = 2 * ROLL_OFF / duration
    window  = tukey(len(strain_array), alpha=alpha)
    h_tilde = np.fft.rfft(strain_array * window) / sampling_frequency
    freq    = np.linspace(0, sampling_frequency / 2, len(h_tilde))
    return freq, np.abs(h_tilde)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    print("Reading PESummary config ...")
    data  = pesummary_read(PESUMMARY_H5)
    label = _choose_label(data, PARAM_KEY)
    cfg   = _parse_analysis_config(data, label, EVENT)
    gps   = cfg.trigger_time
    print(f"  analysis segment: [{cfg.start_time:.2f}, {cfg.end_time:.2f}]")

    t_load_start = cfg.start_time - SEARCH_PAD
    t_load_end   = cfg.end_time   + SEARCH_PAD + 3 * STEP   # same range as L1 script

    # --- Download 16 KHz frame ---
    print("Downloading 16 KHz L1 frame ...")
    frame_path = download_frame(L1_16KHZ_FNAME, L1_16KHZ_URL, FRAME_DIR)

    # --- Read strain from frame ---
    print(f"Reading {L1_16KHZ_CHAN} [{t_load_start:.1f}, {t_load_end:.1f}] ...")
    ts = TimeSeries.read(frame_path, L1_16KHZ_CHAN,
                         start=t_load_start, end=t_load_end)
    print(f"  Sample rate: {ts.sample_rate}")

    # Resample to 4096 Hz to match the L1/H1 scripts
    if ts.sample_rate.value != 4096.0:
        print(f"  Resampling {ts.sample_rate} -> 4096 Hz ...")
        ts = ts.resample(4096.0)

    fs    = ts.sample_rate.value
    n_seg = int(round(SEGMENT_DURATION * fs))

    # --- Sliding windows ---
    t0_arr = np.arange(t_load_start, t_load_end - SEGMENT_DURATION + 1e-6, STEP)
    print(f"  {len(t0_arr)} segments of {SEGMENT_DURATION} s, "
          f"step={STEP} s, overlap={OVERLAP} s")

    segments = []
    for t_start in t0_arr:
        t_end = t_start + SEGMENT_DURATION
        seg   = ts.crop(t_start, t_end)
        if len(seg) < n_seg * 0.95:
            continue
        freq, asd = bilby_fft(seg.value[:n_seg], SEGMENT_DURATION, fs)
        segments.append({
            "t_start":     t_start,
            "t_end":       t_end,
            "freq":        freq,
            "asd":         asd,
            "is_analysis": abs(t_start - cfg.start_time) < 0.05,
        })

    # --- Layout ---
    ncols = 4
    nrows = int(np.ceil(len(segments) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.2 * nrows),
        gridspec_kw={"hspace": 0.55, "wspace": 0.32},
    )
    plt.rcParams["text.usetex"] = False
    axes_flat = axes.flatten()

    all_asd_inband = []
    for s in segments:
        mask = (s["freq"] >= FMIN_PLOT) & (s["freq"] <= FMAX_PLOT)
        all_asd_inband.append(s["asd"][mask])
    all_asd_cat = np.concatenate(all_asd_inband)
    ymin = np.percentile(all_asd_cat, 1)  * 0.3
    ymax = np.percentile(all_asd_cat, 99) * 3.0

    for i, s in enumerate(segments):
        ax    = axes_flat[i]
        color = "C1" if s["is_analysis"] else "C0"
        lw    = 1.0  if s["is_analysis"] else 0.7
        mask  = (s["freq"] >= FMIN_PLOT) & (s["freq"] <= FMAX_PLOT)
        ax.semilogy(s["freq"][mask], s["asd"][mask], lw=lw, color=color)
        ax.axvline(90, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.set_xlim(FMIN_PLOT, FMAX_PLOT)
        ax.set_ylim(ymin, ymax)
        dt_start  = s["t_start"] - gps
        dt_end    = s["t_end"]   - gps
        label_str = f"[{dt_start:+.2f}, {dt_end:+.2f}] s"
        if s["is_analysis"]:
            label_str += "  *"
        ax.set_title(label_str, fontsize=7,
                     color="C1" if s["is_analysis"] else "k")
        ax.tick_params(labelsize=6)
        if i % ncols == 0:
            ax.set_ylabel(r"$|h(f)|$", fontsize=7)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Frequency [Hz]", fontsize=7)

    for j in range(len(segments), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{EVENT}  —  L1 GWOSC 16 KHz frame ({L1_16KHZ_CHAN}, resampled to 4096 Hz)\n"
        f"sliding {SEGMENT_DURATION:.0f}-s Tukey FFT, step={STEP} s (overlap={OVERLAP} s)  "
        f"|  (* = analysis segment, orange)  |  dashed line at 90 Hz",
        fontsize=9,
    )

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
