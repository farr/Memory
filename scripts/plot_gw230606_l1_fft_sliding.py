#!/usr/bin/env python3
"""Sliding 4-second FFT of L1 strain around the GW230606_004305 analysis segment.

Applies the same Tukey-window + rfft/fs conditioning as bilby/gw_residuals.py,
sliding a 4-second window across the L1 data in steps of 3.75 s (overlap
= 0.25 s).  Plots |h(f)| for each segment as a grid of panels so that the
appearance and disappearance of the pronounced low-frequency oscillations can
be timed.

Outputs: results/gw230606_l1_fft_sliding.png
"""

import argparse
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
ROLL_OFF  = 0.2          # bilby default

SEGMENT_DURATION = 4.0   # seconds — match analysis window
OVERLAP          = 0.25  # seconds overlap between consecutive segments
STEP             = SEGMENT_DURATION - OVERLAP   # 3.75 s

SEARCH_PAD = 30.0        # seconds to extend before/after the analysis segment

# Show this frequency range to reveal the sub-90 Hz oscillations clearly.
# Linear x-axis so oscillation periodicity in frequency is easy to read.
FMIN_PLOT = 10.0
FMAX_PLOT = 200.0

REPO_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTFILE  = os.path.join(REPO_DIR, "results", "gw230606_l1_fft_sliding.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bilby_fft(strain_array, duration, sampling_frequency, tukey_alpha=None):
    """Tukey window + rfft/fs, identical to bilby frequency_domain_strain."""
    if tukey_alpha is None:
        tukey_alpha = 2 * ROLL_OFF / duration
    window  = tukey(len(strain_array), alpha=tukey_alpha)
    h_tilde = np.fft.rfft(strain_array * window) / sampling_frequency
    freq    = np.linspace(0, sampling_frequency / 2, len(h_tilde))
    return freq, np.abs(h_tilde)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tukey-alpha", type=float, default=None,
                        help="Tukey window alpha (default: 2*ROLL_OFF/duration = 0.1)")
    args = parser.parse_args()
    tukey_alpha = args.tukey_alpha if args.tukey_alpha is not None else 2 * ROLL_OFF / SEGMENT_DURATION

    alpha_tag = f"_alpha{tukey_alpha:.2f}".replace(".", "p")
    outfile = OUTFILE.replace(".png", f"{alpha_tag}.png") if args.tukey_alpha is not None else OUTFILE
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # --- Parse analysis config from PESummary ---
    print("Reading PESummary config ...")
    data  = pesummary_read(PESUMMARY_H5)
    label = _choose_label(data, PARAM_KEY)
    cfg   = _parse_analysis_config(data, label, EVENT)
    gps   = cfg.trigger_time
    print(f"  analysis segment: [{cfg.start_time:.2f}, {cfg.end_time:.2f}]")

    # --- Load L1 GWOSC data for the full search range ---
    t_load_start = cfg.start_time - SEARCH_PAD
    t_load_end   = cfg.end_time   + SEARCH_PAD + 3 * STEP   # +3 panels to fill last row
    print(f"Loading L1 GWOSC [{t_load_start:.1f}, {t_load_end:.1f}] "
          f"({t_load_end - t_load_start:.0f} s) ...")
    ts = _download_gwosc_strain(
        ["L1"], t_load_start, t_load_end, fs=4096.0, frame_dir=None,
    )["L1"]
    fs = ts.sample_rate.value
    n_seg = int(round(SEGMENT_DURATION * fs))

    # --- Build list of segment start times ---
    t0_arr = np.arange(
        t_load_start,
        t_load_end - SEGMENT_DURATION + 1e-6,
        STEP,
    )
    print(f"  {len(t0_arr)} segments of {SEGMENT_DURATION} s, "
          f"step={STEP} s, overlap={OVERLAP} s")

    # --- Compute FFTs ---
    segments = []
    for t_start in t0_arr:
        t_end = t_start + SEGMENT_DURATION
        seg   = ts.crop(t_start, t_end)
        if len(seg) < n_seg * 0.95:
            continue
        freq, asd = bilby_fft(seg.value[:n_seg], SEGMENT_DURATION, fs, tukey_alpha)
        segments.append({
            "t_start": t_start,
            "t_end":   t_end,
            "freq":    freq,
            "asd":     asd,
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

    # Compute a common y range from the data in the plot frequency band
    all_asd_inband = []
    for s in segments:
        mask = (s["freq"] >= FMIN_PLOT) & (s["freq"] <= FMAX_PLOT)
        all_asd_inband.append(s["asd"][mask])
    all_asd_cat = np.concatenate(all_asd_inband)
    ymin = np.percentile(all_asd_cat, 1)  * 0.3
    ymax = np.percentile(all_asd_cat, 99) * 3.0

    for i, s in enumerate(segments):
        ax = axes_flat[i]
        color = "C1" if s["is_analysis"] else "C0"
        lw    = 1.0  if s["is_analysis"] else 0.7
        mask  = (s["freq"] >= FMIN_PLOT) & (s["freq"] <= FMAX_PLOT)
        ax.semilogy(s["freq"][mask], s["asd"][mask], lw=lw, color=color)
        ax.axvline(90, color="k", lw=0.6, ls=":", alpha=0.5)   # 90 Hz marker
        ax.set_xlim(FMIN_PLOT, FMAX_PLOT)
        ax.set_ylim(ymin, ymax)
        dt_start = s["t_start"] - gps
        dt_end   = s["t_end"]   - gps
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
        f"{EVENT}  —  L1  |  sliding {SEGMENT_DURATION:.0f}-s Tukey FFT (alpha={tukey_alpha:.2f}), "
        f"step={STEP} s (overlap={OVERLAP} s)\n"
        f"search: {SEARCH_PAD:.0f} s before/after analysis segment  "
        f"(*  = analysis segment, orange)  |  dashed line at 90 Hz",
        fontsize=9,
    )

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    main()
