#!/usr/bin/env python3
"""Time-domain and spectrogram comparison for GW230606_004305.

Uses the same segment and conditioning as plot_gw230606_fd_comparison.py:
  - segment duration and boundaries from the PESummary analysis config
  - Tukey window (roll_off=0.2 s) shown on top of the raw strain

Plot (1): time-domain — same 4-second analysis segment, GWOSC vs frame,
          with the Tukey window overlaid and a difference panel.
Plot (2): Q-transform spectrogram computed from a longer (128 s) segment
          so the PSD estimate is stable, zoomed to ±2 s around the merger.

Requires: frames already downloaded by plot_gw230606_data_comparison.py.

Outputs:
  results/gw230606_td_comparison.png
  results/gw230606_spectrogram.png
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
# Paths and constants
# ---------------------------------------------------------------------------

EVENT        = "GW230606_004305"
PESUMMARY_H5 = (
    "/mnt/home/ccalvk/ceph/GWTC-4/"
    "IGWN-GWTC4p0-1a206db3d_721-GW230606_004305-combined_PEDataRelease.hdf5"
)
PARAM_KEY = "C00:NRSur7dq4"
ROLL_OFF  = 0.2   # bilby default

# Extra data loaded around the analysis segment for the Q-transform PSD estimate
QTRANSFORM_PAD = 64.0    # seconds of padding either side
QPLOT_DT       = 2.0     # seconds either side of merger shown in spectrogram
QRANGE         = (4, 64)
FRANGE         = (20, 1000)

REPO_DIR  = os.path.join(os.path.dirname(__file__), "..")
FRAME_DIR = os.path.join(REPO_DIR, "data", "frames")
OUTFILE_TD      = os.path.join(REPO_DIR, "results", "gw230606_td_comparison.png")
OUTFILE_SPEC    = os.path.join(REPO_DIR, "results", "gw230606_spectrogram.png")
OUTFILE_SPEC_8S     = os.path.join(REPO_DIR, "results", "gw230606_spectrogram_8s.png")
OUTFILE_SPEC_8S_10HZ = os.path.join(REPO_DIR, "results", "gw230606_spectrogram_8s_10hz.png")
OUTFILE_PSD     = os.path.join(REPO_DIR, "results", "gw230606_spectrogram_psd.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_strain(det, start, end, frame_dir):
    """Load GWOSC and frame strain for *det* over [start, end]."""
    gwosc = _download_gwosc_strain(
        [det], start, end, fs=4096.0, frame_dir=None,
    )[det]
    gwf = _find_frame_file(frame_dir, det, start, end)
    if gwf is not None:
        frame = _read_frame_strain(
            gwf, det, start, end,
            channel_format=GLITCH_SUBTRACTED_CHANNEL_FORMAT,
        )
        if frame.sample_rate.value != gwosc.sample_rate.value:
            frame = frame.resample(gwosc.sample_rate.value)
    else:
        frame = gwosc
    return gwosc, frame, gwf is not None


def make_tukey_window(n, duration, roll_off=ROLL_OFF):
    return tukey(n, alpha=2 * roll_off / duration)


# ---------------------------------------------------------------------------
# Plot 1 — time domain
# ---------------------------------------------------------------------------

def plot_time_domain(cfg, gps, gwosc_h1, frame_h1, gwosc_l1, frame_l1,
                     has_l1_frame):
    plt.rcParams["text.usetex"] = False
    t0 = gps

    # Tukey window for the analysis segment
    n      = len(gwosc_h1)
    window = make_tukey_window(n, cfg.duration)
    times_h1 = gwosc_h1.times.value - t0
    times_l1 = gwosc_l1.times.value - t0

    # Differences
    h1_diff = frame_h1.value - gwosc_h1.value
    l1_diff = frame_l1.value - gwosc_l1.value if has_l1_frame else None

    nrows = 5 if not has_l1_frame else 6
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(12, 3 * nrows),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # Row 0 — H1 raw strain + window outline
    ax = axes[0]
    ax.plot(times_h1, gwosc_h1.value, lw=0.7, color="C0",
            label="H1  GWOSC (fetch_open_data)")
    ax.plot(times_h1, frame_h1.value, lw=0.7, color="C1", alpha=0.85,
            label=f"H1  BayesWave frame ({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo='H1')})")
    ymax_h1 = np.percentile(np.abs(gwosc_h1.value), 99) * 1.5
    ax.plot(times_h1, window * ymax_h1, lw=1.0, color="k", ls=":", alpha=0.5,
            label=f"Tukey window  (roll_off={ROLL_OFF} s, scaled)")
    ax.plot(times_h1, -window * ymax_h1, lw=1.0, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("H1 strain")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(
        f"{EVENT}  |  GPS {gps:.2f}  |  analysis segment "
        f"[{cfg.start_time:.1f}, {cfg.end_time:.1f}]  ({cfg.duration} s)"
    )

    # Row 1 — H1 windowed strain
    ax = axes[1]
    ax.plot(times_h1, gwosc_h1.value * window, lw=0.7, color="C0",
            label="H1  GWOSC x window")
    ax.plot(times_h1, frame_h1.value * window, lw=0.7, color="C1", alpha=0.85,
            label="H1  frame x window")
    ax.set_ylabel("H1 windowed strain")
    ax.legend(fontsize=7, loc="upper right")

    # Row 2 — H1 difference (frame - GWOSC) = BayesWave glitch model
    ax = axes[2]
    ax.plot(times_h1, h1_diff, lw=0.7, color="C2",
            label="H1  frame - GWOSC  (BayesWave glitch model)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("H1 difference")
    ax.legend(fontsize=7, loc="upper right")

    # Row 3 — L1 raw strain + window outline
    ax = axes[3]
    ax.plot(times_l1, gwosc_l1.value, lw=0.7, color="C0",
            label="L1  GWOSC (fetch_open_data)")
    if has_l1_frame:
        ax.plot(times_l1, frame_l1.value, lw=0.7, color="C3", alpha=0.85,
                label=f"L1  BayesWave frame ({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo='L1')})")
    ymax_l1 = np.percentile(np.abs(gwosc_l1.value), 99) * 1.5
    ax.plot(times_l1, window * ymax_l1, lw=1.0, color="k", ls=":", alpha=0.5,
            label=f"Tukey window  (roll_off={ROLL_OFF} s, scaled)")
    ax.plot(times_l1, -window * ymax_l1, lw=1.0, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("L1 strain")
    ax.legend(fontsize=7, loc="upper right")

    # Row 4 — L1 windowed strain
    ax = axes[4]
    ax.plot(times_l1, gwosc_l1.value * window, lw=0.7, color="C0",
            label="L1  GWOSC x window")
    if has_l1_frame:
        ax.plot(times_l1, frame_l1.value * window, lw=0.7, color="C3", alpha=0.85,
                label="L1  frame x window")
    ax.set_ylabel("L1 windowed strain")
    ax.legend(fontsize=7, loc="upper right")

    if has_l1_frame:
        # Row 5 — L1 difference
        ax = axes[5]
        ax.plot(times_l1, l1_diff, lw=0.7, color="C4",
                label="L1  frame - GWOSC  (BayesWave glitch model)")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel("L1 difference")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel(f"Time from GPS {gps:.2f} [s]")

    for a in axes:
        a.axvline(0, color="k", lw=0.8, ls=":", alpha=0.4)

    fig.savefig(OUTFILE_TD, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE_TD}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Q-transform spectrogram
# ---------------------------------------------------------------------------

def plot_spectrogram(gps, gwosc_h1_long, frame_h1_long, gwosc_l1_long,
                     has_h1_frame):
    plt.rcParams["text.usetex"] = False
    outseg = (gps - QPLOT_DT, gps + QPLOT_DT)
    qt_kwargs = dict(
        qrange=QRANGE, frange=FRANGE,
        outseg=outseg, whiten=True,
    )

    print("  Q-transform H1 GWOSC ...")
    qt_h1_gwosc = gwosc_h1_long.q_transform(**qt_kwargs)
    print("  Q-transform H1 frame ...")
    qt_h1_frame = frame_h1_long.q_transform(**qt_kwargs)
    print("  Q-transform L1 GWOSC ...")
    qt_l1_gwosc = gwosc_l1_long.q_transform(**qt_kwargs)

    nrows = 3 if has_h1_frame else 2
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(12, 4 * nrows),
        gridspec_kw={"hspace": 0.35},
    )

    def _qtplot(ax, qt, title, norm=None):
        pcm = ax.pcolormesh(
            qt.times.value - gps,
            qt.frequencies.value,
            qt.value.T,
            norm=norm,
            cmap="viridis",
            shading="auto",
        )
        ax.set_yscale("log")
        ax.set_ylim(*FRANGE)
        ax.set_ylabel("Frequency [Hz]")
        ax.axvline(0, color="w", lw=0.8, ls=":")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="Normalised energy")

    from matplotlib.colors import Normalize
    vmax = max(qt_h1_gwosc.value.max(), qt_h1_frame.value.max())
    norm = Normalize(vmin=0, vmax=vmax)

    _qtplot(axes[0], qt_h1_gwosc,
            "H1  GWOSC (fetch_open_data)  —  whitened Q-transform",
            norm=norm)
    _qtplot(axes[1], qt_h1_frame,
            f"H1  BayesWave frame ({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo='H1')})"
            "  —  whitened Q-transform",
            norm=norm)
    if nrows == 3:
        _qtplot(axes[2], qt_l1_gwosc,
                "L1  GWOSC  —  whitened Q-transform")

    for ax in axes:
        ax.set_xlabel(f"Time from GPS {gps:.2f} [s]")

    fig.suptitle(
        f"{EVENT}  |  GPS {gps:.2f}  |  Q-transform  "
        f"Q in {QRANGE},  f in {FRANGE} Hz,  +/-{QPLOT_DT} s",
        fontsize=10,
    )

    fig.savefig(OUTFILE_SPEC, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE_SPEC}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — 8-second spectrogram, H1 cleaned + L1 only
# ---------------------------------------------------------------------------

def plot_spectrogram_8s(cfg, gps, frame_h1_long, gwosc_l1_long,
                        frange=FRANGE, outfile=OUTFILE_SPEC_8S, highpass=None):
    plt.rcParams["text.usetex"] = False
    # 8-second window centred on the analysis segment (2 s extra each side)
    outseg = (cfg.start_time - 2.0, cfg.end_time + 2.0)
    qt_kwargs = dict(
        qrange=QRANGE, frange=frange,
        outseg=outseg, whiten=True, highpass=highpass,
    )

    print("  Q-transform H1 BayesWave (8 s) ...")
    qt_h1 = frame_h1_long.q_transform(**qt_kwargs)
    print("  Q-transform L1 GWOSC (8 s) ...")
    qt_l1 = gwosc_l1_long.q_transform(**qt_kwargs)

    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={"hspace": 0.35},
    )

    def _qtplot(ax, qt, title):
        norm = Normalize(vmin=0, vmax=qt.value.max())
        pcm = ax.pcolormesh(
            qt.times.value - gps,
            qt.frequencies.value,
            qt.value.T,
            norm=norm,
            cmap="viridis",
            shading="auto",
        )
        ax.set_yscale("log")
        ax.set_ylim(*frange)
        ax.set_ylabel("Frequency [Hz]")
        ax.axvline(0, color="w", lw=0.8, ls=":")
        # Mark the 4-second analysis segment boundaries
        ax.axvline(cfg.start_time - gps, color="w", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(cfg.end_time   - gps, color="w", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="Normalised energy")

    _qtplot(axes[0], qt_h1,
            f"H1  BayesWave ({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo='H1')})"
            "  —  whitened Q-transform")
    _qtplot(axes[1], qt_l1,
            "L1  GWOSC  —  whitened Q-transform")

    for ax in axes:
        ax.set_xlabel(f"Time from GPS {gps:.2f} [s]")

    fig.suptitle(
        f"{EVENT}  |  GPS {gps:.2f}  |  8-second window  "
        f"[{outseg[0] - gps:+.0f}, {outseg[1] - gps:+.0f}] s\n"
        f"Q in {QRANGE},  f in {frange} Hz  "
        f"|  dashed lines = 4-second analysis segment",
        fontsize=10,
    )

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — PSD used for whitening in the spectrograms
# ---------------------------------------------------------------------------

def plot_whitening_psd(gps, frame_h1_long, gwosc_l1_long):
    """Reproduce exactly the ASD that q_transform(whiten=True) computes.

    gwpy calls ts.asd(fftlength=2, overlap=1, window='hann', method='median')
    on the full 128-second input TimeSeries.
    """
    plt.rcParams["text.usetex"] = False

    # Exact parameters from gwpy q_transform source:
    #   fftlength = _fft_length_default(dt) = 2 s  (for 4096 Hz)
    #   overlap   = recommended_overlap('hann') * fftlength = 0.5 * 2 = 1 s
    #   window    = 'hann'
    #   method    = 'median'
    asd_kwargs = dict(fftlength=2, overlap=1, window="hann", method="median")

    print("  Computing H1 ASD ...")
    asd_h1 = frame_h1_long.asd(**asd_kwargs)
    print("  Computing L1 ASD ...")
    asd_l1 = gwosc_l1_long.asd(**asd_kwargs)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.loglog(asd_h1.frequencies.value, asd_h1.value,
              lw=0.8, color="C1",
              label=f"H1  BayesWave ({GLITCH_SUBTRACTED_CHANNEL_FORMAT.format(ifo='H1')})")
    ax.loglog(asd_l1.frequencies.value, asd_l1.value,
              lw=0.8, color="C0",
              label="L1  GWOSC (fetch_open_data)")

    ax.set_xlim(*FRANGE)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"ASD  [strain / $\sqrt{\mathrm{Hz}}$]")
    ax.set_title(
        f"{EVENT}  |  GPS {gps:.2f}  |  Whitening ASD used by q_transform\n"
        f"Estimated from {2*QTRANSFORM_PAD:.0f}-second segment  "
        f"(Hann window, fftlength=2 s, overlap=1 s, median)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.savefig(OUTFILE_PSD, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE_PSD}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTFILE_TD), exist_ok=True)

    # --- Parse analysis config ---------------------------------------------
    print("Reading PESummary config ...")
    data  = pesummary_read(PESUMMARY_H5)
    label = _choose_label(data, PARAM_KEY)
    cfg   = _parse_analysis_config(data, label, EVENT)
    gps   = cfg.trigger_time

    print(f"  segment: [{cfg.start_time:.1f}, {cfg.end_time:.1f}]  ({cfg.duration} s)")

    # --- Load analysis-window data (for time-domain plot) ------------------
    print("Loading 4-second analysis segment ...")
    gwosc_h1, frame_h1, has_h1_frame = load_strain(
        "H1", cfg.start_time, cfg.end_time, FRAME_DIR,
    )
    gwosc_l1, frame_l1, has_l1_frame = load_strain(
        "L1", cfg.start_time, cfg.end_time, FRAME_DIR,
    )

    # --- Load longer segment (for Q-transform PSD estimation) --------------
    t_long_start = gps - QTRANSFORM_PAD
    t_long_end   = gps + QTRANSFORM_PAD
    print(f"Loading {2*QTRANSFORM_PAD:.0f}-second segment for Q-transform ...")
    gwosc_h1_long, frame_h1_long, _ = load_strain(
        "H1", t_long_start, t_long_end, FRAME_DIR,
    )
    gwosc_l1_long, _, _ = load_strain(
        "L1", t_long_start, t_long_end, FRAME_DIR,
    )

    # --- Plot 1: time domain -----------------------------------------------
    print("Plotting time domain ...")
    plot_time_domain(cfg, gps, gwosc_h1, frame_h1, gwosc_l1, frame_l1,
                     has_l1_frame)

    # --- Plot 2: spectrograms ----------------------------------------------
    print("Computing Q-transforms ...")
    plot_spectrogram(gps, gwosc_h1_long, frame_h1_long, gwosc_l1_long,
                     has_h1_frame)

    # --- Plot 3: 8-second spectrogram, H1 cleaned + L1 --------------------
    print("Computing 8-second Q-transforms ...")
    plot_spectrogram_8s(cfg, gps, frame_h1_long, gwosc_l1_long)

    # --- Plot 3b: same but starting at 10 Hz ------------------------------
    print("Computing 8-second Q-transforms (10 Hz floor) ...")
    plot_spectrogram_8s(cfg, gps, frame_h1_long, gwosc_l1_long,
                        frange=(10, 1000), outfile=OUTFILE_SPEC_8S_10HZ,
                        highpass=10)

    # --- Plot 4: PSD used for whitening ------------------------------------
    print("Plotting whitening PSD ...")
    plot_whitening_psd(gps, frame_h1_long, gwosc_l1_long)


if __name__ == "__main__":
    main()
