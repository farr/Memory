#!/usr/bin/env python3
"""Diagnostic plot of A_hat and A_sigma distributions across events.

For each event in the memory directory, shows whisker plots (50% + 90% CI) of:
  - A_hat  (ML memory amplitude)
  - A_sigma  (amplitude uncertainty)
  - |A_hat / A_sigma|  (unsigned per-sample memory SNR)
  - A_hat / A_sigma    (signed per-sample memory SNR)

Usage:
    uv run python scripts/plot_memory_diagnostics.py \\
        --memory-dir /path/to/memory_dir \\
        --out results/memory_diagnostics.png
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory.hierarchical.data import load_memory_data

POSTERIOR_GLOB = "/mnt/home/ccalvk/ceph/GWTC-4/IGWN-GWTC4p0-*-combined_PEDataRelease.hdf5"
WAVEFORM_LABEL = "C00:NRSur7dq4"


def _filter_to_memory_events(posterior_files, memory_dir):
    """Keep only posterior files whose event has a memory_results.h5 in memory_dir."""
    import re
    kept = []
    for f in posterior_files:
        m = re.search(r"(GW\d{6}_\d{6})", os.path.basename(f))
        if m and os.path.exists(os.path.join(memory_dir, m.group(1), "memory_results.h5")):
            kept.append(f)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory-dir",
        default="/mnt/home/kmitman/work/memory_pop/analysis",
        help="Directory with per-event memory_results.h5 files",
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    memory_dir = args.memory_dir
    outfile = args.out or os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"memory_diagnostics_{os.path.basename(memory_dir.rstrip('/'))}.png",
    )
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)

    posterior_files = sorted(glob.glob(POSTERIOR_GLOB))
    kept_files = _filter_to_memory_events(posterior_files, memory_dir)
    print(f"Found {len(kept_files)} events with memory results in {memory_dir}")
    memory_data = load_memory_data(kept_files, memory_dir, WAVEFORM_LABEL)

    names = [md["event_name"] for md in memory_data]
    n = len(names)
    x = np.arange(n)

    # Compute percentiles for each event
    p05_ah, p25_ah, p50_ah, p75_ah, p95_ah = [], [], [], [], []
    p05_as, p25_as, p50_as, p75_as, p95_as = [], [], [], [], []
    p05_snr, p25_snr, p50_snr, p75_snr, p95_snr = [], [], [], [], []
    p05_ssnr, p25_ssnr, p50_ssnr, p75_ssnr, p95_ssnr = [], [], [], [], []

    for md in memory_data:
        ah  = md["A_hat"]
        as_ = md["A_sigma"]
        snr  = np.abs(ah / as_)
        ssnr = ah / as_
        p05_ah.append(np.percentile(ah, 5));    p25_ah.append(np.percentile(ah, 25))
        p50_ah.append(np.percentile(ah, 50));   p75_ah.append(np.percentile(ah, 75))
        p95_ah.append(np.percentile(ah, 95))
        p05_as.append(np.percentile(as_, 5));   p25_as.append(np.percentile(as_, 25))
        p50_as.append(np.percentile(as_, 50));  p75_as.append(np.percentile(as_, 75))
        p95_as.append(np.percentile(as_, 95))
        p05_snr.append(np.percentile(snr, 5));  p25_snr.append(np.percentile(snr, 25))
        p50_snr.append(np.percentile(snr, 50)); p75_snr.append(np.percentile(snr, 75))
        p95_snr.append(np.percentile(snr, 95))
        p05_ssnr.append(np.percentile(ssnr, 5));  p25_ssnr.append(np.percentile(ssnr, 25))
        p50_ssnr.append(np.percentile(ssnr, 50)); p75_ssnr.append(np.percentile(ssnr, 75))
        p95_ssnr.append(np.percentile(ssnr, 95))

    def to_arr(lst): return np.array(lst)
    p05_ah, p25_ah, p50_ah, p75_ah, p95_ah = map(to_arr, [p05_ah, p25_ah, p50_ah, p75_ah, p95_ah])
    p05_as, p25_as, p50_as, p75_as, p95_as = map(to_arr, [p05_as, p25_as, p50_as, p75_as, p95_as])
    p05_snr, p25_snr, p50_snr, p75_snr, p95_snr = map(to_arr, [p05_snr, p25_snr, p50_snr, p75_snr, p95_snr])
    p05_ssnr, p25_ssnr, p50_ssnr, p75_ssnr, p95_ssnr = map(to_arr, [p05_ssnr, p25_ssnr, p50_ssnr, p75_ssnr, p95_ssnr])

    def whiskers(ax, x, p05, p25, p50, p75, p95, color):
        """Draw 90% CI with caps and 50% CI as a thicker capless bar."""
        ax.errorbar(x, p50, yerr=[p50 - p05, p95 - p50],
                    fmt="o", ms=4, capsize=3, lw=1, color=color)
        ax.errorbar(x, p50, yerr=[p50 - p25, p75 - p50],
                    fmt="none", capsize=0, lw=3, color=color)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(max(14, n * 0.45), 14),
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # --- Row 1: A_hat ---
    whiskers(ax1, x, p05_ah, p25_ah, p50_ah, p75_ah, p95_ah, "C0")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.axhline(1, color="grey", lw=0.8, ls=":")
    ax1.set_ylabel(r"$\hat{A}$")
    ax1.set_yscale("symlog", linthresh=10)
    ax1.set_title(
        r"Memory amplitude $\hat{A}$, $\sigma_A$, $|\hat{A}/\sigma_A|$, $\hat{A}/\sigma_A$ per event"
        r"  —  thick: 50% CI,  thin+cap: 90% CI"
        f"\n{os.path.basename(memory_dir.rstrip('/'))}  ({n} events)"
    )

    # --- Row 2: A_sigma ---
    whiskers(ax2, x, p05_as, p25_as, p50_as, p75_as, p95_as, "C1")
    ax2.set_ylabel(r"$\sigma_A$")
    ax2.set_yscale("log")

    # --- Row 3: |A_hat / A_sigma| ---
    whiskers(ax3, x, p05_snr, p25_snr, p50_snr, p75_snr, p95_snr, "C2")
    ax3.set_ylabel(r"$|\hat{A}\,/\,\sigma_A|$")
    ax3.set_yscale("log")

    # --- Row 4: A_hat / A_sigma (signed) ---
    whiskers(ax4, x, p05_ssnr, p25_ssnr, p50_ssnr, p75_ssnr, p95_ssnr, "C3")
    ax4.axhline(0, color="k", lw=0.8, ls="--")
    ax4.set_ylabel(r"$\hat{A}\,/\,\sigma_A$")
    ax4.set_yscale("symlog", linthresh=1)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=90, fontsize=7)
    ax4.set_xlim(-0.5, n - 0.5)

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    main()
