#!/usr/bin/env python3
"""Diagnostic plot of A_hat and A_sigma distributions across events."""

import glob
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory.hierarchical.data import load_memory_data
from scripts.run_hierarchical_analysis import _filter_to_memory_events

POSTERIOR_GLOB = "/mnt/home/ccalvk/ceph/GWTC-4/IGWN-GWTC4p0-*-combined_PEDataRelease.hdf5"
MEMORY_DIR = "/mnt/home/kmitman/work/memory_pop/analysis"
WAVEFORM_LABEL = "C00:NRSur7dq4"
OUTFILE = "results/memory/diagnostics_Ahat_Asigma.png"


def main():
    posterior_files = sorted(glob.glob(POSTERIOR_GLOB))
    kept_files, _ = _filter_to_memory_events(posterior_files, MEMORY_DIR)
    memory_data = load_memory_data(kept_files, MEMORY_DIR, WAVEFORM_LABEL)

    names = [md["event_name"] for md in memory_data]
    n = len(names)
    x = np.arange(n)

    # Compute percentiles for each event
    p05_ah, p25_ah, p50_ah, p75_ah, p95_ah = [], [], [], [], []
    p05_as, p25_as, p50_as, p75_as, p95_as = [], [], [], [], []
    p05_snr, p25_snr, p50_snr, p75_snr, p95_snr = [], [], [], [], []

    for md in memory_data:
        ah = md["A_hat"]
        as_ = md["A_sigma"]
        snr = np.abs(ah / as_)
        p05_ah.append(np.percentile(ah, 5))
        p25_ah.append(np.percentile(ah, 25))
        p50_ah.append(np.percentile(ah, 50))
        p75_ah.append(np.percentile(ah, 75))
        p95_ah.append(np.percentile(ah, 95))
        p05_as.append(np.percentile(as_, 5))
        p25_as.append(np.percentile(as_, 25))
        p50_as.append(np.percentile(as_, 50))
        p75_as.append(np.percentile(as_, 75))
        p95_as.append(np.percentile(as_, 95))
        p05_snr.append(np.percentile(snr, 5))
        p25_snr.append(np.percentile(snr, 25))
        p50_snr.append(np.percentile(snr, 50))
        p75_snr.append(np.percentile(snr, 75))
        p95_snr.append(np.percentile(snr, 95))

    p05_ah  = np.array(p05_ah);  p25_ah  = np.array(p25_ah)
    p50_ah  = np.array(p50_ah);  p75_ah  = np.array(p75_ah);  p95_ah  = np.array(p95_ah)
    p05_as  = np.array(p05_as);  p25_as  = np.array(p25_as)
    p50_as  = np.array(p50_as);  p75_as  = np.array(p75_as);  p95_as  = np.array(p95_as)
    p05_snr = np.array(p05_snr); p25_snr = np.array(p25_snr)
    p50_snr = np.array(p50_snr); p75_snr = np.array(p75_snr); p95_snr = np.array(p95_snr)

    def whiskers(ax, x, p05, p25, p50, p75, p95, color):
        """Draw 90% CI with caps and 50% CI as a thicker capless bar."""
        ax.errorbar(x, p50,
                    yerr=[p50 - p05, p95 - p50],
                    fmt="o", ms=4, capsize=3, lw=1, color=color)
        ax.errorbar(x, p50,
                    yerr=[p50 - p25, p75 - p50],
                    fmt="none", capsize=0, lw=3, color=color)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(max(14, n * 0.45), 11),
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # --- Row 1: A_hat ---
    whiskers(ax1, x, p05_ah, p25_ah, p50_ah, p75_ah, p95_ah, "C0")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.axhline(1, color="grey", lw=0.8, ls=":")
    ax1.set_ylabel(r"$\hat{A}$")
    ax1.set_yscale("symlog", linthresh=10)
    ax1.set_title(r"Memory amplitude $\hat{A}$, $\sigma_A$, and $|\hat{A}/\sigma_A|$ per event"
                  r"  —  thick: 50% CI,  thin+cap: 90% CI")

    # --- Row 2: A_sigma ---
    whiskers(ax2, x, p05_as, p25_as, p50_as, p75_as, p95_as, "C1")
    ax2.set_ylabel(r"$\sigma_A$")
    ax2.set_yscale("log")

    # --- Row 3: |A_hat / A_sigma| ---
    whiskers(ax3, x, p05_snr, p25_snr, p50_snr, p75_snr, p95_snr, "C2")
    ax3.set_ylabel(r"$|\hat{A}\,/\,\sigma_A|$")
    ax3.set_yscale("log")

    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=90, fontsize=7)
    ax3.set_xlim(-0.5, n - 0.5)

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
