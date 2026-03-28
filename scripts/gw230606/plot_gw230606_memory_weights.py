#!/usr/bin/env python3
"""Histograms of A_hat and A_sigma from the GW230606_004305 memory computation.

Reads memory_results.h5 from the specified directory and plots per-sample
ML amplitude (A_hat) and amplitude uncertainty (A_sigma).

Outputs: results/gw230606_memory_weights_tukey05.png
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

RESULTS_H5 = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "results", "memory_tukey05", "GW230606_004305", "memory_results.h5",
)
OUTFILE = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "results", "gw230606_memory_weights_tukey05.png",
)

EVENT = "GW230606_004305"
TUKEY_ALPHA = 0.5


def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    with h5py.File(RESULTS_H5, "r") as f:
        label = list(f.keys())[0]
        A_hat   = np.real(f[label]["A_hat"][:])
        A_sigma = np.real(f[label]["A_sigma"][:])
        lw      = np.real(f[label]["log_weight"][:])

    n = len(A_hat)
    print(f"Loaded {n} samples from label '{label}'")
    print(f"  A_hat:    mean={np.mean(A_hat):.2f}, std={np.std(A_hat):.2f}, "
          f"median={np.median(A_hat):.2f}")
    print(f"  A_sigma:  mean={np.mean(A_sigma):.2f}, std={np.std(A_sigma):.2f}, "
          f"median={np.median(A_sigma):.2f}")
    print(f"  A_hat/A_sigma: mean={np.mean(A_hat/A_sigma):.2f}, std={np.std(A_hat/A_sigma):.2f}, "
          f"median={np.median(A_hat/A_sigma):.2f}")
    print(f"  log_weight: mean={np.mean(lw):.2f}, std={np.std(lw):.2f}")

    snr = A_hat / A_sigma

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"wspace": 0.30})

    # --- A_hat ---
    ax = axes[0]
    ax.hist(A_hat, bins=30, color="C0", edgecolor="white", linewidth=0.4)
    ax.axvline(1,                  color="k",  lw=1.2, ls="--", label="GR  (A = 1)")
    ax.axvline(np.median(A_hat),   color="C1", lw=1.2, ls=":",
               label=f"median = {np.median(A_hat):.1f}")
    ax.set_xlabel(r"$\hat{A}_\mathrm{mem}$  (ML amplitude per sample)")
    ax.set_ylabel("Count")
    ax.set_title(r"$\hat{A}_\mathrm{mem} = \mathrm{Re}\{\langle h_m | r \rangle\} / \langle h_m | h_m \rangle$",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    # --- A_sigma ---
    ax = axes[1]
    ax.hist(A_sigma, bins=30, color="C2", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(A_sigma), color="C1", lw=1.2, ls=":",
               label=f"median = {np.median(A_sigma):.1f}")
    ax.set_xlabel(r"$\sigma_A$  (amplitude uncertainty per sample)")
    ax.set_ylabel("Count")
    ax.set_title(r"$\sigma_A = 1 / \sqrt{\langle h_m | h_m \rangle}$", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    # --- mu/sigma ---
    ax = axes[2]
    ax.hist(snr, bins=30, color="C3", edgecolor="white", linewidth=0.4)
    ax.axvline(0,                color="k",  lw=1.2, ls="--", label="zero")
    ax.axvline(np.median(snr),   color="C1", lw=1.2, ls=":",
               label=f"median = {np.median(snr):.2f}")
    ax.set_xlabel(r"$\hat{A} / \sigma_A$  (per-sample memory SNR)")
    ax.set_ylabel("Count")
    ax.set_title(r"$\hat{A} / \sigma_A$", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"{EVENT}  |  {label}  |  Tukey alpha = {TUKEY_ALPHA}, BayesWave H1 frame\n"
        f"{n} posterior samples",
        fontsize=10,
    )

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
