#!/usr/bin/env python3
"""Compare memory vs memory-selected posterior distributions for mu_tgr and sigma_tgr."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

RESULTS = Path(__file__).resolve().parent.parent / "results" / "prod_20260402c"

runs = {
    "memory (148 events)": RESULTS / "auto_o3o4a_memory" / "fit_memory_samples.dat",
    "memory-selected (146 events)": RESULTS / "auto_o3o4a_memory-selected" / "fit_memory_samples.dat",
}

colors = ["#1f77b4", "#ff7f0e"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

all_samples = {}
for (label, path), color in zip(runs.items(), colors):
    df = pd.read_csv(path, sep=" ")
    all_samples[label] = df

    for ax, param in zip(axes[:2], ["mu_tgr", "sigma_tgr"]):
        samples = df[param].values
        ax.hist(samples, bins=60, density=True, histtype="step",
                label=label, color=color, linewidth=1.5)
        ax.axvline(np.median(samples), color=color, linestyle="--", linewidth=1)

axes[0].set_xlabel(r"$\mu_\mathrm{TGR}$")
axes[1].set_xlabel(r"$\sigma_\mathrm{TGR}$")
for ax in axes[:2]:
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

# Third panel: posterior predictive population distribution p(A)
# p(A) = mean over posterior samples of N(A | mu_tgr, sigma_tgr)
A = np.linspace(-30, 30, 500)
for (label, df), color in zip(all_samples.items(), colors):
    mu = df["mu_tgr"].values
    sigma = df["sigma_tgr"].values
    # Average Gaussian over posterior samples
    ppd = np.mean(norm.pdf(A[:, None], mu[None, :], sigma[None, :]), axis=1)
    axes[2].plot(A, ppd, label=label, color=color, linewidth=1.5)

axes[2].set_xlabel(r"$A$")
axes[2].set_ylabel("p(A)")
axes[2].set_title("Population distribution of A")
axes[2].legend(fontsize=9)

fig.suptitle("Effect of joint selection criteria on memory-only analysis\n(auto waveform, O3+O4a)")
fig.tight_layout()

outpath = RESULTS / "memory_selection_comparison.png"
fig.savefig(outpath, dpi=150)
print(f"Saved {outpath}")
