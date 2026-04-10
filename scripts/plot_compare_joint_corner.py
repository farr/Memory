"""Compare joint_model_corner between two production runs."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from corner import corner

_JOINT_VARS = [
    "alpha_1", "alpha_2", "b", "beta",
    "frac_bpl", "frac_peak_1", "mu_peak_1", "sigma_peak_1",
    "frac_peak_2", "mu_peak_2", "sigma_peak_2",
    "mu_spin", "sigma_spin", "lamb",
    "f_iso", "mu_tilt", "sigma_tilt",
    "mu_tgr", "sigma_tgr",
]

RESULTS = "/mnt/home/misi/src/Memory/results"

COMPARISONS = [
    ("auto_o3o4a_joint", "auto_o3o4a"),
    ("nrsur_o4a_joint",  "nrsur_o4a"),
]

RUNS = [
    ("prod_20260402",  "C0"),
    ("prod_20260402b", "C1"),
]


def load(run, subdir):
    path = f"{RESULTS}/{run}/{subdir}/fit_joint_samples.dat"
    df = pd.read_csv(path, sep=" ")
    return df[_JOINT_VARS].values


for subdir, label in COMPARISONS:
    fig = None
    for i, (run, color) in enumerate(RUNS):
        data = load(run, subdir)
        fig = corner(
            data,
            labels=_JOINT_VARS,
            figsize=(20, 20),
            color=color,
            fig=fig,
            plot_density=False,
            plot_contours=True,
            hist_kwargs={"density": True},
        )

    # Add legend
    axes = np.array(fig.axes).reshape(len(_JOINT_VARS), len(_JOINT_VARS))
    for i, (run, color) in enumerate(RUNS):
        axes[0, 0].plot([], [], color=color, label=run)
    axes[0, 0].legend(loc="upper right", fontsize=10)

    outpath = f"{RESULTS}/prod_20260402b/{subdir}/joint_model_corner_compare.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")
