"""Plotting and ArviZ post-processing for hierarchical TGR population analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from corner import corner

from memory.kde_contour import kdeplot


def get_samples_df(fit):
    """Convert an ArviZ InferenceData posterior into a flat pandas DataFrame.

    Stacks chains and draws into a single 'sample' dimension, excluding
    any 'neff' diagnostic variables.

    Parameters
    ----------
    fit : arviz.InferenceData
        Inference result containing a posterior group.

    Returns
    -------
    pandas.DataFrame
        One row per sample, one column per non-neff parameter.
    """
    stacked_samples = fit.posterior.stack(sample=("chain", "draw"))
    samples_df = pd.DataFrame()
    for k, v in stacked_samples.data_vars.items():
        if "neff" not in k:
            samples_df[k] = v.values
    return samples_df


def create_plots(fit_joint, fit_tgr, parameter, outdir):
    """Create and save diagnostic plots comparing joint and TGR-only models.

    Generates: population KDE distribution, hyperparameter pairplot,
    TGR corner plot (overlaid if both fits available), and full joint
    model corner plot. Also saves per-fit CSV sample files. If either
    fit is None, plotting proceeds using only the available fit(s).

    Parameters
    ----------
    fit_joint : arviz.InferenceData or None
        Joint model inference result.
    fit_tgr : arviz.InferenceData or None
        TGR-only model inference result.
    parameter : str
        TGR parameter name (used for axis labels).
    outdir : str
        Directory to save plots and sample CSV files.
    """
    # Collect available fits
    fits = [("joint", fit_joint), ("tgr", fit_tgr)]
    fits = [(name, fit) for name, fit in fits if fit is not None]
    if len(fits) == 0:
        print("No fits provided; skipping plots.")
        return

    # Get sample dataframes and save sample data
    df_dict = {}
    for key, fit in fits:
        df = get_samples_df(fit)
        df["draw_tgr"] = np.random.normal(df["mu_tgr"], df["sigma_tgr"])
        df_dict[key] = df
        # Save sample data per available fit
        df.to_csv(
            f"{outdir}/fit_{key}_samples.dat", index=False, sep=" "
        )

    # Create population distribution plot
    plt.figure(figsize=(10, 6))
    x = {k: df["draw_tgr"] for k, df in df_dict.items()}
    sns.kdeplot(x, common_norm=False)
    plt.xlabel(parameter)
    plt.title(f"Population Distribution for {parameter}")
    plt.savefig(
        f"{outdir}/population_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create comparison plots
    dfs = []
    labels = []
    for key, df in df_dict.items():
        df["run"] = key
        dfs.append(df)
        labels.append(key)
    df = pd.concat(dfs, ignore_index=True)

    g = sns.PairGrid(
        df,
        x_vars=["mu_tgr", "sigma_tgr"],
        y_vars=["mu_tgr", "sigma_tgr"],
        hue="run",
        diag_sharey=False,
        corner=True,
    )

    g.map_diag(kdeplot, auto_bound=True)
    g.map_offdiag(kdeplot, y_min=0)

    g.axes[1, 0].set_ylim(0)
    g.axes[1, 1].set_xlim(0)

    for i, label in enumerate(labels):
        g.axes[1, 1].plot([], [], color=f"C{i}", label=label)
    g.axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4))

    plt.savefig(f"{outdir}/hyperparameters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # TGR parameters corner plot (overlay if both fits; otherwise single)
    fig = None
    for i, (key, fit) in enumerate(fits):
        if fit is not None:
            fig = corner(
                fit,
                var_names=["mu_tgr", "sigma_tgr"],
                figsize=(12, 12),
                plot_density=False,
                plot_contours=True,
                color=f"C{i}",
                truths=[0, 0],
                truth_color="k",
                fig=fig
            )
    if fig is not None:
        plt.savefig(
            f"{outdir}/tgr_comparison_corner.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Full joint model corner plot (only if joint available)
    if fit_joint is not None:
        corner(
            fit_joint,
            var_names=[
                "alpha",
                "beta",
                "mu_bump",
                "sigma_bump",
                "frac_bump",
                "mu_spin",
                "sigma_spin",
                "lamb",
                "mu_tgr",
                "sigma_tgr",
            ],
            figsize=(12, 12),
            color="C0",
        )
        plt.savefig(
            f"{outdir}/joint_model_corner.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Plots saved to {outdir}")
