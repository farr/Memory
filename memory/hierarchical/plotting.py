"""Plotting and ArviZ post-processing for hierarchical TGR population analysis."""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from corner import corner

from memory.kde_contour import kdeplot

logger = logging.getLogger(__name__)


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
        if "neff" not in k and v.values.ndim == 1:
            samples_df[k] = v.values
    return samples_df


def _has_tgr(fit):
    """Return True if *fit* contains TGR hyperparameters."""
    return fit is not None and "mu_tgr" in fit.posterior.data_vars


_ASTRO_VARS = [
    "alpha_1", "alpha_2", "b", "beta",
    "frac_bpl", "frac_peak_1", "mu_peak_1", "sigma_peak_1",
    "frac_peak_2", "mu_peak_2", "sigma_peak_2",
    "mu_spin", "sigma_spin", "lamb",
]


def create_plots(fit_astro, fit_joint, fit_memory, outdir):
    """Create and save diagnostic plots for all available model fits.

    Generates:
    - Astrophysical parameter corner plot (astro and/or joint, overlaid).
    - Population distribution KDE, hyperparameter pairplot, and TGR corner
      plot (joint and/or memory, overlaid) — only when TGR parameters are
      present in at least one fit.
    - Full joint model corner plot — only when fit_joint is available.

    Also saves per-fit CSV sample files. Fits that are None are skipped.

    Parameters
    ----------
    fit_astro : arviz.InferenceData or None
        Astrophysical-only model inference result.
    fit_joint : arviz.InferenceData or None
        Joint astrophysical + TGR model inference result.
    fit_memory : arviz.InferenceData or None
        TGR (memory-only) model inference result.
    outdir : str
        Directory to save plots and sample CSV files.
    """
    all_fits = [
        ("astro",   fit_astro),
        ("joint",   fit_joint),
        ("memory",  fit_memory),
    ]
    available = [(name, fit) for name, fit in all_fits if fit is not None]
    if not available:
        logger.warning("No fits provided; skipping plots.")
        return

    # Save sample CSVs
    for name, fit in available:
        get_samples_df(fit).to_csv(
            f"{outdir}/fit_{name}_samples.dat", index=False, sep=" "
        )

    # --- Astrophysical corner plot ----------------------------------------
    astro_fits = [(n, f) for n, f in available if n in ("astro", "joint")]
    if astro_fits:
        fig = None
        for i, (name, fit) in enumerate(astro_fits):
            fig = corner(
                fit,
                var_names=_ASTRO_VARS,
                figsize=(16, 16),
                plot_density=False,
                plot_contours=True,
                color=f"C{i}",
                fig=fig,
                labels=_ASTRO_VARS,
            )
        plt.savefig(
            f"{outdir}/astro_corner.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info("Saved astro_corner.png")

    # --- TGR plots (only when at least one fit has TGR params) -----------
    tgr_fits = [(n, f) for n, f in available if _has_tgr(f)]
    if tgr_fits:
        # Population distribution
        plt.figure(figsize=(10, 6))
        for i, (name, fit) in enumerate(tgr_fits):
            df = get_samples_df(fit)
            draws = np.random.normal(df["mu_tgr"], df["sigma_tgr"])
            sns.kdeplot(draws, label=name, color=f"C{i}")
        plt.xlabel("A")
        plt.title("Population distribution of memory amplitude A")
        plt.legend()
        plt.savefig(
            f"{outdir}/population_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Hyperparameter pairplot
        dfs = []
        for name, fit in tgr_fits:
            df = get_samples_df(fit)
            df["run"] = name
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        g = sns.PairGrid(
            df_all,
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
        for i, (name, _) in enumerate(tgr_fits):
            g.axes[1, 1].plot([], [], color=f"C{i}", label=name)
        g.axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4))
        plt.savefig(
            f"{outdir}/hyperparameters.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # TGR corner plot
        fig = None
        for i, (name, fit) in enumerate(tgr_fits):
            fig = corner(
                fit,
                var_names=["mu_tgr", "sigma_tgr"],
                figsize=(8, 8),
                plot_density=False,
                plot_contours=True,
                color=f"C{i}",
                truths=[1, 0],
                truth_color="k",
                fig=fig,
            )
        plt.savefig(
            f"{outdir}/tgr_comparison_corner.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        logger.info("Saved TGR plots")

    # --- Full joint corner plot ------------------------------------------
    if fit_joint is not None:
        joint_vars = _ASTRO_VARS + (
            ["mu_tgr", "sigma_tgr"] if _has_tgr(fit_joint) else []
        )
        corner(
            fit_joint,
            var_names=joint_vars,
            figsize=(16, 16),
            color="C1",
        )
        plt.savefig(
            f"{outdir}/joint_model_corner.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info("Saved joint_model_corner.png")

    logger.info("Plots saved to %s", outdir)
