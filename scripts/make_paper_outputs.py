#!/usr/bin/env python3
"""Generate paper LaTeX macros and the TGR comparison corner plot.

The source production results directory is intentionally hard-coded below so
the paper outputs carry checked-in data provenance.  The script discovers the
corresponding memory-only and joint analysis subdirectories, reads their ArviZ
NetCDF posterior files, and writes:

* a ``.tex`` file containing macro definitions for event counts and
  TGR hyperposterior summaries;
* a two-parameter corner plot comparing the memory-only and joint
  hyperposteriors for ``mu_tgr`` and ``sigma_tgr``.

By default outputs are written to ``paper/results_macros.tex`` and
``figures/tgr_comparison_corner.pdf``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from corner import corner

TGR_VARS = ("mu_tgr", "sigma_tgr")
SIGMA_LEVELS_2D = tuple(1.0 - np.exp(-0.5 * np.arange(1, 4) ** 2))
REPO_DIR = Path(__file__).resolve().parent.parent

# Paper data provenance: update this checked-in path when refreshing results.
RESULTS_DIR = Path("results") / "prod_20260428b"
DEFAULT_MACROS_OUTPUT = Path("paper") / "results_macros.tex"
DEFAULT_PLOT_OUTPUT = Path("figures")

plt.style.use(REPO_DIR / "style.mplstyle")

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt}}

width1 = jour_sizes["PRD"]["onecol"]
width2 = jour_sizes["PRD"]["twocol"]


@dataclass(frozen=True)
class ResultRun:
    """Resolved result files for one analysis run."""

    label: str
    directory: Path
    nc_file: Path
    analysis: str


def _non_comment_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def count_events(run_dir: Path) -> int:
    """Return the final analysis event count for *run_dir*.

    ``analyzed_events.txt`` records post-cut events and is preferred.  Older
    runs may only have ``event_files.txt``, so use that as a fallback.
    """

    for filename in ("analyzed_events.txt", "event_files.txt"):
        lines = _non_comment_lines(run_dir / filename)
        if lines:
            return len(lines)
    raise FileNotFoundError(
        f"No analyzed_events.txt or event_files.txt found in {run_dir}"
    )


def summarize_gaussian_draws(
    samples: np.ndarray,
) -> tuple[float, float, float]:
    """Draw from Normal(mu_tgr, sigma_tgr) and return mean, q05, q95.

    Each synthetic draw first selects one posterior hyperparameter sample,
    then draws one value from N(mu_tgr, sigma_tgr). This gives draws from the
    posterior-predictive mixture implied by the hyperposterior.
    """

    n_draws = len(samples[:,0])

    rng = np.random.default_rng()

    mu = np.asarray(samples[:, 0])
    sigma = np.asarray(samples[:, 1])

    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
        raise ValueError("mu_tgr/sigma_tgr samples contain non-finite values")

    if np.any(sigma < 0):
        raise ValueError("sigma_tgr contains negative values; cannot use as a Gaussian scale")

    sample_idx = rng.integers(0, len(samples), size=n_draws)
    gaussian_draws = rng.normal(
        loc=mu[sample_idx],
        scale=sigma[sample_idx],
    )

    low, median, high = np.quantile(gaussian_draws, [0.05, 0.5, 0.95])
    return (median, high - median, median - low)


def format_mean_ci(summary: tuple[float, float, float], precision: int) -> str:
    """Format mean and 5th--95th percentile interval for LaTeX."""

    median, plus, minus = summary
    return (
        f"{median:.{precision}f}"
        f"^{{+{plus:.{precision}f}}}_{{-{minus:.{precision}f}}}"
    )


def gaussian_draw_macros(
    stem: str,
    summary: tuple[float, float, float],
    precision: int,
) -> list[str]:
    """Return separate and combined macros for Gaussian-draw summaries.

    Use letter-only macro names because TeX command names do not naturally
    include digits.
    """

    median, plus, minus = summary
    return [
        latex_macro(f"{stem}", f"${format_mean_ci(summary, precision)}$"),
    ]


def load_tgr_samples(nc_file: Path) -> np.ndarray:
    """Return a flat ``(sample, 2)`` array for ``mu_tgr`` and ``sigma_tgr``."""

    print(nc_file)
    idata = az.from_netcdf(nc_file)
    missing = [name for name in TGR_VARS if name not in idata.posterior]
    if missing:
        raise KeyError(f"{nc_file} is missing posterior variable(s): {missing}")

    stacked = idata.posterior.stack(sample=("chain", "draw"))
    cols = []
    for name in TGR_VARS:
        values = np.asarray(stacked[name].values)
        if values.ndim != 1:
            raise ValueError(
                f"{name} in {nc_file} has shape {values.shape}; expected scalar samples"
            )
        cols.append(values)
    return np.column_stack(cols)


def summarize(samples: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Return median and central 90 percent interval deltas for each TGR var."""

    summaries: dict[str, tuple[float, float, float]] = {}
    for index, name in enumerate(TGR_VARS):
        low, median, high = np.quantile(samples[:, index], [0.05, 0.5, 0.95])
        summaries[name] = (median, high - median, median - low)
    return summaries


def discover_runs(prod_dir: Path) -> list[ResultRun]:
    """Find memory and joint result NetCDF files below *prod_dir*."""

    runs: list[ResultRun] = []
    for child in sorted(prod_dir.iterdir()):
        if not child.is_dir():
            continue
        for analysis in ("memory", "joint"):
            nc_file = child / f"result_{analysis}.nc"
            if nc_file.exists():
                runs.append(
                    ResultRun(
                        label=child.name,
                        directory=child,
                        nc_file=nc_file,
                        analysis=analysis,
                    )
                )
    return runs


def _choose_named_run(runs: Iterable[ResultRun], name: str, analysis: str) -> ResultRun:
    matches = [run for run in runs if run.analysis == analysis and run.label == name]
    if not matches:
        raise ValueError(f"Could not find {analysis} run named {name!r}")
    return matches[0]


def _matching_memory_label(joint_label: str, suffix: str) -> str:
    if joint_label.endswith("_joint"):
        return f"{joint_label[:-len('_joint')]}_{suffix}"
    return f"{joint_label}_{suffix}"


def choose_runs(
    runs: list[ResultRun],
    memory_run_name: str | None,
    joint_run_name: str | None,
    memory_count_run_name: str | None,
) -> tuple[ResultRun, ResultRun | None, ResultRun]:
    """Choose plot/stat runs and the event-count memory run."""

    joint_runs = [run for run in runs if run.analysis == "joint"]
    memory_runs = [run for run in runs if run.analysis == "memory"]
    if not memory_runs:
        raise ValueError("No result_memory.nc files found")

    joint_run = (
        _choose_named_run(runs, joint_run_name, "joint")
        if joint_run_name
        else (joint_runs[0] if joint_runs else None)
    )

    if memory_run_name:
        memory_run = _choose_named_run(runs, memory_run_name, "memory")
    elif joint_run is not None:
        preferred = [
            _matching_memory_label(joint_run.label, "memory-selected"),
            _matching_memory_label(joint_run.label, "memory"),
        ]
        memory_by_label = {run.label: run for run in memory_runs}
        memory_run = next(
            (memory_by_label[label] for label in preferred if label in memory_by_label),
            memory_runs[0],
        )
    else:
        memory_run = memory_runs[0]

    if memory_count_run_name:
        memory_count_run = _choose_named_run(runs, memory_count_run_name, "memory")
    else:
        memory_count_run = memory_run

    return memory_run, joint_run, memory_count_run


def format_int(value: int) -> str:
    return f"{value:,}"


def format_constraint(summary: tuple[float, float, float], precision: int) -> str:
    median, plus, minus = summary
    return (
        f"{median:.{precision}f}"
        f"^{{+{plus:.{precision}f}}}_{{-{minus:.{precision}f}}}"
    )


def latex_macro(name: str, value: str) -> str:
    """Return a macro definition that works whether or not *name* exists."""

    return (
        f"\\providecommand{{\\{name}}}{{}}\n"
        f"\\renewcommand{{\\{name}}}{{{value}\\xspace}}"
    )


def repo_relative(path: Path) -> str:
    """Return *path* relative to the repository when possible."""

    try:
        return str(path.resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def write_macros(
    path: Path,
    memory_run: ResultRun,
    joint_run: ResultRun | None,
    memory_count_run: ResultRun,
    precision: int,
) -> None:
    memory_samples = load_tgr_samples(memory_run.nc_file)
    memory_summary = summarize(memory_samples)
    joint_samples = load_tgr_samples(joint_run.nc_file)
    joint_summary = (
        summarize(joint_samples) if joint_run is not None else None
    )
    primary_summary = joint_summary or memory_summary

    memory_gaussian_summary = summarize_gaussian_draws(
        memory_samples,
    )
    joint_gaussian_summary = (
        summarize_gaussian_draws(
            joint_samples,
        )
        if joint_samples is not None
        else None
    )
    primary_gaussian_summary = joint_gaussian_summary or memory_gaussian_summary

    memory_count = count_events(memory_count_run.directory)
    selected_count = count_events(memory_run.directory)
    joint_count = count_events(joint_run.directory) if joint_run else selected_count

    lines = [
        "% Auto-generated by scripts/make_paper_outputs.py.",
        f"% Source results directory: {RESULTS_DIR}",
        f"% Source memory run: {repo_relative(memory_run.directory)}",
    ]
    if joint_run is not None:
        lines.append(f"% Source joint run: {repo_relative(joint_run.directory)}")
    lines.extend(
        [
            "",
            latex_macro("Nmemoryevents", format_int(memory_count)),
            latex_macro("Nevents", format_int(joint_count)),
            latex_macro("NMemorySelectedEvents", format_int(selected_count)),
            latex_macro("NMemoryOnlyEvents", format_int(memory_count)),
            latex_macro("NJointEvents", format_int(joint_count)),
            latex_macro("NLambdaGaussianDraws", format_int(len(joint_samples))),
            latex_macro(
                "muLambdaconstraint",
                f"${format_constraint(primary_summary['mu_tgr'], precision)}$",
            ),
            latex_macro(
                "sigmaLambdaconstraint",
                f"${format_constraint(primary_summary['sigma_tgr'], precision)}$",
            ),
            latex_macro(
                "MemoryMuLambdaconstraint",
                f"${format_constraint(memory_summary['mu_tgr'], precision)}$",
            ),
            latex_macro(
                "MemorySigmaLambdaconstraint",
                f"${format_constraint(memory_summary['sigma_tgr'], precision)}$",
            ),
        ]
    )
    lines.extend(
        gaussian_draw_macros(
            "Aconstraint",
            primary_gaussian_summary,
            precision,
        )
    )
    lines.extend(
        gaussian_draw_macros(
            "MemoryAconstraint",
            memory_gaussian_summary,
            precision,
        )
    )

    if joint_summary is not None:
        lines.extend(
            [
                latex_macro(
                    "JointMuLambdaconstraint",
                    f"${format_constraint(joint_summary['mu_tgr'], precision)}$",
                ),
                latex_macro(
                    "JointSigmaLambdaconstraint",
                    f"${format_constraint(joint_summary['sigma_tgr'], precision)}$",
                ),
            ]
        )
        lines.extend(
            gaussian_draw_macros(
                "JointAconstraint",
                joint_gaussian_summary,
                precision,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

def make_memory_mean_and_uncertainty(
    path: Path,
    memory_analysis: Path,
    dpi: int,
) -> None:
    events = ['GW150914_095045', 'GW190521_074359', 'GW230627_015337', 'GW230814_230901', 'GW231123_135430', 'GW250114_082203']

    memory_values = {}
    for event in events:
        memory_values[event] = {}
        with h5py.File(memory_analysis / f"{event}/memory_results.h5") as input_file:
            if np.any(['IMRPhenomXO4a' in x for x in input_file.keys()]):
                preferred_phenom_key = 'IMRPhenomXO4a'
            elif np.any(['IMRPhenomXPHM-SpinTaylor' in x for x in input_file.keys()]):
                preferred_phenom_key = 'IMRPhenomXPHM-SpinTaylor'
            elif np.any(['IMRPhenomXPHM' in x for x in input_file.keys()]):
                preferred_phenom_key = 'IMRPhenomXPHM'
            else:
                preferred_phenom_key = 'Phenom'
                
            break_me = False
            for label in input_file.keys():
                if 'Phenom' in label:
                    if break_me:
                        continue
                    if preferred_phenom_key in label:
                        break_me = True
                    else:
                        continue
    
                if 'NRSur' in label:
                    waveform_name = 'NRSur'
                elif 'EOB' in label:
                    waveform_name = 'EOB'
                else:
                    waveform_name = 'Phenom'
        
                memory_values[event][waveform_name] = {}
                for key in input_file[label].keys():
                    memory_values[event][waveform_name][key] = np.array(input_file[label][key])

    def ci(x):
        q05, q25, q50, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95])
        return q50, q25, q75, q05, q95
    
    events = list(memory_values.keys())
    
    model_order = ["NRSur", "EOB", "Phenom"]
    models = [m for m in model_order if any(m in ev for ev in memory_values.values())]
    
    # Colorblind-friendly, high-contrast qualitative colors
    # Chosen to be distinct and aesthetically balanced
    colors = {
        "NRSur": "#0072B2",   # blue
        "EOB":   "#E69F00",   # orange
        "Phenom":"#CC79A7",   # reddish purple
    }
    
    fig, axes = plt.subplots(
        3, 1,
        figsize=(width2, 0.3 * width2),
        sharex=True,
        constrained_layout=True
    )
    
    row_defs = [
        ("A_hat", r"$\mu_{s}$"),
        ("A_sigma", r"$\sigma_{s}$"),
        ("A_sample", r"$\mathcal{N}(\mu_{s},\sigma_{s})$"),
    ]
    
    event_gap = 0.5
    centers = np.arange(len(events)) * event_gap
    
    offset_scale = 0.10
    if len(models) == 1:
        offsets = {models[0]: 0.0}
    else:
        base = np.arange(len(models)) - 0.5 * (len(models) - 1)
        offsets = {m: offset_scale * base[i] for i, m in enumerate(models)}
    
    for ax, (key, ylabel) in zip(axes, row_defs):
        for i, event in enumerate(events):
            xc = centers[i]
    
            for model in models:
                if model not in memory_values[event]:
                    continue
    
                d = memory_values[event][model]
    
                if key == "A_hat":
                    x = np.asarray(d["A_hat"])
                elif key == "A_sigma":
                    x = np.asarray(d["A_sigma"])
                else:
                    x = np.asarray(d["A_sample"])
    
                med, lo50, hi50, lo90, hi90 = ci(x)
                xpos = xc + offsets[model]
                color = colors[model]
    
                ax.errorbar(
                    xpos, med,
                    yerr=[[med - lo90], [hi90 - med]],
                    fmt='none', color=color, lw=1.5, alpha=0.7, capsize=4
                )
    
                ax.errorbar(
                    xpos, med,
                    yerr=[[med - lo50], [hi50 - med]],
                    fmt='none', color=color, lw=3, capsize=0
                )
                ax.plot(xpos, med, "o", color=color, ms=4)
    
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels([x.split("_")[0] for x in events], rotation=0, ha="center")
    
    # Add horizontal padding so first/last labels stay inside the plot border
    edge_pad = 0.15
    axes[-1].set_xlim(centers[0] - edge_pad, centers[-1] + edge_pad)
    
    axes[2].axhline(1, color='k', ls='--', zorder=-1, label='GR')
    
    axes[2].legend(loc='upper right', frameon=True, framealpha=1, fontsize=6)
    
    axes[0].set_yscale('symlog', linthresh=10)
    axes[1].set_yscale('log')
    axes[2].set_yscale('symlog', linthresh=10)
    
    handles = [Line2D([0], [0], color=colors[m], lw=1.5, label=m) for m in models]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(models),
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    fig.align_ylabels()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / "memory_mean_stddev_from_posteriors.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def make_tgr_corner(
    path: Path,
    memory_run: ResultRun,
    joint_run: ResultRun | None,
    dpi: int,
) -> None:
    data_memory = az.from_netcdf(memory_run.nc_file)
    data_joint = az.from_netcdf(joint_run.nc_file)
    
    color_memory = "#0072B2"
    color_joint  = "#E69F00"
    
    fig = plt.figure(figsize=(width1, width1))
    
    fig = corner(
        data_memory,
        var_names=["mu_tgr", "sigma_tgr"],
        figsize=(width1, width1),
        plot_density=False,
        plot_contours=True,
        plot_datapoints=False,
        fill_contours=False,
        color=color_memory,
        levels=SIGMA_LEVELS_2D,
        fig=fig,
    )
    
    fig = corner(
        data_joint,
        var_names=["mu_tgr", "sigma_tgr"],
        labels=[r"memory $\mu_{\Lambda}$", r"memory $\sigma_{\Lambda}$"],
        figsize=(width1, width1),
        plot_density=False,
        plot_contours=True,
        plot_datapoints=False,
        fill_contours=False,
        color=color_joint,
        levels=SIGMA_LEVELS_2D,
        truths=[1, 0],
        truth_color="k",
        fig=fig,
    )
    
    axes = np.array(fig.axes).reshape((2, 2))
    for ax in axes[-1, :]:  # bottom row
        ax.xaxis.set_label_coords(0.5, -0.2)
    for ax in axes[:, 0]:  # left column
        ax.yaxis.set_label_coords(-0.2, 0.5)

    ax.set_xlim(-18, 18)
    ax.set_ylim(0, 18)
    
    mem_line = Line2D([], [], lw=1.5, color=color_memory, label='memory only')
    joint_line = Line2D([], [], lw=1.5, color=color_joint, label='joint')
    truth_line = Line2D([], [], lw=1.5, color='k', linestyle='-', label='GR')
    
    ax_empty = axes[0, 1]
    ax_empty.axis("off")
    ax_empty.legend(
        handles=[mem_line, joint_line, truth_line],
        loc="upper right",
        frameon=False,
        handlelength=1.2
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / "tgr_comparison_corner.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def make_forecast(
    path: Path,
    forecast_run: Path,
    dpi: int,
) -> None:
    df = pd.read_csv(forecast_run / "fisher_forecast_summary.csv").sort_values(["scenario", "n_total"])
    assumptions = pd.read_csv(forecast_run / "fisher_forecast_assumptions.csv").set_index("key")["value"]
    
    target = float(assumptions["target_halfwidth"])
    current = float(assumptions["current_mu_halfwidth68"])
    n_current = int(float(assumptions["current_catalog_size"]))
    quick_guess = n_current * (current/target)**(2)
    
    rate_O4 = 180   # events/year
    rate_O5 = 500   # events/year
    
    fig, ax = plt.subplots(figsize=(width2, 0.35 * width2))
    
    color_O4 = "#0072B2"
    color_O5 = "#E69F00"
    colors = {
        "current": color_O4,
        "O5a": color_O5
    }
    labels = {
        "current": "O4a-like observations",
        "O5a": "O5a-like observations"
    }
    
    for i, (scenario, g) in enumerate(df.groupby("scenario", sort=False)):
        if scenario != "current":
            continue
        x = g["n_total"].to_numpy()
        idx2 = np.argmin(abs(x - 3000)) + 1
        y50 = g["mu_halfwidth68_p50"].to_numpy()
        y16 = g["mu_halfwidth68_p16"].to_numpy()
        y84 = g["mu_halfwidth68_p84"].to_numpy()
    
        ax.fill_between(x[:idx2], y16[:idx2], y84[:idx2], alpha=0.15, color=colors[scenario])
        ax.plot(x[:idx2], y50[:idx2], label=labels[scenario], color=colors[scenario])
    
    ax.axhline(target, ls="--", lw=1.5, label=fr"$1\sigma$ away from zero", color="k")
    ax.axvline(n_current, ls=":", lw=1.5, color="k")
    ax.axvline(
        quick_guess,
        ls="-.",
        lw=1.5,
        color="0.35",
        label=fr"$1/\sqrt{{N}}$ estimate = {quick_guess:,.0f}",
    )
    
    ax.set_ylim(0)
    
    ax.text(
        n_current + 35,
        0.42,
        fr"$N={n_current}$",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        fontsize=8.2,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="white", alpha=1.0),
    )
    
    ax.set_xlabel("Number of GW observations")
    ax.set_ylabel(r"68\% half-width on $\mu_{\Lambda}$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, framealpha=1)
    
    # --- Conversion functions: number of events <-> observing time since current catalog ---
    def N_to_time_O4(N):
        return (np.asarray(N) - n_current) / rate_O4
    
    def time_to_N_O4(t):
        return n_current + np.asarray(t) * rate_O4
    
    def N_to_time_O5(N):
        return (np.asarray(N) - n_current) / rate_O5
    
    def time_to_N_O5(t):
        return n_current + np.asarray(t) * rate_O5
    
    # First top axis: O4a runtime since current catalog
    secax_O4 = ax.secondary_xaxis("top", functions=(N_to_time_O4, time_to_N_O4))
    secax_O4.set_xlabel("Time from current catalog [yr] at O4a sensitivity", labelpad=5)
    secax_O4.tick_params(axis="x", colors=color_O4)
    secax_O4.xaxis.label.set_color(color_O4)
    
    # Second top axis: O5a runtime since current catalog
    secax_O5 = ax.secondary_xaxis(1.25, functions=(N_to_time_O5, time_to_N_O5))
    secax_O5.set_xlabel("Time from current catalog [yr] at O5a sensitivity", labelpad=5)
    secax_O5.tick_params(axis="x", colors=color_O5)
    secax_O5.xaxis.label.set_color(color_O5)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / "forecast_w_time.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LaTeX macros and tgr_comparison_corner from the "
            "hard-coded production hierarchical-analysis results directory."
        )
    )
    parser.add_argument(
        "--macros-output",
        type=Path,
        default=DEFAULT_MACROS_OUTPUT,
        help=f"Output .tex macro file. Defaults to {DEFAULT_MACROS_OUTPUT}.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=f"Output corner plot. Defaults to {DEFAULT_PLOT_OUTPUT}.",
    )
    parser.add_argument(
        "--memory-run",
        help=(
            "Memory result subdirectory to summarize/plot. Defaults to the "
            "memory-selected run matching the joint run, if present."
        ),
    )
    parser.add_argument(
        "--joint-run",
        help="Joint result subdirectory to plot and summarize. Defaults to first joint run.",
    )
    parser.add_argument(
        "--memory-count-run",
        help=(
            "Memory result subdirectory used for \\Nmemoryevents. Defaults to "
            "the same run used for memory-only summaries and plotting."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal places for posterior summaries (default: 2).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Plot DPI (default: 300).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write macros; skip the corner plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = (REPO_DIR / RESULTS_DIR).resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)

    macros_output = (
        args.macros_output.expanduser()
        if args.macros_output.is_absolute()
        else REPO_DIR / args.macros_output
    ).resolve()
    plot_output = (
        args.plot_output.expanduser()
        if args.plot_output.is_absolute()
        else REPO_DIR / args.plot_output
    ).resolve()

    runs = discover_runs(results_dir)
    memory_run, joint_run, memory_count_run = choose_runs(
        runs, args.memory_run, args.joint_run, args.memory_count_run
    )

    memory_analysis = REPO_DIR / "analysis"
    
    forecast_run = results_dir / "forecast_fisher"

    write_macros(
        macros_output,
        memory_run=memory_run,
        joint_run=joint_run,
        memory_count_run=memory_count_run,
        precision=args.precision,
    )
    print(f"Wrote macros: {macros_output}")
    primary_label = joint_run.label if joint_run is not None else memory_run.label
    print(f"  primary summary run: {primary_label}")
    print(f"  memory comparison run: {memory_run.label}")
    print(f"  memory count run:   {memory_count_run.label}")
    if joint_run is not None:
        print(f"  joint run:          {joint_run.label}")

    if not args.no_plot:
        make_memory_mean_and_uncertainty(
            plot_output,
            memory_analysis=memory_analysis,
            dpi=args.dpi,
        )
        print(f"Wrote memory mean and uncertainty plot.")
        
        make_tgr_corner(
            plot_output,
            memory_run=memory_run,
            joint_run=joint_run,
            dpi=args.dpi,
        )
        print(f"Wrote TGR corner plot.")

        make_forecast(
            plot_output,
            forecast_run=forecast_run,
            dpi=args.dpi,
        )
        print(f"Wrote forecast plot.")


if __name__ == "__main__":
    main()
