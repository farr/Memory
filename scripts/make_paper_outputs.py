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


def round_to_nearest_500(value: float | int) -> int:
    """Round to the nearest 500 using half-up rounding."""
    return int(500 * np.floor(float(value) / 500.0 + 0.5))


def format_optional_int(value: int | None) -> str:
    return "N/A" if value is None else format_int(value)


def forecast_crossing(
    df: pd.DataFrame,
    target: float | None = None,
    metric_col: str = "metric_p50",
) -> float | None:
    """Return the interpolated first N where the forecast metric crosses target."""

    if target is None:
        if "target" in df and not df["target"].dropna().empty:
            target = float(df["target"].dropna().iloc[0])
        else:
            target = 1.0

    g = df.sort_values("n_total")
    x = g["n_total"].to_numpy(float)
    y = g[metric_col].to_numpy(float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) == 0:
        return None

    if y[0] <= target:
        return float(x[0])

    for i in range(1, len(x)):
        if y[i] <= target:
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[i - 1], y[i]

            if y1 == y0:
                return float(x1)

            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

    return None


def rounded_forecast_crossing(
    df: pd.DataFrame,
    target: float | None = None,
    metric_col: str = "metric_p50",
) -> int | None:
    crossing = forecast_crossing(df, target=target, metric_col=metric_col)
    return None if crossing is None else round_to_nearest_500(crossing)


def forecast_csv_for_kind(prod_dir: Path, kind: str) -> Path | None:
    """Find memory/joint parametric forecast CSVs under the production directory."""

    csv_names = (
        "forecast_A_parametric.csv",
        "forecast_A_parametric_v2.csv",
    )

    candidate_dirs = (
        prod_dir / f"forecast_parameteric_invchi2_{kind}",  # current typo
        prod_dir / f"forecast_parametric_invchi2_{kind}",   # corrected spelling
    )

    for run_dir in candidate_dirs:
        for csv_name in csv_names:
            candidate = run_dir / csv_name
            if candidate.exists():
                return candidate

    return None


def rounded_forecast_crossing_for_kind(prod_dir: Path, kind: str) -> int | None:
    csv_path = forecast_csv_for_kind(prod_dir, kind)
    if csv_path is None:
        return None
    return rounded_forecast_crossing(pd.read_csv(csv_path))


def summarize_gaussian_draws(
    samples: np.ndarray,
) -> tuple[float, float, float]:
    """Draw from Normal(mu_tgr, sigma_tgr) and return mean, q16, q85.

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

    low, median, high = np.quantile(gaussian_draws, [0.16, 0.5, 0.84])
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
    """Return median and central 68 percent interval deltas for each TGR var."""

    summaries: dict[str, tuple[float, float, float]] = {}
    for index, name in enumerate(TGR_VARS):
        low, median, high = np.quantile(samples[:, index], [0.16, 0.5, 0.84])
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

    prod_dir = memory_run.directory.parent

    memory_forecast_crossing = rounded_forecast_crossing_for_kind(prod_dir, "memory")
    joint_forecast_crossing = rounded_forecast_crossing_for_kind(prod_dir, "joint")
    primary_forecast_crossing = (
        joint_forecast_crossing
        if joint_forecast_crossing is not None
        else memory_forecast_crossing
    )
        
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
            latex_macro(
                "NForecastCrossing",
                format_optional_int(primary_forecast_crossing),
            ),
            latex_macro(
                "NMemoryForecastCrossing",
                format_optional_int(memory_forecast_crossing),
            ),
            latex_macro(
                "NJointForecastCrossing",
                format_optional_int(joint_forecast_crossing),
            ),
            latex_macro("NAGaussianDraws", format_int(len(joint_samples))),
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
    """Plot parametric posterior-predictive A forecasts for memory-only and joint runs.

    Expected forecast directories, relative to RESULTS_DIR:

        forecast_parameteric_invchi2_memory/
        forecast_parameteric_invchi2_joint/
    """

    color_memory = "#0072B2"
    color_joint = "#E69F00"

    forecast_run = Path(forecast_run)
    prod_dir = forecast_run.parent

    csv_names = (
        "forecast_A_parametric.csv",
    )

    def has_forecast_csv(run_dir: Path) -> bool:
        return any((run_dir / name).exists() for name in csv_names)

    def csv_for(run_dir: Path) -> Path:
        for name in csv_names:
            candidate = run_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No parametric forecast CSV found in {run_dir}. "
            f"Tried: {', '.join(csv_names)}"
        )

    def find_run(kind: str) -> Path:
        candidates = [
            prod_dir / f"forecast_parameteric_invchi2_{kind}",  # matches current typo
            prod_dir / f"forecast_parametric_invchi2_{kind}",   # corrected spelling
        ]

        if kind in forecast_run.name and has_forecast_csv(forecast_run):
            candidates.insert(0, forecast_run)

        for token in ("memory", "joint"):
            if token in forecast_run.name:
                candidates.append(forecast_run.with_name(forecast_run.name.replace(token, kind)))

        for candidate in candidates:
            if has_forecast_csv(candidate):
                return candidate

        raise FileNotFoundError(
            f"Could not find the {kind} parametric forecast directory near {prod_dir}"
        )

    def load_forecast(run_dir: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_for(run_dir)).sort_values("n_total")
        needed = {"n_total", "metric_p16", "metric_p50", "metric_p84"}
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"{csv_for(run_dir)} is missing columns: {sorted(missing)}")
        return df

    memory_dir = find_run("memory")
    joint_dir = find_run("joint")

    runs = [
        ("memory only", load_forecast(memory_dir), color_memory),
        ("joint", load_forecast(joint_dir), color_joint),
    ]

    targets = [
        float(df["target"].dropna().iloc[0])
        for _, df, _ in runs
        if "target" in df and not df["target"].dropna().empty
    ]
    target = targets[0] if targets else 1.0

    metrics = [
        str(df["metric"].dropna().iloc[0])
        for _, df, _ in runs
        if "metric" in df and not df["metric"].dropna().empty
    ]
    metric = metrics[0] if metrics else "halfwidth68"

    n_refs = []
    for _, df, _ in runs:
        if "n_ref" in df and not df["n_ref"].dropna().empty:
            n_refs.append(int(round(float(df["n_ref"].dropna().iloc[0]))))
        else:
            n_refs.append(int(df["n_total"].min()))

    crossings: list[float] = []
    for _, df, _ in runs:
        crossing = forecast_crossing(df, target=target)
        if crossing is not None:
            crossings.append(crossing)

    all_max_n = max(int(df["n_total"].max()) for _, df, _ in runs)
    if crossings:
        x_max = min(all_max_n, int(1.25 * max(crossings)))
        x_max = max(x_max, int(1.2 * max(n_refs)))
    else:
        x_max = all_max_n
    x_max = 6000

    fig, ax = plt.subplots(figsize=(width2, 0.35 * width2))

    for label, df, color in runs:
        x = df["n_total"].to_numpy(float)
        y16 = df["metric_p16"].to_numpy(float)
        y50 = df["metric_p50"].to_numpy(float)
        y84 = df["metric_p84"].to_numpy(float)

        crossed = df.loc[df["metric_p50"] <= target, "n_total"]
        crossing = None if crossed.empty else int(crossed.iloc[0])

        curve_label = label
        if crossing is not None:
            curve_label = rf"{label} ($N\approx$ {crossing:,.0f})"

        ax.fill_between(x, y16, y84, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y50, color=color, lw=1.6, label=curve_label)

        if crossing is not None:
            ax.axvline(crossing, color=color, ls="-.", lw=1.1, alpha=0.8)

    ax.axhline(
        target,
        ls="--",
        lw=1.5,
        color="k",
        label=rf"$1\sigma$ away from zero",
    )

    for i, n_ref in enumerate(sorted(set(n_refs))):
        ax.axvline(
            n_ref,
            ls=":",
            lw=1.5,
            color="k",
            label=rf"current $N=\,${n_ref}" if i == 0 else None,
        )

    ax.set_xlim(0, x_max)
    
    visible_ymax = target
    for _, df, _ in runs:
        visible = df["n_total"] <= x_max
        if np.any(visible):
            visible_ymax = max(visible_ymax, float(np.nanmax(df.loc[visible, "metric_p84"])))
    ax.set_ylim(0.5, 1.08 * visible_ymax)
    ax.set_yscale('log')

    ax.set_xlabel("Number of GW observations")
    if metric == "halfwidth68":
        ax.set_ylabel(r"68\% half-width on $A$")
    elif metric == "std":
        ax.set_ylabel(r"Standard deviation of $A$")
    else:
        ax.set_ylabel(r"Uncertainty on $A$")

    ax.grid(alpha=0.3)
    ax.legend(frameon=True, framealpha=1)

     # --- Conversion functions: number of events <-> observing time since current catalog ---
    rate_O4 = 180   # events/year
    rate_O5 = 500   # events/year
    
    def N_to_time_O4(N):
        return (np.asarray(N) - n_ref) / rate_O4
    
    def time_to_N_O4(t):
        return n_ref + np.asarray(t) * rate_O4
    
    def N_to_time_O5(N):
        return (np.asarray(N) - n_ref) / rate_O5
    
    def time_to_N_O5(t):
        return n_ref + np.asarray(t) * rate_O5
    
    # First top axis: O4a runtime since current catalog
    secax_O4 = ax.secondary_xaxis("top", functions=(N_to_time_O4, time_to_N_O4))
    secax_O4.set_xlabel("Time from current catalog [yr] at O4a sensitivity", labelpad=5)
    secax_O4.tick_params(axis="x", colors='black')
    secax_O4.xaxis.label.set_color('black')
    
    # Second top axis: O5a runtime since current catalog
    secax_O5 = ax.secondary_xaxis(1.25, functions=(N_to_time_O5, time_to_N_O5))
    secax_O5.set_xlabel("Time from current catalog [yr] at O5a sensitivity", labelpad=5)
    secax_O5.tick_params(axis="x", colors='black')
    secax_O5.xaxis.label.set_color('black')

    path.mkdir(parents=True, exist_ok=True)
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

    forecast_run_joint = results_dir / "forecast_fisher_joint"

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
