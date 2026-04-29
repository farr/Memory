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

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from corner import corner


TGR_VARS = ("mu_tgr", "sigma_tgr")
SIGMA_LEVELS_2D = tuple(1.0 - np.exp(-0.5 * np.arange(1, 4) ** 2))
REPO_DIR = Path(__file__).resolve().parent.parent

# Paper data provenance: update this checked-in path when refreshing results.
RESULTS_DIR = Path("results") / "prod_20260428"
DEFAULT_MACROS_OUTPUT = Path("paper") / "results_macros.tex"
DEFAULT_PLOT_OUTPUT = Path("figures") / "tgr_comparison_corner.pdf"


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


def load_tgr_samples(nc_file: Path) -> np.ndarray:
    """Return a flat ``(sample, 2)`` array for ``mu_tgr`` and ``sigma_tgr``."""

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
    joint_summary = (
        summarize(load_tgr_samples(joint_run.nc_file)) if joint_run is not None else None
    )
    primary_summary = joint_summary or memory_summary

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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def make_tgr_corner(
    path: Path,
    memory_run: ResultRun,
    joint_run: ResultRun | None,
    dpi: int,
) -> None:
    plot_runs = [(memory_run, "memory", "C0")]
    if joint_run is not None:
        plot_runs.append((joint_run, "joint", "C1"))

    fig = None
    for run, label, color in plot_runs:
        samples = load_tgr_samples(run.nc_file)
        fig = corner(
            samples,
            labels=[r"$\mu_{\Lambda}$", r"$\sigma_{\Lambda}$"],
            color=color,
            fig=fig,
            figsize=(6, 6),
            levels=SIGMA_LEVELS_2D,
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            fill_contours=False,
            hist_kwargs={"density": True, "linewidth": 1.5},
            contour_kwargs={"linewidths": 1.2},
            truths=[1.0, 0.0],
            truth_color="k",
            label_kwargs={"fontsize": 13},
        )
        axes = np.asarray(fig.axes).reshape(2, 2)
        axes[0, 0].plot([], [], color=color, label=label)

    assert fig is not None
    axes = np.asarray(fig.axes).reshape(2, 2)
    axes[0, 0].legend(loc="upper right", fontsize=10, framealpha=0.8)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_xlim(left=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
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
        make_tgr_corner(
            plot_output,
            memory_run=memory_run,
            joint_run=joint_run,
            dpi=args.dpi,
        )
        print(f"Wrote plot:   {plot_output}")


if __name__ == "__main__":
    main()
