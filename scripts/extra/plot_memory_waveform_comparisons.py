#!/usr/bin/env python3
"""Plot per-event memory-quantity comparisons across waveform labels.

For each event with multiple waveform groups in ``memory_results.h5``, create a
figure with overlaid histograms of ``A_hat``, ``A_sigma``, and ``log_weight``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay waveform-specific histograms of A_hat, A_sigma, and "
            "log_weight for events in a memory analysis directory."
        )
    )
    parser.add_argument(
        "--memory-dir",
        default="/mnt/home/kmitman/work/memory_pop/analysis",
        help="Directory containing per-event memory_results.h5 files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory for comparison plots. Defaults to "
            "results/waveform_comparison_histograms_<memory-dir-name>."
        ),
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=None,
        help="Optional list of event names to plot.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional limit on the number of events processed.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins per panel.",
    )
    return parser.parse_args()


def _load_event_waveforms(memory_path: Path) -> dict[str, dict[str, np.ndarray]]:
    waveforms: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(memory_path, "r") as f:
        for label in sorted(f.keys()):
            grp = f[label]
            waveforms[label] = {
                "A_hat": np.asarray(grp["A_hat"][()]).real,
                "A_sigma": np.asarray(grp["A_sigma"][()]).real,
                "log_weight": np.asarray(grp["log_weight"][()]).real,
            }
    return waveforms


def _select_events(memory_dir: Path, requested_events: list[str] | None) -> list[Path]:
    if requested_events is not None:
        event_dirs = [memory_dir / event for event in requested_events]
    else:
        event_dirs = sorted(p for p in memory_dir.iterdir() if p.is_dir())

    selected: list[Path] = []
    for event_dir in event_dirs:
        memory_path = event_dir / "memory_results.h5"
        if not memory_path.exists():
            continue
        selected.append(event_dir)
    return selected


def _robust_linear_bins(
    arrays: list[np.ndarray], bins: int, low_q: float = 0.5, high_q: float = 99.5
) -> tuple[np.ndarray, list[np.ndarray]]:
    finite_arrays = [arr[np.isfinite(arr)] for arr in arrays]
    finite_arrays = [arr for arr in finite_arrays if arr.size > 0]
    if not finite_arrays:
        edges = np.linspace(-1.0, 1.0, bins + 1)
        return edges, [np.array([]) for _ in arrays]

    combined = np.concatenate(finite_arrays)
    lo = np.nanpercentile(combined, low_q)
    hi = np.nanpercentile(combined, high_q)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = np.nanmin(combined), np.nanmax(combined)
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    clipped = [np.clip(arr[np.isfinite(arr)], lo, hi) for arr in arrays]
    edges = np.linspace(lo, hi, bins + 1)
    return edges, clipped


def _positive_log_bins(
    arrays: list[np.ndarray], bins: int, low_q: float = 0.5, high_q: float = 99.5
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    positive_arrays = [arr[np.isfinite(arr) & (arr > 0)] for arr in arrays]
    positive_arrays = [arr for arr in positive_arrays if arr.size > 0]
    if not positive_arrays:
        return None

    combined = np.concatenate(positive_arrays)
    lo = np.nanpercentile(combined, low_q)
    hi = np.nanpercentile(combined, high_q)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = np.nanmin(combined), np.nanmax(combined)
    lo = max(lo, np.finfo(float).tiny)
    if lo >= hi:
        return None

    clipped = [np.clip(arr[np.isfinite(arr) & (arr > 0)], lo, hi) for arr in arrays]
    edges = np.geomspace(lo, hi, bins + 1)
    return edges, clipped


def _prepare_hist_data(name: str, arrays: list[np.ndarray], bins: int) -> tuple[np.ndarray, list[np.ndarray], str]:
    finite_positive = [
        arr[np.isfinite(arr) & (arr > 0)]
        for arr in arrays
    ]
    combined_positive = np.concatenate([arr for arr in finite_positive if arr.size > 0]) if any(
        arr.size > 0 for arr in finite_positive
    ) else np.array([])

    if (
        name == "A_sigma"
        and combined_positive.size > 0
        and np.nanpercentile(combined_positive, 99.5) / np.nanpercentile(combined_positive, 0.5) > 100
    ):
        result = _positive_log_bins(arrays, bins=bins)
        if result is not None:
            edges, clipped = result
            return edges, clipped, "log"

    edges, clipped = _robust_linear_bins(arrays, bins=bins)
    return edges, clipped, "linear"


def _plot_event(event_name: str, waveforms: dict[str, dict[str, np.ndarray]], outpath: Path, bins: int) -> None:
    quantities = ("A_hat", "A_sigma", "log_weight")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("tab10")

    for q_idx, quantity in enumerate(quantities):
        ax = axes[q_idx]
        arrays = [waveforms[label][quantity] for label in waveforms]
        bin_edges, clipped_arrays, scale = _prepare_hist_data(quantity, arrays, bins=bins)

        for idx, (label, values) in enumerate(zip(waveforms, clipped_arrays, strict=True)):
            if values.size == 0:
                continue
            median = np.nanmedian(values)
            p16, p84 = np.nanpercentile(values, [16, 84])
            ax.hist(
                values,
                bins=bin_edges,
                density=True,
                histtype="step",
                linewidth=1.7,
                color=cmap(idx % 10),
                label=f"{label}  med={median:.3g}  16-84%=[{p16:.3g}, {p84:.3g}]",
            )

        if scale == "log":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(quantity)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)

    axes[0].set_title("A_hat")
    axes[1].set_title("A_sigma")
    axes[2].set_title("log_weight")
    fig.suptitle(f"{event_name}: waveform comparison")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir is not None
        else PROJECT_ROOT / "results" / f"waveform_comparison_histograms_{memory_dir.name}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    event_dirs = _select_events(memory_dir, args.events)
    if args.max_events is not None:
        event_dirs = event_dirs[: args.max_events]

    manifest_rows: list[str] = ["event_name\tn_waveforms\twaveform_labels\tplot_path"]
    plotted = 0
    skipped = 0

    for event_dir in event_dirs:
        memory_path = event_dir / "memory_results.h5"
        waveforms = _load_event_waveforms(memory_path)
        if len(waveforms) <= 1:
            skipped += 1
            continue

        outpath = outdir / f"{event_dir.name}_waveform_comparison.png"
        _plot_event(event_dir.name, waveforms, outpath, bins=args.bins)
        manifest_rows.append(
            f"{event_dir.name}\t{len(waveforms)}\t{' | '.join(waveforms.keys())}\t{outpath}"
        )
        plotted += 1

    manifest_path = outdir / "manifest.tsv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="ascii")

    print(f"Memory directory: {memory_dir}")
    print(f"Output directory: {outdir}")
    print(f"Plotted events with multiple waveform groups: {plotted}")
    print(f"Skipped single-waveform events: {skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
