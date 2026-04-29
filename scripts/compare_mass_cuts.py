"""Compare the median-m2 cut against the GWTC-4 paper's 1%-quantile cut.

The current implementation in `memory.hierarchical.data.generate_data` keeps an
event when the median of the m2_source posterior exceeds 3 Msun. The GWTC-4
populations paper (https://arxiv.org/html/2508.18083v2#S6) instead keeps an
event when "the 1% lower limit on both component mass posteriors (under the PE
priors) is larger than 3 Msun". This script applies both criteria to the same
PE samples used in a given hierarchical analysis run, so we can see which
events disagree between the two cuts (and how the resulting event count
compares to the paper's 153).

Usage
-----
python scripts/compare_mass_cuts.py \
    --event-files results/prod_20260428/auto_o1o2o3o4a_joint/event_files.txt \
    --ifar-cache  results/prod_20260428/auto_o1o2o3o4a_joint/event_ifars.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

from memory.hierarchical.data import (
    MIN_MASS_2_SOURCE,
    _EXCLUDED_EVENTS,
    _pick_waveform_label,
    _resolve_waveform_label,
    load_event_ifars,
)

LOG = logging.getLogger("compare_mass_cuts")
EVENT_RE = re.compile(r"(GW\d{6}_\d{6})")


def _event_name_from_path(path: str) -> str:
    m = EVENT_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Could not extract event name from: {path}")
    return m.group(1)


def _read_lines(path: Path) -> List[str]:
    with open(path) as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _component_masses(
    pe_path: str, waveform: Optional[str] = None
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Return (waveform_label, m1_source, m2_source) for the chosen waveform."""
    with h5py.File(pe_path, "r") as f:
        groups = [
            k
            for k in f.keys()
            if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]
        ]
        if not groups:
            raise RuntimeError(f"No posterior_samples groups in {pe_path}")
        try:
            label = _resolve_waveform_label(groups, waveform)
        except KeyError:
            label = _pick_waveform_label(groups)
        ps = f[label]["posterior_samples"][()]

    cols = ps.dtype.names or ()
    m1 = None
    for cand in ("mass_1_source", "m1_source", "mass1_source"):
        if cand in cols:
            m1 = np.asarray(ps[cand], dtype=float)
            break
    if m1 is None:
        raise KeyError(f"mass_1_source missing in {pe_path}; have {cols}")

    m2 = None
    for cand in ("mass_2_source", "m2_source", "mass2_source"):
        if cand in cols:
            m2 = np.asarray(ps[cand], dtype=float)
            break
    if m2 is None:
        for cand in ("mass_ratio", "q"):
            if cand in cols:
                q = np.asarray(ps[cand], dtype=float)
                m2 = m1 * q
                break
    if m2 is None:
        raise KeyError(f"Could not derive mass_2_source for {pe_path}; have {cols}")

    return label, m1, m2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--event-files",
        type=Path,
        required=True,
        help="Path to event_files.txt produced by the run",
    )
    p.add_argument(
        "--ifar-cache",
        type=Path,
        default=None,
        help="Path to event_ifars.txt cache (defaults to event_files dir)",
    )
    p.add_argument(
        "--ifar-threshold",
        type=float,
        default=1.0,
        help="IFAR threshold in years (default: 1.0)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=MIN_MASS_2_SOURCE,
        help="Component-mass threshold in Msun (default: 3.0)",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.01,
        help="Lower-quantile used by the paper's cut (default: 0.01)",
    )
    p.add_argument(
        "--waveform",
        type=str,
        default=None,
        help="Override waveform selection (default: auto -- match the analysis)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    event_files = _read_lines(args.event_files)
    if not event_files:
        raise SystemExit(f"No event files in {args.event_files}")

    ifar_cache = args.ifar_cache or args.event_files.parent / "event_ifars.txt"
    names = [_event_name_from_path(p) for p in event_files]
    ifars = load_event_ifars(names, str(ifar_cache) if ifar_cache.exists() else None)

    rows: List[Tuple[str, str, float, float, float, float, str]] = []
    print(
        f"{'event':<22} {'waveform':<35} {'m2_q01':>8} {'m2_med':>8} "
        f"{'m1_q01':>8} {'IFAR(yr)':>10}  status"
    )
    print("-" * 120)
    median_pass: List[str] = []
    paper_pass: List[str] = []
    nan_events: List[str] = []
    for path, name in zip(event_files, names):
        ifar = float(ifars.get(name, float("inf")))
        excluded_hardcoded = name in _EXCLUDED_EVENTS

        try:
            label, m1, m2 = _component_masses(path, args.waveform)
        except Exception as exc:
            print(f"{name:<22} {'<load failed>':<35} {'-':>8} {'-':>8} {'-':>8} "
                  f"{ifar:>10.3g}  ERROR: {exc}")
            continue

        m1_q = float(np.nanquantile(m1, args.quantile))
        m2_q = float(np.nanquantile(m2, args.quantile))
        m2_med = float(np.nanmedian(m2))
        m1_med = float(np.nanmedian(m1))

        status_bits = []
        ifar_ok = ifar >= args.ifar_threshold
        if not ifar_ok:
            status_bits.append("IFAR<thr")
        if excluded_hardcoded:
            status_bits.append("hardcoded-excluded")

        med_keep = ifar_ok and (not excluded_hardcoded) and (
            m2_med >= args.threshold
        )
        paper_keep = ifar_ok and (not excluded_hardcoded) and (
            m1_q > args.threshold and m2_q > args.threshold
        )

        if med_keep and not paper_keep:
            status_bits.append("MED-only")
        elif paper_keep and not med_keep:
            status_bits.append("PAPER-only")
        elif med_keep and paper_keep:
            status_bits.append("both")
        else:
            status_bits.append("neither")

        status = ", ".join(status_bits)

        print(
            f"{name:<22} {label:<35} {m2_q:>8.2f} {m2_med:>8.2f} "
            f"{m1_q:>8.2f} {ifar:>10.3g}  {status}"
        )

        if med_keep:
            median_pass.append(name)
        if paper_keep:
            paper_pass.append(name)
        if not np.isfinite(m2_q) or not np.isfinite(m2_med):
            nan_events.append(name)

        rows.append((name, label, m1_q, m2_q, m1_med, m2_med, ifar, status))

    print()
    print(f"Median-cut keeps:     {len(median_pass)} events")
    print(f"Paper-cut keeps:      {len(paper_pass)} events")

    only_median = sorted(set(median_pass) - set(paper_pass))
    only_paper = sorted(set(paper_pass) - set(median_pass))
    print(f"In median-cut but NOT paper-cut ({len(only_median)}): {only_median}")
    print(f"In paper-cut but NOT median-cut ({len(only_paper)}): {only_paper}")

    if args.out is not None:
        with open(args.out, "w") as fh:
            fh.write("event,waveform,m1_q01,m2_q01,m1_median,m2_median,ifar_yr,status\n")
            for r in rows:
                fh.write(
                    f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},"
                    f"{r[6]:.6g},{r[7]}\n"
                )
        print(f"Wrote per-event diagnostics to {args.out}")


if __name__ == "__main__":
    main()
