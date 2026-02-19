#!/usr/bin/env python3

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["LAL_DATA_PATH"] = f"{PROJECT_ROOT}/data"
import re
import argparse
import numpy as np
import h5py
import multiprocessing as mp
import matplotlib.pyplot as plt

import lalsimulation as lalsim
from lal import MSUN_SI, PC_SI

from memory.gw_residuals import (
    compute_one_sample_fd,
    compute_bbh_residuals_with_spline_calibration,
)

from memory.gw_memory import (
    make_memories,
    compute_memory_variables_likelihoods_and_weights
)


# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GW memory quantities for GWTC events."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/home/ccalvk/ceph/",
        help="Base directory containing GWTC subdirectories.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where output will be written.",
    )

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ell-max", type=int, default=4)
    parser.add_argument("--thin", type=int, default=1)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories.",
    )

    return parser.parse_args()


# ================================================================
# Utility functions
# ================================================================

def find_gwtc_directories(base_dir):
    base = Path(base_dir)
    return sorted(
        [d for d in base.iterdir() if d.is_dir() and "GWTC" in d.name]
    )


def extract_event_string(filename):
    """
    Extract GWYYYYMMDD_HHMMSS from filename.
    """
    match = re.search(r"(GW\d{6}_\d{6})", filename)
    return match.group(1) if match else None


def find_unique_event_files(gwtc_dir):
    """
    Returns dict: {event_string: filepath}
    If multiple files for same event exist, pick first alphabetically.
    """
    event_files = {}

    for f in sorted(gwtc_dir.glob("*.h5")) + sorted(gwtc_dir.glob("*.hdf5")):
        event = extract_event_string(f.name)
        if event is None:
            continue

        if event not in event_files:
            event_files[event] = f

    return event_files


def get_waveform_labels_from_hdf5(filepath):
    """
    Return all keys starting with 'C'.
    """
    labels = []
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            if key.startswith("C"):
                labels.append(key)
    return labels


def parse_approximant_from_label(label):
    """
    Example: 'C00:NRSur7dq4' -> 'NRSur7dq4'
    """
    return label.split(":")[1]


# ================================================================
# Approximant validation cache
# ================================================================

APPROXIMANT_CACHE = {}


def approximant_has_td_generator(approximant_name):
    """
    Check:
      1. lalsim.<approximant_name> exists
      2. SimInspiralChooseTDModes works for it
    Cache results so we don't re-check.
    """
    if approximant_name in APPROXIMANT_CACHE:
        return APPROXIMANT_CACHE[approximant_name]

    if not hasattr(lalsim, approximant_name):
        APPROXIMANT_CACHE[approximant_name] = False
        return False

    approximant = getattr(lalsim, approximant_name)

    try:
        # Real test call — minimal dummy parameters
        lalsim.SimInspiralChooseTDModes(
            0, 1/4096,
            30 * MSUN_SI,
            30 * MSUN_SI,
            0, 0, 0,
            0, 0, 0,
            16,
            50,
            400 * PC_SI * 1e6,
            None,
            4,
            approximant,
        )
    except:
        APPROXIMANT_CACHE[approximant_name] = False
        return False

    APPROXIMANT_CACHE[approximant_name] = True
    return True


# ================================================================
# Output handling
# ================================================================

def prepare_output_directory(output_dir, event, overwrite=False):
    event_dir = Path(output_dir) / event

    if event_dir.exists() and not overwrite:
        raise RuntimeError(
            f"Output directory {event_dir} already exists. "
            "Use --overwrite to overwrite."
        )

    event_dir.mkdir(parents=True, exist_ok=True)
    return event_dir


def save_results_hdf5(filepath, results_dict):
    """
    results_dict = {
        label: ndarray (n_samples, 5)
    }
    """
    with h5py.File(filepath, "w") as f:
        for label, arr in results_dict.items():
            grp = f.create_group(label)
            grp.create_dataset("A_hat", data=arr[:, 0])
            grp.create_dataset("A_sigma", data=arr[:, 1])
            grp.create_dataset("A_sample", data=arr[:, 2])
            grp.create_dataset("log_weight", data=arr[:, 3])
            grp.create_dataset("log_likelihood", data=arr[:, 4])


def save_histogram(event_dir, label, arr):
    plt.figure()
    plt.hist(arr[:, 2], bins=40, density=True)
    plt.xlabel(r"$A_{\mathrm{memory}}$")
    plt.ylabel("Density")
    plt.title(f"{label} Memory Amplitude Posterior")

    safe_label = label.replace(":", "_")
    outfile = event_dir / f"histogram_{safe_label}.png"
    plt.savefig(outfile)
    plt.close()


# ================================================================
# Main processing
# ================================================================

def process_event(filepath, event, args):
    labels = get_waveform_labels_from_hdf5(filepath)

    results = {}

    for label in labels:
        approximant_name = parse_approximant_from_label(label)

        if not approximant_has_td_generator(approximant_name):
            continue

        approximant = getattr(lalsim, approximant_name)

        res = compute_bbh_residuals_with_spline_calibration(
            str(filepath),
            event=event,
            max_samples=args.max_samples,
            label=label,
            thin=args.thin,
        )

        h_memories_in_det = make_memories(
            res,
            approximant=approximant,
            ell_max=args.ell_max,
        )

        memory_vars = compute_memory_variables_likelihoods_and_weights(
            res,
            h_memories_in_det,
        )

        results[label] = memory_vars

    return results


def process_event_wrapper(task):
    filepath, event, args_dict = task
    
    # Rebuild args namespace (safer for multiprocessing)
    args = argparse.Namespace(**args_dict)

    event_dir = Path(args.output_dir) / event
    if event_dir.exists() and not args.overwrite:
        return

    try:
        results = process_event(filepath, event, args)
    except Exception as e:
        print(f"Error processing event {event}: {e}")
        return None
        
    if len(results) == 0:
        return None

    event_dir = prepare_output_directory(
        args.output_dir,
        event,
        overwrite=args.overwrite,
    )

    h5_path = event_dir / "memory_results.h5"
    save_results_hdf5(h5_path, results)

    for label, arr in results.items():
        save_histogram(event_dir, label, arr)

    return event

# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    gwtc_dirs = find_gwtc_directories(args.base_dir)

    tasks = []
    for gwtc_dir in gwtc_dirs:
        event_files = find_unique_event_files(gwtc_dir)

        for event, filepath in event_files.items():
            tasks.append((str(filepath), event, vars(args)))

    nproc = min(mp.cpu_count() - 1, len(tasks))

    with mp.get_context("spawn").Pool(nproc) as pool:
        pool.map(process_event_wrapper, tasks)


if __name__ == "__main__":
    main()
