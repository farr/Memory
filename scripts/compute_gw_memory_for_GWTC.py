#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
import lalsimulation.gwsignal as gwsignal
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
        "--base-dirs",
        type=list,
        default=["/mnt/home/ccalvk/ceph/"], #,"/mnt/home/misi/ceph/rp.04/catalogs/GWTC-5/GWTC5-Draft_Release-6/15d62fc1_17/bbh_only"],
        help="Base directories containing GWTC subdirectories.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where output will be written.",
    )

    parser.add_argument(
        "--multiprocess_events",
        type=bool,
        help="Whether or not to parallelize over events.",
        default=False
    )

    parser.add_argument(
        "--multiprocess_samples",
        type=bool,
        help="Whether or not to parallelize over samples.",
        default=False
    )
    
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ell-max", type=int, default=4)
    parser.add_argument("--thin", type=int, default=1)

    parser.add_argument(
        "--frame-dir",
        type=str,
        default=None,
        help=(
            "Directory containing local BayesWave GWF frame files "
            "(e.g. downloaded from Zenodo 16857060). For any detector whose "
            "event time range is covered by a file in this directory, the "
            "glitch-subtracted strain channel is used instead of GWOSC open "
            "data. Falls back to GWOSC for uncovered detectors."
        ),
    )
    parser.add_argument(
        "--glitch-channel-format",
        type=str,
        default="{ifo}:DCS-CALIB_STRAIN_CLEAN_C00",
        help=(
            "Python format string for the glitch-subtracted channel inside "
            "the GWF frame files; {ifo} is substituted with the detector "
            "name. Default: '{ifo}:DCS-CALIB_STRAIN_CLEAN_C00'."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories.",
    )

    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Only process these events (e.g. GW230606_004305). "
            "If not given, all events in base-dir are processed."
        ),
    )

    return parser.parse_args()


# ================================================================
# Utility functions
# ================================================================

def find_gwtc_directories(base_dirs):
    gwtc_dirs = []
    for base_dir in base_dirs:
        base = Path(base_dir)
        if not "misi" in base_dir:
            for gwtc_dir in sorted(
                [d for d in base.iterdir() if d.is_dir() and "GWTC" in d.name]
            ):
                gwtc_dirs.append(gwtc_dir)
        else:
            gwtc_dirs.append(base)
            
    return gwtc_dirs


def extract_event_string(filename):
    """
    Extract GWYYYYMMDD_HHMMSS from filename.
    """
    match = re.search(r"(GW\d{6}_\d{6})", filename)
    return match.group(1) if match else None


from collections import defaultdict

def find_unique_event_files(gwtc_dir):
    """
    Returns dict: {event_string: filepath}

    For each event:
        - Prefer *_nocosmo* file if present
        - Otherwise use first file alphabetically
    """
    grouped = defaultdict(list)

    # Collect all candidate files
    files = sorted(gwtc_dir.glob("*.h5")) + sorted(gwtc_dir.glob("*.hdf5"))

    for f in files:
        event = extract_event_string(f.name)
        if event is None:
            continue
        grouped[event].append(f)

    event_files = {}

    for event, flist in grouped.items():
        # Prefer nocosmo if present
        nocosmo_files = [f for f in flist if "nocosmo" in f.name]

        if nocosmo_files:
            event_files[event] = sorted(nocosmo_files)[0]
        else:
            # fallback: first alphabetically
            event_files[event] = sorted(flist)[0]

    return event_files


def get_waveform_labels_from_hdf5(filepath, start_string='C'):
    """
    Return all keys starting with a string.
    """
    labels = []
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            if key.startswith(start_string):
                labels.append(key)
    return labels


def parse_approximant_from_label(label, split_char=':'):
    """
    Example: 'C00:NRSur7dq4' -> 'NRSur7dq4'
    """
    return label.split(split_char)[1]


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
            20,
            20,
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
    a_sample = arr[:, 2]
    finite = a_sample[np.isfinite(a_sample)]
    plt.figure()
    if len(finite) > 0:
        plt.hist(finite, bins=40, density=True)
    else:
        plt.text(0.5, 0.5, "No finite samples", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.xlabel(r"$A_{\mathrm{memory}}$")
    plt.ylabel("Density")
    plt.title(f"{label} Memory Amplitude Posterior")

    safe_label = label.replace(":", "_")
    outfile = event_dir / f"histogram_{safe_label}.png"
    plt.savefig(outfile)
    plt.close()


def update_results_hdf5(filepath, label, arr):
    """Create/append one label into the HDF5 file, replacing it if it exists."""
    with h5py.File(filepath, "a") as f:
        if label in f:
            del f[label]
        grp = f.create_group(label)
        grp.create_dataset("A_hat", data=arr[:, 0])
        grp.create_dataset("A_sigma", data=arr[:, 1])
        grp.create_dataset("A_sample", data=arr[:, 2])
        grp.create_dataset("log_weight", data=arr[:, 3])
        grp.create_dataset("log_likelihood", data=arr[:, 4])


def expected_histogram_path(event_dir: Path, label: str) -> Path:
    safe_label = label.replace(":", "_")
    return event_dir / f"histogram_{safe_label}.png"


def label_results_complete_in_h5(h5_path: Path, label: str) -> bool:
    """
    True iff memory_results.h5 contains the label group and all expected datasets
    have the same length (and non-zero length).
    """
    if not h5_path.exists():
        return False

    required = ["A_hat", "A_sigma", "A_sample", "log_weight", "log_likelihood"]
    try:
        with h5py.File(h5_path, "r") as f:
            if label not in f:
                return False
            grp = f[label]
            if not all(k in grp for k in required):
                return False
            lens = [grp[k].shape[0] for k in required]
            return (min(lens) > 0) and (len(set(lens)) == 1)
    except OSError:
        # Corrupt/unreadable file => treat as incomplete
        return False


def label_is_finished(event_dir: Path, label: str) -> bool:
    h5_path = event_dir / "memory_results.h5"
    return label_results_complete_in_h5(h5_path, label) and expected_histogram_path(event_dir, label).exists()


# ================================================================
# Approximant resolution
# ================================================================

def _resolve_approximant(event, approximant_name):
    """
    Resolve an approximant name to either a LAL integer enum or a gwsignal
    CompactBinaryCoalescenceGenerator.  Returns None if not resolvable.

    Resolution order:
      1. Native LAL integer (SimInspiralGetApproximantFromString)
      2. gwsignal generator (handles pyseobnr models like SEOBNRv5PHM)
      3. Strip a trailing '-SpinTaylor' suffix and retry both paths
         (GWTC-4 labels 'IMRPhenomXPHM-SpinTaylor' map to 'IMRPhenomXPHM')
    """
    candidates = [approximant_name]
    if "-" in approximant_name:
        candidates.append(approximant_name.split("-")[0])

    for name in candidates:
        # LAL integer path
        try:
            return lalsim.SimInspiralGetApproximantFromString(name)
        except Exception:
            pass
        # gwsignal path (pyseobnr etc.)
        try:
            gen = gwsignal.gwsignal_get_waveform_generator(name)
            if gen is not None:
                return gen
        except Exception:
            pass

    print(f"[{event}] skipping {approximant_name}: not resolvable via LAL or gwsignal.", flush=True)
    return None


# ================================================================
# Main processing
# ================================================================

def process_event(filepath, event, args, event_dir, multiprocess):
    """
    Process all labels for a single event.

    NEW BEHAVIOR:
      - After each label finishes, immediately:
          * append/replace that label into memory_results.h5
          * write that label's histogram plot
    """
    if not "misi" in filepath:
        if not "kmitman" in filepath or "GW190521" in filepath:
            labels = get_waveform_labels_from_hdf5(filepath)
        else:
            if event != "GW250114_082203":
                labels = ['Bilby:NRSur7dq4']
            else:
                if "NRSur7dq4" in filepath:
                    labels = ["bilby-NRSur7dq4_prod-reweighted"]
                elif "PhenomXO4a" in filepath:
                    labels = ['bilby-IMRPhenomXO4a_prod-reweighted']
                elif "PhenomXPHM" in filepath:
                    labels = ['bilby-IMRPhenomXPHM-SpinTaylor_prod-reweighted']
                elif "SEOBNRv5PHM" in filepath:
                    labels = ['bilby-SEOBNRv5PHM-reweighted']
    else:
        labels = get_waveform_labels_from_hdf5(filepath, "bilby-")
        
    results = {}
    h5_path = Path(event_dir) / "memory_results.h5"

    for label in labels:
        if "Mixed" in label:
            continue

        if not "misi" in filepath:
            if not "kmitman" in filepath or "GW190521" in filepath:
                approximant_name = parse_approximant_from_label(label)
            else:
                if event != "GW250114_082203":
                    approximant_name = "NRSur7dq4"
                else:
                    if "NRSur7dq4" in filepath:
                        approximant_name = "NRSur7dq4"
                    elif "PhenomXO4a" in filepath:
                        approximant_name = "IMRPhenomXO4a"
                    elif "PhenomXPHM" in filepath:
                        approximant_name = "IMRPhenomXPHM"
                    elif "SEOBNRv5PHM" in filepath:
                        approximant_name = "SEOBNRv5PHM"
        else:
            approximant_name = parse_approximant_from_label(label, "bilby-").split("-SpinTaylor")[0]
            
        if (not args.overwrite) and label_is_finished(Path(event_dir), label):
            print(f"[{event}] skipping finished model {approximant_name}.", flush=True)
            continue

        print(f"[{event}] working with model {approximant_name}.", flush=True)

        approximant = _resolve_approximant(event, approximant_name)
        if approximant is None:
            continue

        try:
            res = compute_bbh_residuals_with_spline_calibration(
                str(filepath),
                event=event,
                max_samples=args.max_samples,
                label=label,
                thin=args.thin,
                frame_dir=args.frame_dir,
                glitch_channel_format=args.glitch_channel_format,
            )

            print(f"[{event}] making memories!", flush=True)

            memory_vars = make_memories(
                res['samples'],
                [
                    {det: res["fd"][det]["residual"][k] for det in res["fd"].keys()}
                    for k in range(len(next(iter(res["fd"].values()))["residual"]))
                ],
                res['config'],
                res['ifos'],
                approximant=approximant,
                ell_max=args.ell_max,
                multiprocess=multiprocess,
            )

            results[label] = memory_vars

            update_results_hdf5(h5_path, label, memory_vars)
            save_histogram(event_dir, label, memory_vars)
        except Exception as e:
            print(f"[{event}] failed for {approximant_name}: {e}", flush=True)

    return results


def process_event_wrapper(task, multiprocess=False):
    """
    Wrapper used by both serial and multiprocessing event loops.
    """
    filepath, event, args_dict = task

    # Rebuild args namespace (safer for multiprocessing)
    args = argparse.Namespace(**args_dict)

    event_dir = Path(args.output_dir) / event
    if event_dir.exists() and args.overwrite:
        # optional: wipe only the per-event outputs you generate
        h5p = event_dir / "memory_results.h5"
        if h5p.exists():
            h5p.unlink()
        for p in event_dir.glob("histogram_*.png"):
            p.unlink()
    
    event_dir = prepare_output_directory(args.output_dir, event, overwrite=args.overwrite)

    try:
        results = process_event(filepath, event, args, event_dir, multiprocess)
    except Exception as e:
        print(f"Error processing event {event}: {e}")
        return None

    if results is None or len(results) == 0:
        # No successful labels; keep directory (or remove if you prefer)
        return None

    return event
    

# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    gwtc_dirs = find_gwtc_directories(args.base_dirs)

    tasks = []
    for gwtc_dir in gwtc_dirs:
        event_files = find_unique_event_files(gwtc_dir)
        
        for event, filepath in event_files.items():
            if args.events is not None and event not in args.events:
                continue
            
            tasks.append((str(filepath), event, vars(args)))

    if "GW150914_095045" in args.events:
        tasks.append(("/mnt/home/kmitman/work/memory_pop/data/GW150914/GW150914_095045_NRSur7dq4.h5", "GW150914_095045", vars(args)))

    if "GW190521_074359" in args.events:
        tasks.append(("/mnt/home/kmitman/work/memory_pop/data/GW190521/GW190521_074359.h5", "GW190521_074359", vars(args)))
    
    if "GW250114_082203" in args.events:
        tasks.append(("/mnt/home/kmitman/work/memory_pop/data/GW250114/posterior_samples_PhenomXO4a.h5", "GW250114_082203", vars(args)))
        tasks.append(("/mnt/home/kmitman/work/memory_pop/data/GW250114/posterior_samples_PhenomXPHM.h5", "GW250114_082203", vars(args)))
        tasks.append(("/mnt/home/kmitman/work/memory_pop/data/GW250114/posterior_samples_SEOBNRv5PHM.h5", "GW250114_082203", vars(args)))
        tasks.append(("/mnt/home/kmitman/work/memory_pop/data/GW250114/posterior_samples_NRSur7dq4.h5", "GW250114_082203", vars(args)))
            
    nproc = min(mp.cpu_count() - 1, len(tasks))

    if args.multiprocess_events:
        with mp.get_context("spawn").Pool(nproc) as pool:
            pool.map(process_event_wrapper, tasks)
    else:
        for task in tasks:
            process_event_wrapper(task, args.multiprocess_samples)

    print("Done!", flush=True)

if __name__ == "__main__":
    main()
