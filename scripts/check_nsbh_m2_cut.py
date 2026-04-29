"""Check whether NSBH events are caught by the min_mass_2_source cut.

Mimics the exact logic in generate_data:
    m1_q = quantile(m1_source, MIN_MASS_QUANTILE)
    m2_q = quantile(m1_source * mass_ratio, MIN_MASS_QUANTILE)
    excluded if either m1_q or m2_q is at or below MIN_MASS_2_SOURCE

Fetches posteriors from GWOSC and applies the exact same logic.
"""

import numpy as np
import requests
import h5py
import tempfile
import os

from gwosc.api import fetch_event_json
from memory.hierarchical.data import MIN_MASS_2_SOURCE, MIN_MASS_QUANTILE

NSBH_EVENTS = [
    "GW200105_162426",
    "GW200115_042309",
    "GW230518_125908",
    "GW230529_181500",
]


def get_preferred_pe_url(event_name):
    """Return data_url from the preferred PE parameter set for an event."""
    data = fetch_event_json(event_name)
    events = data.get("events", {})
    # Pick the highest version
    ver = sorted(events.keys())[-1]
    params = events[ver].get("parameters", {})
    preferred = None
    fallback = None
    for key, entry in params.items():
        if entry.get("pipeline_type") != "pe":
            continue
        url = entry.get("data_url", "")
        if not url:
            continue
        if entry.get("is_preferred"):
            preferred = url
            break
        fallback = url
    return preferred or fallback


def load_m1_q_from_url(url):
    """Download PE HDF5 and return (m1_source, mass_ratio) arrays."""
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "r") as f:
            def find_ps(grp, depth=0):
                if depth > 5:
                    return None
                if "posterior_samples" in grp:
                    return grp["posterior_samples"]
                for key in grp:
                    if isinstance(grp[key], h5py.Group):
                        r = find_ps(grp[key], depth + 1)
                        if r is not None:
                            return r
                return None

            ps = find_ps(f)
            if ps is None:
                raise RuntimeError(f"No posterior_samples dataset in {url}")

            cols = [n.lower() for n in ps.dtype.names]

            for cand in ("mass_1_source", "m1_source", "mass1_source"):
                if cand in cols:
                    m1 = np.asarray(ps[cand], dtype=float)
                    break
            else:
                raise KeyError(f"mass_1_source not found; columns: {ps.dtype.names}")

            for cand in ("mass_ratio", "q"):
                if cand in cols:
                    q = np.asarray(ps[cand], dtype=float)
                    break
            else:
                # fall back to computing q from m2
                for cand in ("mass_2_source", "m2_source", "mass2_source"):
                    if cand in cols:
                        q = np.asarray(ps[cand], dtype=float) / m1
                        break
                else:
                    raise KeyError(f"mass_ratio not found; columns: {ps.dtype.names}")

            return m1, q
    finally:
        os.unlink(tmp_path)


qpct = 100.0 * MIN_MASS_QUANTILE
print(f"MIN_MASS_2_SOURCE threshold: {MIN_MASS_2_SOURCE} Msun"
      f" (applied as a {qpct:g}%-quantile cut on both component masses)\n")
print(f"{'Event':<25} {'m1_q [Msun]':>14} {'m2_q [Msun]':>14} {'cut?':>20}")
print("-" * 80)

all_cut = True
for event in NSBH_EVENTS:
    try:
        url = get_preferred_pe_url(event)
        if url is None:
            print(f"{event:<25} {'no PE URL found':>14}")
            continue
        m1, q = load_m1_q_from_url(url)
        m2 = m1 * q
        m1_q = float(np.nanquantile(m1, MIN_MASS_QUANTILE))
        m2_q = float(np.nanquantile(m2, MIN_MASS_QUANTILE))
        cut = (m1_q <= MIN_MASS_2_SOURCE) or (m2_q <= MIN_MASS_2_SOURCE)
        all_cut = all_cut and cut
        verdict = "YES (excluded)" if cut else "NO (passes!)"
        print(f"{event:<25} {m1_q:>14.3f} {m2_q:>14.3f} {verdict:>20}")
    except Exception as e:
        all_cut = False
        print(f"{event:<25} ERROR: {e}")

print()
if all_cut:
    print("All NSBH events are caught by the m2 cut — hard-coded exclusion list is redundant.")
else:
    print("WARNING: at least one NSBH event is NOT caught by the m2 cut!")
