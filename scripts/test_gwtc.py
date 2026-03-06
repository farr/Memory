import os
from memory import gw_residuals, gw_memory
from pesummary.io import read as pesummary_read
import lalsimulation as lalsim

for GWTC in sorted(x for x in os.listdir('/mnt/home/ccalvk/ceph/') if 'GWTC' in x):
    for event in sorted(os.listdir(f'/mnt/home/ccalvk/ceph/{GWTC}')):
        if not (event.endswith('.hdf5') or event.endswith('.h5')):
            continue
        filename = f'/mnt/home/ccalvk/ceph/{GWTC}/{event}'
        try:
            data = pesummary_read(filename)
        except Exception as e:
            print(f"Cannot open {filename}: {e}")
            continue
        for key in data.labels:
            # Derive LAL approximant from label (e.g. "C00:NRSur7dq4" or "C00:IMRPhenomNSBH:LowSpin")
            parts = key.split(":")
            if len(parts) < 2:
                print(f"Skipping {event}/{key}: cannot extract approximant from label")
                continue
            approx_name = parts[1]
            try:
                approximant = getattr(lalsim, approx_name)
            except AttributeError:
                print(f"Skipping {event}/{key}: lalsim has no attribute '{approx_name}'")
                continue
            try:
                res = gw_residuals.compute_bbh_residuals_with_spline_calibration(
                    filename,
                    event=event,
                    max_samples=1,
                    label=key,
                    thin=1
                )
                memory_variables_likelihoods_and_weights = gw_memory.make_memories(
                    res['samples'],
                    [
                        {det: res["fd"][det]["residual"][k] for det in res["fd"].keys()}
                        for k in range(len(next(iter(res["fd"].values()))["residual"]))
                    ],
                    res['config'],
                    res['ifos'],
                    approximant=approximant,
                    ell_max=4
                )
            except Exception as e:
                print(f"Error processing {event}/{key}: {e}")
                continue
