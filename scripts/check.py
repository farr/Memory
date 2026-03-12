import os
import h5py

events = {}
for catalog in sorted([x for x in os.listdir('/mnt/home/ccalvk/ceph/') if 'GWTC' in x]):
    for event in sorted(os.listdir(f'/mnt/home/ccalvk/ceph/{catalog}')):
        if not '.h5' in event and not '.hdf5' in event:
            continue

        with h5py.File(f'/mnt/home/ccalvk/ceph/{catalog}/{event}') as input_file:
            if not any(["C" in x and ":" in x for x in input_file.keys()]):
                continue

            event_name = "GW" + event.split("GWTC")[1].split("-GW")[-1]
            event_name = (event_name.split("_")[0] + "_" + event.split("GWTC")[1].split("-GW")[1].split("_")[1]).split("-")[0]
            events[event_name] = [key.split(":")[1] for key in input_file.keys() if not 'Mixed' in key and "C" in key and ":" in key]

failure_count = 0
specific_failure_count = 0
complete_failure_count = 0
OUTDIR = "results/memory_gwtc"
for event in sorted(os.listdir(OUTDIR)):
    count = 0
    keys = []
    for x in sorted(os.listdir(f"{OUTDIR}/{event}")):
        if not "histogram" in x:
            continue
        key = x.split("_")[-1].split(".png")[0]
        keys.append(key)
        if any([key in approx for approx in events[event]]):
            count += 1

    print(event)
    if count == len(events[event_name]):
        print("pass!")
    else:
        failure_count += 1
        specific_failure_count += len(events[event_name]) - count
        print(events[event_name])
        print(keys)
        if count == 0:
            complete_failure_count += 1
    print()
print("N failures:", failure_count)
print("N specific failures:", specific_failure_count)
print("N complete failures:", complete_failure_count)
