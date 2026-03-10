#!/bin/bash

EVENT_FILE="scripts/events.txt"

if [ ! -f "$EVENT_FILE" ]; then
    echo "Error: $EVENT_FILE not found."
    exit 1
fi

N=$(wc -l < "$EVENT_FILE")

if [ "$N" -eq 0 ]; then
    echo "Error: $EVENT_FILE is empty."
    exit 1
fi

echo "Submitting $N events as Slurm array..."

sbatch --array=0-$((N-1)) scripts/submit_script.sh
