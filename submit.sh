#!/bin/bash
TASKFILE=${1:-taskfiles/TaskFileMemory}
sbatch -p gpu -n 1 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=closest -t 0-6 disBatch "$TASKFILE"
