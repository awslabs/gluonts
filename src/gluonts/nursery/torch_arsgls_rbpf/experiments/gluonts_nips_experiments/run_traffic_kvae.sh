#!/usr/bin/env bash

EXPERIMENT_NAME="kvae"
DATASET_NAME="traffic_nips"
#GPUS="0 1 2 3"
MAX_RUNS_PARALLEL=4
NUM_RUNS=4

PYTHONPATH=$PYTHONPATH:.:..:../..
for IDX_RUN in $(seq 0 $(($NUM_RUNS-1)));
do
  #python run.py -dataset_name $DATASET_NAME -experiment_name $EXPERIMENT_NAME -IDX_RUN $IDX_RUN -gpus $GPUS;
  python3.6 kvae_run.py -dataset_name $DATASET_NAME -experiment_name $EXPERIMENT_NAME -run_nr $IDX_RUN -gpus  $IDX_RUN &
        pids[${IDX_RUN}]=$!
        if ((($IDX_RUN % ($MAX_RUNS_PARALLEL))==($MAX_RUNS_PARALLEL-1))); then
            echo "$(($IDX_RUN+1)) runs started for experiment: $EXPERIMENT_NAME"
            for pid in ${pids[*]}; do
                wait $pid
            done
        fi
done