#!/bin/bash

python3 examples/article/run_all_experiments.py \
--ncad_dir='~/ncad' \
--data_dir='~/ncad_datasets' \
--hparams_dir='~/ncad/examples/article/hparams' \
--out_dir='~/ncad_output' \
--download_data=True \
--yahoo_path='~/ncad/yahoo_dataset.tgz' \
--number_of_trials=10 \
--run_swat=False \
--run_yahoo=False
