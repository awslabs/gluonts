# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import sys
from pathlib import Path, PosixPath
import argparse

import json

import numpy as np

import torch
from pytorch_lightning.utilities.parsing import str_to_bool

import ncad

# python3 examples/article/run_all_experiments.py \
# --ncad_dir='~/ncad/01_code/ncad' \
# --data_dir='~/ncad/02_data' \
# --hparams_dir='~/ncad/01_code/ncad/examples/article/hparams' \
# --out_dir='~/ncad/04_output/article' \
# --download_data=False \
# --yahoo_path=None


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ncad_dir", type=PosixPath)
    parser.add_argument("--data_dir", type=PosixPath)
    parser.add_argument("--hparams_dir", type=PosixPath)
    parser.add_argument("--out_dir", type=PosixPath)
    parser.add_argument("--download_data", type=str_to_bool, default=False)
    parser.add_argument("--yahoo_path", type=PosixPath, default="~")
    parser.add_argument("--number_of_trials", type=int, default=10)
    parser.add_argument("--run_swat", type=str_to_bool, default=True)
    parser.add_argument("--run_yahoo", type=str_to_bool, default=True)
    args, _ = parser.parse_known_args()
    # args = parser.parse_args()

    return args


def main(
    ncad_dir,
    data_dir,
    hparams_dir,
    out_dir,
    download_data,
    yahoo_path=None,
    number_of_trials=10,
    run_swat=True,
    run_yahoo=True,
):

    # ncad_dir = Path.home()/'ncad/01_code/ncad'
    # data_dir = Path.home()/'ncad/02_data'
    # hparams_dir = Path.home()/'ncad/01_code/ncad/examples/article/hparams'
    # out_dir = Path.home()/'ncad/04_output/article'
    # download_data = False
    # yahoo_path = Path.home()/'ncad/yahoo_dataset.tgz'

    ncad_dir = ncad_dir.expanduser()
    data_dir = data_dir.expanduser()
    hparams_dir = hparams_dir.expanduser()
    out_dir = out_dir.expanduser()

    sys.path.append(str(ncad_dir / "examples"))

    # Benchmark datasets to consider
    benchmarks = ["kpi", "nasa", "smd", "swat", "yahoo"]
    if not run_swat:
        benchmarks.remove("swat")
    if not run_yahoo:
        benchmarks.remove("yahoo")

    # Import pipelines
    for bmk in benchmarks:
        exec(f"from {bmk} import {bmk}_pipeline")

    if download_data:
        yahoo_path = yahoo_path.expanduser()
        ncad.datasets.download(
            data_dir=data_dir,
            benchmarks=benchmarks,
            yahoo_path=yahoo_path,
        )

    # Hyperparameter configurations
    hparams_files = [file for file in os.listdir(hparams_dir) if (file.endswith(".json"))]
    hparams_files.sort()
    # Keep only some hyperparameters
    # hparams_files = hparams_files[:2]
    # hparams_files = ['smd-01.json']

    for file in hparams_files:
        # file=hparams_files[-1]
        if not any([file.startswith(bmk) for bmk in benchmarks]):
            continue

        print(f"\n Executing hparams: \n {file} \n")
        with open(hparams_dir / file, "r") as f:
            hparams = json.load(f)

        for trail_i in range(number_of_trials):
            # Identify corresponding benckmark dataset
            bmk = benchmarks[np.where([file.startswith(bmk) for bmk in benchmarks])[0][0]]

            # Modify hyperparameters
            hparams.update(
                dict(
                    data_dir=data_dir / bmk,
                    model_dir=out_dir / bmk,
                    log_dir=out_dir / bmk,
                    gpus=1 if torch.cuda.is_available() else 0,
                    evaluation_result_path=out_dir / bmk / file.replace(".json", "_results.json"),
                    # epochs = 1,
                )
            )

            eval(f"{bmk}_pipeline")(**hparams)


if __name__ == "__main__":

    args = parse_arguments()
    args_dict = vars(args)  # arguments as dictionary
    main(**args_dict)
