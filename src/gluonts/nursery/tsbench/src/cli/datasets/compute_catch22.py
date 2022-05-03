# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from pathlib import Path
from typing import Any, Dict, Optional
import catch22
import click
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from tsbench.config import DATASET_REGISTRY
from tsbench.constants import DEFAULT_DATA_CATCH22_PATH, DEFAULT_DATA_PATH
from ._main import datasets


@datasets.command(short_help="Compute catch22 features.")
@click.option(
    "--dataset",
    type=str,
    default=None,
    help=(
        "The dataset to compute the catch22 features for. "
        "If not provided, computes features for all datasets."
    ),
)
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="The local path where datasets are stored.",
)
@click.option(
    "--output_path",
    type=click.Path(),
    default=DEFAULT_DATA_CATCH22_PATH,
    show_default=True,
    help=(
        "The path where catch22 features are written to. "
        "Features are written as `<dataset>.parquet` files."
    ),
)
def compute_catch22(dataset: Optional[str], data_path: str, output_path: str):
    """
    Computes the catch22 features for each time series in a dataset.

    Computations are either run for a single dataset or all datasets in the
    registry.
    """
    target = Path(data_path)
    target.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset_names = [(k, v(target)) for k, v in DATASET_REGISTRY.items()]
    else:
        dataset_names = [(dataset, DATASET_REGISTRY[dataset](target))]

    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)

    for dataset_name, config in tqdm(
        dataset_names, disable=len(dataset_names) == 1
    ):
        file = directory / f"{dataset_name}.parquet"
        if file.exists():
            continue

        ts_features = process_map(
            _get_features,
            config.data.train(
                val=False
            ).gluonts(),  # Get features on train set
            max_workers=os.cpu_count(),
            desc=dataset_name,
            chunksize=1,
        )
        df = pd.DataFrame(ts_features)
        df.to_parquet(file)  # type: ignore


def _get_features(ts: Dict[str, Any]) -> Dict[str, Any]:
    features = catch22.catch22_all(ts["target"])
    return dict(zip(features["names"], features["values"]))
