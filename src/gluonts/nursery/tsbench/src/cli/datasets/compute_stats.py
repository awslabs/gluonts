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

import json
from pathlib import Path
from typing import Optional
import click
from gluonts.dataset.stat import calculate_dataset_statistics
from tqdm.auto import tqdm
from tsbench.config import DATASET_REGISTRY
from tsbench.constants import DEFAULT_DATA_PATH, DEFAULT_DATA_STATS_PATH
from ._main import datasets


@datasets.command(short_help="Compute basic dataset statistics.")
@click.option(
    "--dataset",
    type=str,
    default=None,
    help=(
        "The dataset to compute basic statistics for. "
        "If not provided, computes statistics for all datasets."
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
    nargs=1,
    default=DEFAULT_DATA_STATS_PATH,
    show_default=True,
    help=(
        "The path where the dataset statistics are written to. "
        "Statistics are written as `<dataset>.json` files."
    ),
)
def compute_stats(dataset: Optional[str], data_path: str, output_path: str):
    """
    Computes simple dataset features either for a single dataset or all
    datasets in the registry.
    """
    source = Path(data_path)
    target = Path(output_path)
    target.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset_list = list(DATASET_REGISTRY.items())
    else:
        dataset_list = [(dataset, DATASET_REGISTRY[dataset])]

    for dataset_name, config in tqdm(dataset_list):
        file = target / f"{dataset_name}.json"
        if file.exists():
            continue

        stats = calculate_dataset_statistics(
            config(source).data.train(val=False).gluonts()
        )
        with file.open("w+") as f:
            json.dump(
                {
                    "integer_dataset": stats.integer_dataset,
                    "mean_target_length": stats.mean_target_length,
                    "num_time_observations": stats.num_time_observations,
                    "num_time_series": stats.num_time_series,
                },
                f,
            )
