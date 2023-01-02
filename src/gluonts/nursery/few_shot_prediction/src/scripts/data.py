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
import itertools
import yaml
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import click
import torch

from meta.datasets import get_data_module, DATASETS_FILTERED
from meta.datasets.cheat import CheatMetaData
from meta.common.torch import tensor_to_np
import pandas as pd


@click.group()
def main():
    return


@main.command()
@click.option(
    "--catch22",
    default=False,
    help="Compute catch22 features and nearest neighbors. Can be time consuming!",
)
def real(catch22: bool):
    """
    Download and process real-world datasets specified in 'DATASETS_FILTERED'.

    Parameters
    ----------
    catch22: bool
        If true, catch22 features for each time series and 100 nearest neighbors w.r.t. l2 distance in
        feature space are computed.
    """
    for dataset in tqdm(DATASETS_FILTERED):
        dm = get_data_module(
            name="dm_" + dataset,
            dataset_name=dataset,
            context_length_multiple=4,
            support_length_multiple=4,
            support_set_size=1,
            catch22_train=catch22,  # compute catch22 features and nearest neighbors
        )
        dm.setup()


@main.command()
@click.argument("config", type=click.Path(exists=True), nargs=1)
def artificial(
    config: str,
):
    """
    Generate artificial datasets specified in a config file.

    Parameters
    ----------
    config: str
        Path to config file.
    """
    with Path(config).open() as f:
        options = yaml.safe_load(f)

    assert len(options["dm_name"]) == 1, len(options["dm_name"])
    dm_name = options["dm_name"][0]
    prefix = f"{dm_name}_"

    data_configs = {}
    for key in options:
        if key in CheatMetaData.__annotations__.keys():
            data_configs[key] = options[key]
        elif key.startswith(prefix):
            data_configs[key[len(prefix) :]] = options[key]

    values = list(itertools.product(*data_configs.values()))
    all_configurations = [dict(zip(data_configs.keys(), v)) for v in values]

    for config in all_configurations:
        dm = get_data_module(
            name=dm_name,
            **config,
        )
        dm.setup()


@main.command()
def statistics():
    """Compute statistics of real-world datasets."""

    path_to_data = Path.home() / ".mxnet" / "gluon-ts" / "datasets"
    path_save_plots = Path.home() / "data" / "plots"

    datasets = DATASETS_FILTERED

    dataset_stats = {}
    for dataset in tqdm(datasets):
        dm = get_data_module(
            name="dm_" + dataset,
            dataset_name=dataset,
            context_length_multiple=4,
            support_length_multiple=4,
            support_set_size=1,
        )
        dm.setup()
        train_file_size = os.path.getsize(
            path_to_data / dataset / "train" / "data.json"
        )

        dataset_stats[dataset] = {
            "length": len(dm.splits.train().data()),
            "n_total_observations": dm.splits.train()
            .data()
            .number_of_time_steps,
            "freq": dm.meta.freq,
            "train_file_size": train_file_size,
            "prediction_length": dm.meta.prediction_length,
        }

    lengths = np.asarray([dataset_stats[d]["length"] for d in datasets])
    lengths_normalized = lengths / sum(lengths)

    n_total = np.asarray(
        [dataset_stats[d]["n_total_observations"] for d in datasets]
    )
    n_total_normalized = n_total / sum(n_total)

    datasets, lengths_normalized, n_total_normalized = zip(
        *(
            sorted(
                list(zip(datasets, lengths_normalized, n_total_normalized)),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    )

    x = np.arange(len(datasets))

    # -------- plot without logarithmic y-axis ------
    fig, ax = plt.subplots(figsize=[25, 8])
    width = 0.35  # the width of the bars
    rects1 = ax.bar(
        x - width / 2, lengths_normalized, width, label="number of ts"
    )
    rects2 = ax.bar(
        x + width / 2,
        n_total_normalized,
        width,
        label="total number of time steps",
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("normalized size")
    ax.set_title("Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path_save_plots / "datasets.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    # -------- plot with logarithmic y-axis ------

    fig, ax = plt.subplots(figsize=[25, 8])
    width = 0.35  # the width of the bars
    rects1 = ax.bar(
        x - width / 2, lengths_normalized, width, label="number of ts"
    )
    rects2 = ax.bar(
        x + width / 2,
        n_total_normalized,
        width,
        label="total number of time steps",
    )
    ax.set_yscale("log")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("normalized size")
    ax.set_title("Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path_save_plots / "datasets_log_scale.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    # ----- print some statistics ---------

    print(f"total number of time series: {sum(lengths)}")
    print(f"total number of time observations: {sum(n_total)}")

    f_size = sum([dataset_stats[d]["train_file_size"] for d in datasets])
    print(f"training set memory size: {f_size / float(1<<30)} GB")

    prediction_lengths = np.asarray(
        [dataset_stats[d]["prediction_length"] for d in datasets]
    )
    print(f"max prediction length is {np.max(prediction_lengths)}")


if __name__ == "__main__":
    main()
