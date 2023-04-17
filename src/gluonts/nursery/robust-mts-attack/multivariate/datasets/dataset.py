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

from collections import OrderedDict

import numpy as np
import os
import logging
from typing import NamedTuple

import pandas as pd
from pathlib import Path

import tarfile
from gluonts.dataset.common import Dataset, load_datasets
from gluonts.dataset.repository import (
    default_dataset_path,
)


def extract_dataset(dataset_name: str):
    dataset_folder = default_dataset_path / dataset_name

    if os.path.exists(dataset_folder):
        logging.info(f"found local file in {dataset_folder}, skip extracting")
        return

    compressed_data_path = Path("datasets")
    tf = tarfile.open(compressed_data_path / (dataset_name + ".tar.gz"))
    tf.extractall(default_dataset_path)


def pivot_dataset(dataset):
    ds_list = list(dataset)
    return [
        {
            "item": "0",
            "start": ds_list[0]["start"],
            "target": np.vstack([d["target"] for d in ds_list]),
        }
    ]


class MultivariateDatasetInfo(NamedTuple):
    name: str
    train_ds: Dataset
    test_ds: Dataset
    prediction_length: int
    freq: str
    target_dim: int


def make_dataset(
    values: np.ndarray,
    prediction_length: int,
    start: str = "1700-01-01",
    freq: str = "1H",
):
    target_dim = values.shape[0]

    print(
        f"making dataset with {target_dim} dimension and {values.shape[1]} observations."
    )

    start = pd.Timestamp(start, freq)

    train_ds = [
        {"item": "0", "start": start, "target": values[:, :-prediction_length]}
    ]
    test_ds = [{"item": "0", "start": start, "target": values}]

    return MultivariateDatasetInfo(
        name="custom",
        train_ds=train_ds,
        test_ds=test_ds,
        target_dim=target_dim,
        freq=freq,
        prediction_length=prediction_length,
    )


def make_multivariate_dataset(
    dataset_name: str,
    num_test_dates: int,
    prediction_length: int,
    max_target_dim: int = None,
    dataset_benchmark_name: str = None,
):
    """
    :param dataset_name:
    :param num_test_dates:
    :param prediction_length:
    :param max_target_dim:
    :param dataset_benchmark_name: in case the name is different in the repo and in the benchmark, for instance
    'wiki-rolling' and 'wikipedia'
    :return:
    """

    extract_dataset(dataset_name=dataset_name)

    metadata, train_ds, test_ds = load_datasets(
        metadata=default_dataset_path / dataset_name,
        train=default_dataset_path / dataset_name / "train",
        test=default_dataset_path / dataset_name / "test",
    )
    dim = len(train_ds) if max_target_dim is None else max_target_dim
    from multivariate.datasets.grouper import Grouper

    grouper_train = Grouper(max_target_dim=dim)
    grouper_test = Grouper(
        align_data=False, num_test_dates=num_test_dates, max_target_dim=dim
    )
    return MultivariateDatasetInfo(
        dataset_name
        if dataset_benchmark_name is None
        else dataset_benchmark_name,
        grouper_train(train_ds),
        grouper_test(test_ds),
        prediction_length,
        # metadata.time_granularity,
        metadata.freq,
        dim,
    )


def random_periodic(max_target_dim: int = 16, prediction_length: int = 24):
    """

    :param max_target_dim:
    :param prediction_length:
    :param noise_scale: the scale factor for the noise w.r.t the mean of the input, sigma_noise = input_mean*noise_scale
    :param levels: each input timeseries is drawn from normal distributions with mean=l, for each l in levels
    :return:
    """
    num_periods = 100
    levels = np.random.uniform(low=0, high=100, size=(max_target_dim,))
    levels = np.expand_dims(levels, axis=1) * np.ones(
        (max_target_dim, prediction_length)
    )

    noise_level = 1.0
    seasonal_noise = np.random.uniform(
        low=0, high=noise_level, size=(max_target_dim,)
    )

    seasonal_noise = np.expand_dims(seasonal_noise, axis=1) * np.ones(
        (max_target_dim, prediction_length)
    )
    seed_values = np.random.normal(loc=levels, scale=seasonal_noise * levels)

    values = np.hstack([seed_values for _ in range(num_periods)])

    levels = np.hstack([levels for _ in range(num_periods)])
    noise = np.random.normal(
        loc=0 * levels, scale=levels / 5, size=values.shape
    )

    return make_dataset(values + noise, prediction_length)


def electricity(max_target_dim: int = None):
    return make_multivariate_dataset(
        dataset_name="electricity_nips",
        num_test_dates=7,
        prediction_length=24,
        max_target_dim=max_target_dim,
    )


def solar(max_target_dim: int = None):
    return make_multivariate_dataset(
        dataset_name="solar_nips",
        num_test_dates=7,
        prediction_length=24,
        max_target_dim=max_target_dim,
    )


def traffic(max_target_dim: int = None):
    return make_multivariate_dataset(
        dataset_name="traffic_nips",
        num_test_dates=7,
        prediction_length=24,
        max_target_dim=max_target_dim,
    )


def wiki(max_target_dim: int = None):
    return make_multivariate_dataset(
        dataset_name="wiki-rolling_nips",
        dataset_benchmark_name="wikipedia",
        num_test_dates=5,
        prediction_length=30,
        # we dont use 9K timeseries due to OOM issues
        max_target_dim=2000 if max_target_dim is None else max_target_dim,
    )


def exchange_rate(max_target_dim: int = None):
    return make_multivariate_dataset(
        dataset_name="exchange_rate_nips",
        dataset_benchmark_name="exchange_rate_nips",
        num_test_dates=5,
        prediction_length=30,
        max_target_dim=max_target_dim,
    )


def taxi_30min(max_target_dim: int = None):
    """
    Taxi dataset limited to the most active area, with lower and upper bound:
        lb = [ 40.71, -74.01]
        ub = [ 40.8 , -73.95]
    :param max_target_dim:
    :return:
    """
    return make_multivariate_dataset(
        dataset_name="taxi_30min",
        dataset_benchmark_name="taxi_30min",
        # The dataset corresponds to the taxi dataset used in this reference:
        # https://arxiv.org/abs/1910.03002 but only contains 56 evaluation
        # windows. The last evaluation window was removed because there was an
        # overlap of five time steps in the last and the penultimate
        # evaluation window.
        num_test_dates=56,
        prediction_length=24,
        max_target_dim=max_target_dim,
    )


datasets = OrderedDict(
    [
        ("solar", solar),
        ("exchange_rate", exchange_rate),
        ("electricity", electricity),
        ("traffic", traffic),
        ("wikipedia", wiki),
        ("taxi_30min", taxi_30min),
    ]
)

if __name__ == "__main__":
    extract_dataset("electricity_nips")
