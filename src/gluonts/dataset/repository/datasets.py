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

import logging
from collections import OrderedDict
from functools import partial
from pathlib import Path

from gluonts.dataset.artificial import ConstantDataset
from gluonts.dataset.common import TrainDatasets, load_datasets
from gluonts.dataset.repository._artificial import generate_artificial_dataset
from gluonts.dataset.repository._gp_copula_2019 import (
    generate_gp_copula_dataset,
)
from gluonts.dataset.repository._lstnet import generate_lstnet_dataset
from gluonts.dataset.repository._m3 import generate_m3_dataset
from gluonts.dataset.repository._m4 import generate_m4_dataset
from gluonts.dataset.repository._m5 import generate_m5_dataset
from gluonts.support.util import get_download_path

m4_freq = "Hourly"
pandas_freq = "H"
dataset_path = Path(f"m4-{m4_freq}")
prediction_length = 48

dataset_recipes = OrderedDict(
    {
        # each recipe generates a dataset given a path
        "constant": partial(
            generate_artificial_dataset, dataset=ConstantDataset()
        ),
        "exchange_rate": partial(
            generate_lstnet_dataset, dataset_name="exchange_rate"
        ),
        "solar-energy": partial(
            generate_lstnet_dataset, dataset_name="solar-energy"
        ),
        "electricity": partial(
            generate_lstnet_dataset, dataset_name="electricity"
        ),
        "traffic": partial(generate_lstnet_dataset, dataset_name="traffic"),
        "exchange_rate_nips": partial(
            generate_gp_copula_dataset, dataset_name="exchange_rate_nips"
        ),
        "electricity_nips": partial(
            generate_gp_copula_dataset, dataset_name="electricity_nips"
        ),
        "traffic_nips": partial(
            generate_gp_copula_dataset, dataset_name="traffic_nips"
        ),
        "solar_nips": partial(
            generate_gp_copula_dataset, dataset_name="solar_nips"
        ),
        "wiki-rolling_nips": partial(
            generate_gp_copula_dataset, dataset_name="wiki-rolling_nips"
        ),
        "taxi_30min": partial(
            generate_gp_copula_dataset, dataset_name="taxi_30min"
        ),
        "m3_monthly": partial(generate_m3_dataset, m3_freq="monthly"),
        "m3_quarterly": partial(generate_m3_dataset, m3_freq="quarterly"),
        "m3_yearly": partial(generate_m3_dataset, m3_freq="yearly"),
        "m3_other": partial(generate_m3_dataset, m3_freq="other"),
        "m4_hourly": partial(
            generate_m4_dataset,
            m4_freq="Hourly",
            pandas_freq="H",
            prediction_length=48,
        ),
        "m4_daily": partial(
            generate_m4_dataset,
            m4_freq="Daily",
            pandas_freq="D",
            prediction_length=14,
        ),
        "m4_weekly": partial(
            generate_m4_dataset,
            m4_freq="Weekly",
            pandas_freq="W",
            prediction_length=13,
        ),
        "m4_monthly": partial(
            generate_m4_dataset,
            m4_freq="Monthly",
            pandas_freq="M",
            prediction_length=18,
        ),
        "m4_quarterly": partial(
            generate_m4_dataset,
            m4_freq="Quarterly",
            pandas_freq="Q",
            prediction_length=8,
        ),
        "m4_yearly": partial(
            generate_m4_dataset,
            m4_freq="Yearly",
            pandas_freq="Y",
            prediction_length=6,
        ),
        "m5": partial(
            generate_m5_dataset, pandas_freq="D", prediction_length=28
        ),
    }
)

dataset_names = list(dataset_recipes.keys())

default_dataset_path = get_download_path() / "datasets"


def materialize_dataset(
    dataset_name: str,
    path: Path = default_dataset_path,
    regenerate: bool = False,
) -> Path:
    """
    Ensures that the dataset is materialized under the `path / dataset_name`
    path.

    Parameters
    ----------
    dataset_name
        name of the dataset, for instance "m4_hourly"
    regenerate
        whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        where the dataset should be saved
    Returns
    -------
        the path where the dataset is materialized
    """
    assert dataset_name in dataset_recipes.keys(), (
        f"{dataset_name} is not present, please choose one from "
        f"{dataset_recipes.keys()}."
    )

    path.mkdir(parents=True, exist_ok=True)
    dataset_path = path / dataset_name

    dataset_recipe = dataset_recipes[dataset_name]

    if not dataset_path.exists() or regenerate:
        logging.info(f"downloading and processing {dataset_name}")
        dataset_recipe(dataset_path=dataset_path)
    else:
        logging.info(
            f"using dataset already processed in path {dataset_path}."
        )

    return dataset_path


def get_dataset(
    dataset_name: str,
    path: Path = default_dataset_path,
    regenerate: bool = False,
) -> TrainDatasets:
    """
    Get a repository dataset.

    The datasets that can be obtained through this function have been used
    with different processing over time by several papers (e.g., [SFG17]_,
    [LCY+18]_, and [YRD15]_).

    Parameters
    ----------
    dataset_name
        name of the dataset, for instance "m4_hourly"
    regenerate
        whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        where the dataset should be saved
    Returns
    -------
        dataset obtained by either downloading or reloading from local file.
    """
    dataset_path = materialize_dataset(dataset_name, path, regenerate)

    return load_datasets(
        metadata=dataset_path,
        train=dataset_path / "train",
        test=dataset_path / "test",
    )


if __name__ == "__main__":
    for dataset in dataset_names:
        print(f"generate {dataset}")
        ds = get_dataset(dataset, regenerate=True)
        print(ds.metadata)
        print(sum(1 for _ in list(iter(ds.train))))
