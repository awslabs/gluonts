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
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from gluonts.dataset import DatasetWriter
from gluonts.dataset.artificial import ConstantDataset
from gluonts.dataset.common import TrainDatasets, load_datasets
from gluonts.dataset.jsonl import JsonLinesWriter

from ._artificial import generate_artificial_dataset
from ._airpassengers import generate_airpassengers_dataset
from ._gp_copula_2019 import generate_gp_copula_dataset
from ._lstnet import generate_lstnet_dataset
from ._m3 import generate_m3_dataset
from ._m4 import generate_m4_dataset
from ._m5 import generate_m5_dataset
from ._tsf_datasets import generate_forecasting_dataset
from ._uber_tlc import generate_uber_dataset


def get_download_path() -> Path:
    """

    Returns
    -------
    Path
        default path to download datasets or models of gluon-ts.
        The path is either $MXNET_HOME if the environment variable is defined
        or /home/username/.mxnet/gluon-ts/
    """
    mxnet_home = os.environ.get("MXNET_HOME", None)

    if mxnet_home is not None:
        return Path(mxnet_home)

    return Path.home() / ".mxnet" / "gluon-ts"


dataset_recipes = {
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
    "kaggle_web_traffic_with_missing": partial(
        generate_forecasting_dataset,
        dataset_name="kaggle_web_traffic_with_missing",
    ),
    "kaggle_web_traffic_without_missing": partial(
        generate_forecasting_dataset,
        dataset_name="kaggle_web_traffic_without_missing",
    ),
    "kaggle_web_traffic_weekly": partial(
        generate_forecasting_dataset,
        dataset_name="kaggle_web_traffic_weekly",
    ),
    "m1_yearly": partial(
        generate_forecasting_dataset, dataset_name="m1_yearly"
    ),
    "m1_quarterly": partial(
        generate_forecasting_dataset, dataset_name="m1_quarterly"
    ),
    "m1_monthly": partial(
        generate_forecasting_dataset, dataset_name="m1_monthly"
    ),
    "nn5_daily_with_missing": partial(
        generate_forecasting_dataset, dataset_name="nn5_daily_with_missing"
    ),
    "nn5_daily_without_missing": partial(
        generate_forecasting_dataset,
        dataset_name="nn5_daily_without_missing",
    ),
    "nn5_weekly": partial(
        generate_forecasting_dataset, dataset_name="nn5_weekly"
    ),
    "tourism_monthly": partial(
        generate_forecasting_dataset, dataset_name="tourism_monthly"
    ),
    "tourism_quarterly": partial(
        generate_forecasting_dataset, dataset_name="tourism_quarterly"
    ),
    "tourism_yearly": partial(
        generate_forecasting_dataset, dataset_name="tourism_yearly"
    ),
    "cif_2016": partial(generate_forecasting_dataset, dataset_name="cif_2016"),
    "london_smart_meters_without_missing": partial(
        generate_forecasting_dataset,
        dataset_name="london_smart_meters_without_missing",
    ),
    "wind_farms_without_missing": partial(
        generate_forecasting_dataset,
        dataset_name="wind_farms_without_missing",
    ),
    "car_parts_without_missing": partial(
        generate_forecasting_dataset,
        dataset_name="car_parts_without_missing",
    ),
    "dominick": partial(generate_forecasting_dataset, dataset_name="dominick"),
    "fred_md": partial(generate_forecasting_dataset, dataset_name="fred_md"),
    "pedestrian_counts": partial(
        generate_forecasting_dataset, dataset_name="pedestrian_counts"
    ),
    "hospital": partial(generate_forecasting_dataset, dataset_name="hospital"),
    "covid_deaths": partial(
        generate_forecasting_dataset, dataset_name="covid_deaths"
    ),
    "kdd_cup_2018_without_missing": partial(
        generate_forecasting_dataset,
        dataset_name="kdd_cup_2018_without_missing",
    ),
    "weather": partial(generate_forecasting_dataset, dataset_name="weather"),
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
        generate_m5_dataset,
        pandas_freq="D",
        prediction_length=28,
        m5_file_path=get_download_path() / "m5",
    ),
    "uber_tlc_daily": partial(
        generate_uber_dataset, uber_freq="Daily", prediction_length=7
    ),
    "uber_tlc_hourly": partial(
        generate_uber_dataset, uber_freq="Hourly", prediction_length=24
    ),
    "airpassengers": partial(generate_airpassengers_dataset),
}


dataset_names = list(dataset_recipes.keys())

default_dataset_path = get_download_path() / "datasets"

default_dataset_writer = JsonLinesWriter()


def materialize_dataset(
    dataset_name: str,
    path: Path = default_dataset_path,
    regenerate: bool = False,
    dataset_writer: DatasetWriter = default_dataset_writer,
    prediction_length: Optional[int] = None,
) -> Path:
    """
    Ensures that the dataset is materialized under the `path / dataset_name`
    path.

    Parameters
    ----------
    dataset_name
        Name of the dataset, for instance "m4_hourly".
    regenerate
        Whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        Where the dataset should be saved.
    prediction_length
        The prediction length to be used for the dataset. If None, the default
        prediction length will be used. The prediction length might not be
        available for all datasets.

    Returns
    -------
        The path where the dataset is materialized
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
        if dataset_path.exists():
            # If regenerating, we need to remove the directory contents
            shutil.rmtree(dataset_path)
            dataset_path.mkdir()

        # Optionally pass prediction length to not override any non-None
        # defaults (e.g. for M4)
        kwargs: Dict[str, Any] = {"dataset_path": dataset_path}
        if prediction_length is not None:
            kwargs["prediction_length"] = prediction_length
        kwargs["dataset_writer"] = dataset_writer
        dataset_recipe(**kwargs)
    else:
        logging.info(
            f"using dataset already processed in path {dataset_path}."
        )

    return dataset_path


def get_dataset(
    dataset_name: str,
    path: Path = default_dataset_path,
    regenerate: bool = False,
    dataset_writer: DatasetWriter = default_dataset_writer,
    prediction_length: Optional[int] = None,
) -> TrainDatasets:
    """
    Get a repository dataset.

    The datasets that can be obtained through this function have been used
    with different processing over time by several papers (e.g., [SFG17]_,
    [LCY+18]_, and [YRD15]_) or are obtained through the `Monash Time Series
    Forecasting Repository <https://forecastingdata.org/>`_.

    Parameters
    ----------
    dataset_name
        Name of the dataset, for instance "m4_hourly".
    regenerate
        Whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        Where the dataset should be saved.
    prediction_length
        The prediction length to be used for the dataset. If None, the default
        prediction length will be used. If the dataset is already materialized,
        setting this option to a different value does not have an effect.
        Make sure to set `regenerate=True` in this case. Note that some
        datasets from the Monash Time Series Forecasting Repository do not
        actually have a default prediction length -- the default then depends
        on the frequency of the data:
        - Minutely data --> prediction length of 60 (one hour)
        - Hourly data --> prediction length of 48 (two days)
        - Daily data --> prediction length of 30 (one month)
        - Weekly data --> prediction length of 8 (two months)
        - Monthly data --> prediction length of 12 (one year)
        - Yearly data --> prediction length of 4 (four years)

    Returns
    -------
        Dataset obtained by either downloading or reloading from local file.
    """
    dataset_path = materialize_dataset(
        dataset_name,
        path,
        regenerate,
        dataset_writer,
        prediction_length,
    )

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
        print(len(ds.train))
