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

from pathlib import Path
from typing import Iterator, Tuple
import logging
import numpy as np
import pandas as pd

from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from gluonts.dataset.jsonl import JsonLinesFile
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.dataset.util import forecast_start

logger = logging.getLogger(__name__)


def load_forecasts_from_json_file(path: Path) -> list:
    ds = []
    for line in JsonLinesFile(path):
        ds.append(line.content)

    return ds


def numpify_samples(fcsts: list) -> list:
    for item in fcsts:
        item["SageMakerOutput"]["samples"] = np.array(
            item["SageMakerOutput"]["samples"], dtype="float"
        )

    return fcsts


def add_ts_dataframe(
    data_iterator: Iterator[DataEntry],
) -> Iterator[DataEntry]:
    for data_entry in data_iterator:
        data = data_entry.copy()
        index = pd.date_range(
            start=data["start"],
            freq=data["start"].freq,
            periods=data["target"].shape[-1],
        )
        data["ts"] = pd.DataFrame(index=index, data=data["target"].transpose())
        yield data


def ts_iter(dataset: Dataset) -> pd.DataFrame:
    for data_entry in add_ts_dataframe(iter(dataset)):
        yield data_entry["ts"]


def prepare_targets_for_metric_computation(
    targets_in: list, freq: str
) -> Iterator[DataEntry]:

    targets = ListDataset(targets_in, freq=freq)
    return ts_iter(targets)


def prepare_fcsts_for_metric_computation(
    fcsts: list, freq: str
) -> Iterator[Forecast]:

    sample_forecasts = []
    for fcst in fcsts:
        item = fcst.copy()
        samples = item["SageMakerOutput"]["samples"]
        item_id = item["item_id"]
        item["start"] = pd.Timestamp(item["start"], freq=freq)

        sample_forecasts.append(
            SampleForecast(
                samples=samples,
                start_date=forecast_start(item),
                freq=freq,
                item_id=item_id,
            )
        )

    return iter(sample_forecasts)


def compute_metrics_of_user_provided_forecasts(
    filename_fcsts: str, filename_targets: str, freq: str
) -> Tuple[dict, pd.DataFrame]:

    logger.info("Loading fcsts from json file")
    fcsts = load_forecasts_from_json_file(Path(filename_fcsts))
    fcsts = numpify_samples(fcsts)

    logger.info("Loading targets from json file")
    targets = load_forecasts_from_json_file(Path(filename_targets))

    targets_iter = prepare_targets_for_metric_computation(targets, freq)
    fcsts_iter = prepare_fcsts_for_metric_computation(fcsts, freq)

    agg_metrics, item_metrics = Evaluator(num_workers=0)(
        targets_iter, fcsts_iter, num_series=len(targets)
    )

    return agg_metrics, item_metrics
