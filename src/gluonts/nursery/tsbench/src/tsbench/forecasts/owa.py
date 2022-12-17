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
from typing import cast, Dict, Hashable, List, Tuple, TypeVar
import numpy as np
from gluonts.dataset.common import Dataset, MetaData
from gluonts.model.naive_2 import Naive2Predictor
from gluonts.time_feature.seasonality import get_seasonality
from tsbench.config.dataset import DatasetSplit
from .metrics import mase, naive_error, smape
from .prediction import generate_forecasts
from .quantile import QuantileForecasts

K = TypeVar("K", bound=Hashable)


def compute_owa(
    metrics: List[Tuple[float, float]],
    datasets: Dict[K, DatasetSplit],
    metadata: List[MetaData],
) -> float:
    """
    Computes the OWA metric from the M4 competition, using a weighted average
    of the relative MASE and sMAPE metrics depending on the size of the
    datasets.

    Args:
        metrics: The forecast's metrics (MASE and sMAPE).
        datasets: The datasets for which the forecasts have been generated, mapped from a hashable
            so that computations do not have to be repeated.
        metadata: Metadata available for the dataset.

    Returns:
        The OWA metric value.
    """
    assert (
        len(metrics) == len(datasets) == len(metadata)
    ), "The lengths of the provided lists must be equal."

    dataset_weights = np.array([len(d.gluonts()) for d in datasets.values()])
    dataset_weights = dataset_weights / dataset_weights.sum()

    naive_mase = 0
    naive_smape = 0
    actual_mase = 0
    actual_smape = 0

    for metric, (dataset_key, split), meta, weight in zip(
        metrics, datasets.items(), metadata, dataset_weights
    ):
        cache_file = Path.home() / ".cache" / "naive2" / f"{dataset_key}"
        if cache_file.exists():
            naive_forecast = QuantileForecasts.load(cache_file)
        else:
            naive_forecast = _naive_2_forecasts(
                split.gluonts(), meta.freq, cast(int, meta.prediction_length)
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            naive_forecast.save(cache_file)

        data = split.evaluation()
        seasonal_error = naive_error(data.past, get_seasonality(meta.freq))

        naive_mase += (
            mase(naive_forecast.median, data.future, seasonal_error) * weight
        )
        naive_smape += smape(naive_forecast.median, data.future) * weight

        actual_mase += metric[0] * weight
        actual_smape += metric[1] * weight

    return 0.5 * (actual_smape / naive_smape + actual_mase / naive_mase)


def _naive_2_forecasts(
    dataset: Dataset, freq: str, prediction_length: int
) -> QuantileForecasts:
    naive_predictor = Naive2Predictor(freq, prediction_length)
    return generate_forecasts(naive_predictor, dataset)[0]
