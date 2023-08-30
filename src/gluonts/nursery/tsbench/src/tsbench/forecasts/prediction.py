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
import time
from typing import cast, Tuple
import numpy as np
from gluonts.dataset.common import Dataset
from gluonts.env import env
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.model.predictor import ParallelizedPredictor, Predictor
from gluonts.support.util import maybe_len
from pandas.tseries.frequencies import to_offset
from tqdm.auto import tqdm
from .quantile import QuantileForecasts


def generate_forecasts(
    predictor: Predictor,
    dataset: Dataset,
    num_samples: int = 100,
    parallelize: bool = False,
) -> Tuple[QuantileForecasts, float]:
    """
    Generates the predictions of the given predictor for the provided dataset.
    The returned prediction object provides the forecasts along with some
    metadata.

    Args:
        predictor: The predictor which is used to make forecasts.
        dataset: The GluonTS dataset which is used for testing.
        num_samples: The number of samples to use for making predictions.
        parallelize: Whether predictions ought to be parallelized.

    Returns:
        The forecasts for the dataset.
        The average latency for generating a single forecast.
    """
    if parallelize:
        predictor = ParallelizedPredictor(
            predictor, num_workers=os.cpu_count()
        )

    # First, perform the predictions...
    tic = time.time()
    forecast_pred, _ = make_evaluation_predictions(
        dataset, predictor, num_samples
    )

    # ...and compute the quantiles
    quantiles = [f"0.{i+1}" for i in range(9)]
    forecasts = []
    for i, forecast in tqdm(
        enumerate(forecast_pred),
        total=maybe_len(dataset),
        disable=not env.use_tqdm,
    ):
        result = None
        if isinstance(forecast, QuantileForecast):
            if forecast.forecast_keys == quantiles:
                result = forecast
        elif isinstance(forecast, SampleForecast):
            quantile_forecast = forecast.to_quantile_forecast(quantiles)  # type: ignore
            result = quantile_forecast

        if result is None:
            # If none of the above checks added a quantile forecast, we resort to a method that
            # should work on all types of forecasts
            result = QuantileForecast(
                forecast_arrays=np.stack(
                    [forecast.quantile(q) for q in quantiles], axis=0
                ),
                start_date=forecast.start_date,
                freq=forecast.freq,
                forecast_keys=quantiles,
                item_id=forecast.item_id,
            )

        if result.item_id is None:
            result.item_id = i
        forecasts.append(result)

    toc = time.time()

    # Then, we compute the prediction latency
    latency = (toc - tic) / len(dataset)
    if parallelize:
        # We observed that N CPUs only brought a speedup of ~N/2
        latency = latency * (cast(int, os.cpu_count()) / 2)

    # And convert the list of forecasts into a QuantileForecasts object
    quantile_forecasts = QuantileForecasts(
        values=np.stack([f.forecast_array for f in forecasts]),
        start_dates=np.array([f.start_date for f in forecasts]),
        item_ids=np.array([str(f.item_id) for f in forecasts]),
        freq=to_offset(forecasts[0].freq),  # type: ignore
        quantiles=forecasts[0].forecast_keys,
    )
    return quantile_forecasts, latency
