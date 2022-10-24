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


from typing import ChainMap, Dict, Iterator

import numpy as np

from gluonts.time_feature.seasonality import get_seasonality
from gluonts.model.predictor import Predictor
from gluonts.dataset.split import TestData
from .stats import seasonal_error


# TODO: maybe move the content of this file out of ev module


class EvaluationDataBatch:
    """Used to add batch dimension
    Should be replaced by a `ForecastBatch` eventually"""

    def __init__(self, values) -> None:
        self.values = values

    def __getitem__(self, name):
        return np.array([self.values[name]])


def construct_data(
    test_data: TestData, predictor: Predictor, **predictor_kwargs
) -> Iterator[Dict[str, np.ndarray]]:
    forecasts = predictor.predict(dataset=test_data.input, **predictor_kwargs)

    for input, label, forecast in zip(
        test_data.input, test_data.label, forecasts
    ):
        batching_used = False  # isinstance(forecast, ForecastBatch)

        non_forecast_data = {
            "label": label["target"],
            "seasonal_error": seasonal_error(
                input["target"],
                seasonality=get_seasonality(freq=forecast.start_date.freqstr),
            ),
        }
        joint_data = ChainMap(non_forecast_data, forecast)

        if batching_used:
            yield joint_data
        else:
            yield EvaluationDataBatch(joint_data)
