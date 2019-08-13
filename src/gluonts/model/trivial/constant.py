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

# Standard library imports
from functools import partial
from typing import Iterator

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor, FallbackPredictor
from gluonts.support.pandas import forecast_start


class ConstantPredictor(RepresentablePredictor):
    """
    A `Predictor` that always produces the same forecast.

    Parameters
    ----------
    samples
        Samples to use to construct SampleForecast objects for every
        prediction.
    freq
        Frequency of the predicted data.
    """

    @validated()
    def __init__(self, samples: np.ndarray, freq: str) -> None:
        super().__init__(samples.shape[1], freq)
        self.samples = samples

    def predict_item(self, item: DataEntry) -> SampleForecast:
        return SampleForecast(
            samples=self.samples,
            start_date=item["start"],
            freq=self.freq,
            item_id=item["id"] if "id" in item else None,
        )


class ConstantValuePredictor(RepresentablePredictor, FallbackPredictor):
    """
    A `Predictor` that always produces the same value as forecast.

    Parameters
    ----------
    value
        The value to use as forecast.
    prediction_length
        Prediction horizon.
    freq
        Frequency of the predicted data.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        value: float = 0.0,
        # since we are emitting a constant values, we just predict a single
        # line on default
        num_eval_samples: int = 1,
    ) -> None:
        super().__init__(prediction_length, freq)
        self.value = value
        self.num_eval_samples = num_eval_samples

    def predict_item(self, item: DataEntry) -> SampleForecast:
        samples_shape = self.num_eval_samples, self.prediction_length
        samples = np.full(samples_shape, self.value)
        return SampleForecast(
            samples=samples,
            start_date=forecast_start(item),
            freq=self.freq,
            item_id=item.get("id"),
        )
