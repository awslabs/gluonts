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

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import FallbackPredictor, RepresentablePredictor


class ConstantPredictor(RepresentablePredictor):
    """
    A `Predictor` that always produces the same forecast.

    Parameters
    ----------
    samples
        Samples to use to construct SampleForecast objects for every
        prediction.
    """

    @validated()
    def __init__(self, samples: np.ndarray) -> None:
        super().__init__(samples.shape[1])
        self.samples = samples

    def predict_item(self, item: DataEntry) -> SampleForecast:
        return SampleForecast(
            samples=self.samples,
            start_date=item["start"],
            item_id=item.get(FieldName.ITEM_ID),
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
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        value: float = 0.0,
        # since we are emitting a constant values, we just predict a single
        # line on default
        num_samples: int = 1,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.value = value
        self.num_samples = num_samples

    def predict_item(self, item: DataEntry) -> SampleForecast:
        samples_shape = self.num_samples, self.prediction_length
        samples = np.full(samples_shape, self.value)
        return SampleForecast(
            samples=samples,
            start_date=forecast_start(item),
            item_id=item.get("item_id"),
        )
