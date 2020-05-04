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
from typing import Iterator, Optional

# Third-party imports
import numpy as np
from pydantic import PositiveInt

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.trivial.constant import ConstantPredictor
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor, FallbackPredictor
from gluonts.support.pandas import frequency_add


class MeanPredictor(RepresentablePredictor, FallbackPredictor):
    """
    A :class:`Predictor` that predicts the samples based on the mean of the
    last `context_length` elements of the input target.

    Parameters
    ----------
    context_length
        Length of the target context used to condition the predictions.
    prediction_length
        Length of the prediction horizon.
    num_samples
        Number of samples to use to construct :class:`SampleForecast` objects
        for every prediction.
    freq
        Frequency of the predicted data.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        num_samples: int = 100,
        context_length: Optional[int] = None,
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)
        self.context_length = context_length
        self.num_samples = num_samples
        self.shape = (self.num_samples, self.prediction_length)

    def predict_item(self, item: DataEntry) -> SampleForecast:
        if self.context_length is not None:
            target = item["target"][-self.context_length :]
        else:
            target = item["target"]

        mean = np.nanmean(target)
        std = np.nanstd(target)
        normal = np.random.standard_normal(self.shape)

        start_date = frequency_add(item["start"], len(item["target"]))
        return SampleForecast(
            samples=std * normal + mean,
            start_date=start_date,
            freq=self.freq,
            item_id=item.get(FieldName.ITEM_ID),
        )


class MeanEstimator(Estimator):
    """
    An `Estimator` that computes the mean targets in the training data,
    in the trailing `prediction_length` observations, and produces
    a `ConstantPredictor` that always predicts such mean value.

    Parameters
    ----------
    prediction_length
        Prediction horizon.
    freq
        Frequency of the predicted data.
    num_samples
        Number of samples to include in the forecasts. Not that the samples
        produced by this predictor will all be identical.
    """

    @validated()
    def __init__(
        self,
        prediction_length: PositiveInt,
        freq: str,
        num_samples: PositiveInt,
    ) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.freq = freq
        self.num_samples = num_samples

    def train(
        self,
        training_data: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> ConstantPredictor:
        contexts = np.array(
            [
                item["target"][-self.prediction_length :]
                for item in training_data
            ]
        )

        samples = np.broadcast_to(
            array=contexts.mean(axis=0),
            shape=(self.num_samples, self.prediction_length),
        )

        return ConstantPredictor(samples=samples, freq=self.freq)
