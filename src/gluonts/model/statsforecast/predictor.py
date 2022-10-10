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
from gluonts.dataset import DataEntry
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import SampleForecast


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.

    Objects of this class are constructed with a type (for example,
    ``statsforecast.models.AutoARIMA``) and a dictionary of parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    constructor
        Model constructor (e.g. ``statsforecast.models.AutoARIMA``).
    kwargs
        Dictionary of keyword arguments to be passed for model construction.
    """

    @validated()
    def __init__(self, prediction_length: int, constructor, kwargs: dict):
        super().__init__(prediction_length=prediction_length)
        self.model = constructor(**kwargs)

    def predict_item(self, entry: DataEntry) -> SampleForecast:
        # TODO use also exogenous features
        pred = self.model.forecast(
            y=entry["target"],
            h=self.prediction_length,
        )
        # TODO return QuantileForecast instead
        # NOTE currently it's tricky to get quantiles out of
        # NOTE statsforecast predictions
        # NOTE see https://github.com/Nixtla/statsforecast/issues/256
        return SampleForecast(
            samples=np.expand_dims(pred["mean"], axis=0),
            start_date=entry["start"] + len(entry["target"]),
            item_id=entry.get("item_id"),
        )
