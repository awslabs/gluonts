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

from statsforecast.models import (
    ADIDA,
    AutoARIMA,
    AutoCES,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    ETS,
    IMAPA,
    TSB,
)

from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import QuantileForecast


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.

    Objects of this class are constructed with a type (for example,
    ``statsforecast.models.AutoARIMA``) and a dictionary of parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    model_constructor
        Model constructor (e.g. ``statsforecast.models.AutoARIMA``).
    kwargs
        Dictionary of keyword arguments to be passed for model construction.
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        List of quantile levels that we want predictions for. By default this
        is empty, in which case only the mean predition is returned.
    """

    @validated()
    def __init__(
        self,
        model_constructor,
        kwargs: dict,
        prediction_length: int,
        quantile_levels: list = [],
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = model_constructor(**kwargs)
        self.quantile_levels = quantile_levels
    
    def quantile_key(self, quantile_level) -> str:
        """
        Turn a given quantile level into the key to extract the quantile
        from ``statsforecast`` predictions.
        """
        # NOTE currently it's tricky to get quantiles out of
        # NOTE statsforecast predictions
        # NOTE see https://github.com/Nixtla/statsforecast/issues/256
        raise NotImplementedError()

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        # TODO use also exogenous features
        pred = self.model.forecast(
            y=entry["target"],
            h=self.prediction_length,
        )

        to_stack = [pred["mean"]]
        for quantile_level in self.quantile_levels:
            key = self.quantile_key(quantile_level)
            to_stack.append(pred[key].values)

        return QuantileForecast(
            forecast_arrays=np.stack(to_stack, axis=0),
            forecast_keys=["mean"] + self.quantile_levels,
            start_date=entry["start"] + len(entry["target"]),
            item_id=entry.get("item_id"),
        )


def ADIDAPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``ADIDA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``ADIDA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=ADIDA,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def AutoARIMAPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``AutoARIMA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``AutoARIMA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=AutoARIMA,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def AutoCESPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``AutoCES`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``AutoCES``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=AutoCES,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def CrostonClassicPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``CrostonClassic`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``CrostonClassic``: please refer
    to the documentation of `statsforecast`_ to know more about its
    parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=CrostonClassic,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def CrostonOptimizedPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``CrostonOptimized`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``CrostonOptimized``: please refer
    to the documentation of `statsforecast`_ to know more about its
    parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=CrostonOptimized,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def CrostonSBAPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``CrostonSBA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``CrostonSBA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=CrostonSBA,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def ETSPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``ETS`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``ETS``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=ETS,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def IMAPAPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``IMAPA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``IMAPA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=IMAPA,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )


def TSBPredictor(prediction_length: int, quantile_levels=[], **kwargs) -> StatsForecastPredictor:
    """
    A predictor based on the ``TSB`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``TSB``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    return StatsForecastPredictor(
        model_constructor=TSB,
        kwargs=kwargs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
