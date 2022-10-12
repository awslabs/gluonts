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

from dataclasses import dataclass, field
from typing import List, Optional

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
from gluonts.dataset.util import forecast_start
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import QuantileForecast


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(
                200 * (max(quantile_level, 1 - quantile_level) - 0.5)
            )
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


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
        Optional list of quantile levels that we want predictions for.
        By default this is ``None``, giving only the mean predition.
    """

    @validated()
    def __init__(
        self,
        model_constructor,
        kwargs: dict,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = model_constructor(**kwargs)
        self.config = ModelConfig(quantile_levels=quantile_levels)

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        # TODO use also exogenous features
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals

        pred = self.model.forecast(
            y=entry["target"],
            h=self.prediction_length,
            **kwargs,
        )

        forecast_arrays = [pred[k] for k in self.config.statsforecast_keys]

        return QuantileForecast(
            forecast_arrays=np.stack(forecast_arrays, axis=0),
            forecast_keys=self.config.forecast_keys,
            start_date=forecast_start(entry),
            item_id=entry.get("item_id"),
            info=entry.get("info"),
        )


class ADIDAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``ADIDA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``ADIDA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=ADIDA,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class AutoARIMAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoARIMA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``AutoARIMA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=AutoARIMA,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class AutoCESPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoCES`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``AutoCES``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=AutoCES,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class CrostonClassicPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonClassic`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``CrostonClassic``: please refer
    to the documentation of `statsforecast`_ to know more about its
    parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=CrostonClassic,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class CrostonOptimizedPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonOptimized`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``CrostonOptimized``: please refer
    to the documentation of `statsforecast`_ to know more about its
    parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=CrostonOptimized,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class CrostonSBAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonSBA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``CrostonSBA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=CrostonSBA,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class ETSPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``ETS`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``ETS``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=ETS,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class IMAPAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``IMAPA`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``IMAPA``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=IMAPA,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )


class TSBPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``TSB`` model from `statsforecast`_.

    The predictor is constructed with a ``prediction_length`` parameters,
    plus all additional arguments needed by ``TSB``: please refer to
    the documentation of `statsforecast`_ to know more about its parameters.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    @validated()
    def __init__(
        self, prediction_length: int, quantile_levels=None, **kwargs
    ) -> None:
        super().__init__(
            model_constructor=TSB,
            kwargs=kwargs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
