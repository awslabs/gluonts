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
from typing import List, Optional, Type

import numpy as np

from statsforecast.models import (
    ADIDA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTheta,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    DynamicOptimizedTheta,
    DynamicTheta,
    HistoricAverage,
    Holt,
    HoltWinters,
    IMAPA,
    MSTL,
    Naive,
    OptimizedTheta,
    RandomWalkWithDrift,
    SeasonalExponentialSmoothing,
    SeasonalExponentialSmoothingOptimized,
    SeasonalNaive,
    SeasonalWindowAverage,
    SimpleExponentialSmoothing,
    SimpleExponentialSmoothingOptimized,
    TSB,
    Theta,
    WindowAverage,
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

    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsforecast`` model type to use.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsforecast``
        for details.
    """

    ModelType: Type

    @validated()
    def __init__(
        self,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = self.ModelType(**model_params)
        self.config = ModelConfig(quantile_levels=quantile_levels)

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        # TODO use also exogenous features
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals

        prediction = self.model.forecast(
            y=entry["target"],
            h=self.prediction_length,
            **kwargs,
        )

        forecast_arrays = [
            prediction[k] for k in self.config.statsforecast_keys
        ]

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

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = ADIDA


class AutoARIMAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoARIMA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = AutoARIMA


class AutoCESPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoCES`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = AutoCES


class AutoETSPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoETS`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = AutoETS


class AutoThetaPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``AutoTheta`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = AutoTheta


class CrostonClassicPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonClassic`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = CrostonClassic


class CrostonOptimizedPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonOptimized`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = CrostonOptimized


class CrostonSBAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``CrostonSBA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = CrostonSBA


class IMAPAPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``IMAPA`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = IMAPA


class DynamicOptimizedThetaPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``DynamicOptimizedTheta`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = DynamicOptimizedTheta


class DynamicThetaPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``DynamicTheta`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = DynamicTheta


class HistoricAveragePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``HistoricAverage`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = HistoricAverage


class HoltPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``Holt`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = Holt


class HoltWintersPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``HoltWinters`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = HoltWinters


class MSTLPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``MSTL`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = MSTL


class NaivePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``Naive`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = Naive


class OptimizedThetaPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``OptimizedTheta`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = OptimizedTheta


class RandomWalkWithDriftPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``RandomWalkWithDrift`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = RandomWalkWithDrift


class SeasonalExponentialSmoothingPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``SeasonalExponentialSmoothing`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = SeasonalExponentialSmoothing


class SeasonalExponentialSmoothingOptimizedPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``SeasonalExponentialSmoothingOptimized`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = SeasonalExponentialSmoothingOptimized


class SeasonalNaivePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``SeasonalNaive`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = SeasonalNaive


class SeasonalWindowAveragePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``SeasonalWindowAverage`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = SeasonalWindowAverage


class SimpleExponentialSmoothingPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``SimpleExponentialSmoothing`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = SimpleExponentialSmoothing


class SimpleExponentialSmoothingOptimizedPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``SimpleExponentialSmoothingOptimized`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = SimpleExponentialSmoothingOptimized


class TSBPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``TSB`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = TSB


class ThetaPredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``Theta`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = Theta


class WindowAveragePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``WindowAverage`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = WindowAverage
