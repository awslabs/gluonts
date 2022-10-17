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

from dataclasses import dataclass
from functools import partial
from typing import Optional
from gluonts.ev.helpers import EvalData

import numpy as np
from gluonts.ev.api import (
    DerivedMetricEvaluator,
    MetricEvaluator,
    StandardMetricEvaluator,
)
from gluonts.ev.batch_aggregations import Mean, Sum
from gluonts.ev.metric_functions import (
    abs_error,
    abs_label,
    absolute_percentage_error,
    coverage,
    quantile_loss,
    squared_error,
    symmetric_absolute_percentage_error,
)
from ..exceptions import GluonTSUserError
from ..model.forecast import Forecast
from gluonts.time_feature import seasonality


class Metric:
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        raise NotImplementedError


@dataclass
class AbsLabelMean(Metric):
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=abs_label,
            aggregate=Mean(axis=axis),
        )


@dataclass
class AbsLabelSum(Metric):
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=abs_label,
            aggregate=Sum(axis=axis),
        )


@dataclass
class AbsErrorSum(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(abs_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MSE(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class QuantileLossSum(Metric):
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


# TODO: maybe just call this Coverage?
@dataclass
class CoverageMean(Metric):
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(coverage, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MAPE(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(absolute_percentage_error, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SMAPE(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(symmetric_absolute_percentage_error, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SeasonalError(Metric):
    seasonality: int = 1

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        if axis != 1:
            raise GluonTSUserError(
                "Seasonal error only works per data entry (axis 1)"
            )

        def seasonal_error_without_mean(
            data: dict,
            seasonality: int,
        ):
            past_data = data["input"]

            if seasonality < np.shape(past_data)[1]:
                forecast_freq = seasonality
            else:
                # edge case: the seasonal freq is larger than the length of ts
                forecast_freq = 1

            y_t = past_data[:, :-forecast_freq]
            y_tm = past_data[:, forecast_freq:]

            return np.abs(y_t - y_tm)

        return StandardMetricEvaluator(
            map=partial(
                seasonal_error_without_mean, seasonality=self.seasonality
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MSISNumerator(Metric):
    alpha: str = "0.05"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def msis_numerator(
            data: EvalData,
            alpha: float,
        ) -> np.ndarray:
            lower_quantile = data[str(alpha / 2)]
            upper_quantile = data[str(1.0 - alpha / 2)]
            label = data["label"]

            numerator = (
                upper_quantile
                - lower_quantile
                + 2.0
                / alpha
                * (lower_quantile - label)
                * (label < lower_quantile)
                + 2.0
                / alpha
                * (label - upper_quantile)
                * (label > upper_quantile)
            )

            return numerator

        return StandardMetricEvaluator(
            map=partial(msis_numerator, alpha=self.alpha),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MSIS(Metric):
    alpha: float = 0.05
    seasonality: int = 1

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            msis_numerator: np.ndarray, seasonal_error: np.ndarray
        ) -> np.ndarray:
            return msis_numerator / seasonal_error

        return DerivedMetricEvaluator(
            metrics={
                "msis_numerator": MSISNumerator(self.alpha)(axis=axis),
                "seasonal_error": SeasonalError(seasonality=self.seasonality)(
                    axis=axis
                ),
            },
            post_process=post_process,
        )


@dataclass
class MASE(Metric):
    forecast_type: str = "0.5"
    seasonality: int = 1

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            abs_error_sum: np.ndarray, seasonal_error: np.ndarray
        ) -> np.ndarray:
            return abs_error_sum / seasonal_error

        return DerivedMetricEvaluator(
            metrics={
                "abs_error_sum": AbsErrorSum(forecast_type=self.forecast_type)(
                    axis=axis
                ),
                "seasonal_error": SeasonalError()(axis=axis),
            },
            post_process=post_process,
        )


@dataclass
class ND(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            abs_error_sum: np.ndarray, abs_label_sum: np.ndarray
        ) -> np.ndarray:
            return abs_error_sum / abs_label_sum

        return DerivedMetricEvaluator(
            metrics={
                "abs_error_sum": AbsErrorSum(forecast_type=self.forecast_type)(
                    axis=axis
                ),
                "abs_label_sum": AbsLabelSum(forecast_type=self.forecast_type)(
                    axis=axis
                ),
            },
            post_process=post_process,
        )


@dataclass
class RMSE(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(mse: np.ndarray) -> np.ndarray:
            return np.sqrt(mse)

        return DerivedMetricEvaluator(
            metrics={"mse": MSE(forecast_type=self.forecast_type)(axis=axis)},
            post_process=post_process,
        )


@dataclass
class NRMSE(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            rmse: np.ndarray, abs_label_mean: np.ndarray
        ) -> np.ndarray:
            return rmse / abs_label_mean

        return DerivedMetricEvaluator(
            metrics={
                "rmse": RMSE()(axis=axis),
                "abs_label_mean": AbsLabelMean()(axis=axis),
            },
            post_process=post_process,
        )
