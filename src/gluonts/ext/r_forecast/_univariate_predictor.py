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

from typing import Dict, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.model.forecast import QuantileForecast

from . import RBasePredictor
from .util import (
    unlist,
    quantile_to_interval_level,
    interval_to_quantile_level,
)


UNIVARIATE_QUANTILE_FORECAST_METHODS = [
    "arima",
    "ets",
    "tbats",
    "thetaf",
    "stlar",
    "fourier.arima",
]
UNIVARIATE_POINT_FORECAST_METHODS = ["croston", "mlp"]
SUPPORTED_UNIVARIATE_METHODS = (
    UNIVARIATE_QUANTILE_FORECAST_METHODS + UNIVARIATE_POINT_FORECAST_METHODS
)


class RForecastPredictor(RBasePredictor):
    """
    Wrapper for calling the `R forecast package
    <http://pkg.robjhyndman.com/forecast/>`_.

    In order to use it you need to install R and rpy2. You also need the R `forecast` package which
    can be installed by running:

        R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")' # noqa

    Parameters
    ----------
    freq
        The granularity of the time series (e.g. '1H')
    prediction_length
        Number of time points to be predicted.
    method_name
        The method from rforecast to be used one of
        "ets", "arima", "tbats", "croston", "mlp", "thetaf".
    period
        The period to be used (this is called `frequency` in the R forecast
        package), result to a tentative reasonable default if not specified
        (for instance 24 for hourly freq '1H')
    trunc_length
        Maximum history length to feed to the model (some models become slow
        with very long series).
    params
        Parameters to be used when calling the forecast method default.
        For `output_type`, 'mean' and `quantiles` are supported (depending
        on the underlying R method).
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        method_name: str = "ets",
        period: int = None,
        trunc_length: Optional[int] = None,
        save_info: bool = False,
        params: Dict = dict(),
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            period=period,
            trunc_length=trunc_length,
            save_info=save_info,
            r_file_prefix="univariate",
        )
        assert method_name in SUPPORTED_UNIVARIATE_METHODS, (
            f"method {method_name} is not supported please "
            f"use one of {SUPPORTED_UNIVARIATE_METHODS}"
        )

        self.method_name = method_name
        self._r_method = self._robjects.r[method_name]

        self.params = {
            "prediction_length": self.prediction_length,
            "frequency": self.period,
        }

        if self.method_name in UNIVARIATE_POINT_FORECAST_METHODS:
            self.params["output_types"] = ["mean"]
        elif self.method_name in UNIVARIATE_QUANTILE_FORECAST_METHODS:
            self.params["output_types"] = ["mean", "quantiles"]
            self.params["intervals"] = list(range(0, 100, 10))

        if "quantiles" in params:
            assert (
                "intervals" not in params
            ), "Cannot specify both 'quantiles' and 'intervals'."
            intervals_info = [
                quantile_to_interval_level(ql) for ql in params["quantiles"]
            ]
            params["intervals"] = sorted(
                set([level for level, _ in intervals_info])
            )
            params.pop("quantiles")

        self.params.update(params)

        # Always ask for the mean prediction to be given,
        # since QuantileForecast will otherwise impute it
        # using the median, which is undesired.
        if "mean" not in self.params["output_types"]:
            self.params["output_types"].append("mean")

    def _get_r_forecast(self, data: Dict) -> Dict:
        make_ts = self._stats_pkg.ts
        r_params = self._robjects.vectors.ListVector(self.params)
        vec = self._robjects.FloatVector(data["target"])
        ts = make_ts(vec, frequency=self.period)
        forecast = self._r_method(ts, r_params)

        forecast_dict = dict(zip(forecast.names, map(unlist, list(forecast))))

        if "quantiles" in self.params["output_types"]:
            upper_quantiles = [
                str(interval_to_quantile_level(interval, side="upper"))
                for interval in self.params["intervals"]
            ]

            lower_quantiles = [
                str(interval_to_quantile_level(interval, side="lower"))
                for interval in self.params["intervals"]
            ]

            forecast_dict["quantiles"] = dict(
                zip(
                    lower_quantiles + upper_quantiles,
                    forecast_dict["lower_quantiles"]
                    + forecast_dict["upper_quantiles"],
                )
            )

        return forecast_dict

    def _preprocess_data(self, data: Dict) -> Dict:
        if self.trunc_length:
            shift_by = max(data["target"].shape[0] - self.trunc_length, 0)
            data["start"] = data["start"] + shift_by
            data["target"] = data["target"][-self.trunc_length :]
        return data

    def _warning_message(self) -> None:
        if self.method_name in UNIVARIATE_POINT_FORECAST_METHODS:
            print(
                "Overriding `output_types` to `mean` since"
                f" {self.method_name} is a point forecast method."
            )
        elif self.method_name in UNIVARIATE_QUANTILE_FORECAST_METHODS:
            print(
                "Overriding `output_types` to `quantiles` since "
                f"{self.method_name} is a quantile forecast method."
            )

    def _forecast_dict_to_obj(
        self,
        forecast_dict: Dict,
        forecast_start_date: pd.Timestamp,
        item_id: Optional[str],
        info: Dict,
    ) -> QuantileForecast:
        stats_dict = {"mean": forecast_dict["mean"]}

        if "quantiles" in forecast_dict:
            stats_dict.update(forecast_dict["quantiles"])

        forecast_arrays = np.array(list(stats_dict.values()))

        assert forecast_arrays.shape[1] == self.prediction_length

        return QuantileForecast(
            forecast_arrays=forecast_arrays,
            forecast_keys=list(stats_dict.keys()),
            start_date=forecast_start_date,
            item_id=item_id,
            info=info,
        )
