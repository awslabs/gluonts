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

from typing import Dict, Optional, List, Union

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.model.r_forecast import RBasePredictor


R_FILE_PREFIX = "univariate"

UNIVARIATE_SAMPLE_FORECAST_METHODS = ["ets", "arima"]
UNIVARIATE_QUANTILE_FORECAST_METHODS = ["tbats", "thetaf", "stlar"]
UNIVARIATE_POINT_FORECAST_METHODS = ["croston", "mlp"]
SUPPORTED_UNIVARIATE_METHODS = (
    UNIVARIATE_SAMPLE_FORECAST_METHODS
    + UNIVARIATE_QUANTILE_FORECAST_METHODS
    + UNIVARIATE_POINT_FORECAST_METHODS
)


class RForecastPredictor(RBasePredictor):
    """
    Wrapper for calling the `R forecast package
    <http://pkg.robjhyndman.com/forecast/>`_.

    In order to use it you need to install R and run::

        pip install 'rpy2>=2.9.*,<3.*'
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
        Note that, as `output_type`, only 'samples' and `quantiles` (depending on the underlying R method)
        are supported currently.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        method_name: str = "ets",
        period: int = None,
        trunc_length: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> None:

        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            period=period,
            trunc_length=trunc_length,
            r_file_prefix=R_FILE_PREFIX,
        )
        assert method_name in SUPPORTED_UNIVARIATE_METHODS, (
            f"method {method_name} is not supported please "
            f"use one of {SUPPORTED_UNIVARIATE_METHODS}"
        )

        self.method_name = method_name
        self._r_method = self._robjects.r[method_name]
        self.params = {
            "prediction_length": self.prediction_length,
            "output_types": ["samples"],
            "frequency": self.period,
        }
        if params is not None:
            self.params.update(params)

    def _get_r_forecast(self, data: Dict, params: Dict) -> Dict:
        make_ts = self._stats_pkg.ts
        r_params = self._robjects.vectors.ListVector(params)
        vec = self._robjects.FloatVector(data["target"])
        ts = make_ts(vec, frequency=self.period)
        forecast = self._r_method(ts, r_params)
        forecast_dict = dict(
            zip(forecast.names, map(self._unlist, list(forecast)))
        )

        if "quantiles" in forecast_dict or "upper_quantiles" in forecast_dict:

            def from_interval_to_level(interval: int, side: str):
                if side == "upper":
                    level = 50 + interval / 2
                elif side == "lower":
                    level = 50 - interval / 2
                else:
                    raise ValueError
                return level / 100

            # Post-processing quantiles on then Python side for the
            # convenience of asserting and debugging.
            upper_quantiles = [
                str(from_interval_to_level(interval, side="upper"))
                for interval in params["intervals"]
            ]

            lower_quantiles = [
                str(from_interval_to_level(interval, side="lower"))
                for interval in params["intervals"]
            ]

            # Median forecasts would be available at two places:
            # Lower 0 and Higher 0 (0-prediction interval)
            forecast_dict["quantiles"] = dict(
                zip(
                    lower_quantiles + upper_quantiles[1:],
                    forecast_dict["lower_quantiles"]
                    + forecast_dict["upper_quantiles"][1:],
                )
            )

            # `QuantileForecast` allows "mean" as the key;
            # we store them as well since they can differ from median.
            forecast_dict["quantiles"].update(
                {"mean": forecast_dict.pop("mean")}
            )

        return forecast_dict

    def _preprocess_data(self, data: Dict) -> Dict:
        if self.trunc_length:
            shift_by = max(data["target"].shape[0] - self.trunc_length, 0)
            data["start"] = data["start"] + shift_by
            data["target"] = data["target"][-self.trunc_length :]
        return data

    def _override_params(
        self, params: Dict, num_samples: int, intervals: Optional[List] = None
    ) -> Dict:
        params["num_samples"] = num_samples

        if self.method_name in UNIVARIATE_POINT_FORECAST_METHODS:
            params["output_types"] = ["mean"]
        elif self.method_name in UNIVARIATE_QUANTILE_FORECAST_METHODS:
            params["output_types"] = ["quantiles", "mean"]
            if intervals is None:
                # This corresponds to quantiles: 0.05 to 0.95 in steps of 0.05.
                params["intervals"] = list(range(0, 100, 10))
            else:
                params["intervals"] = np.sort(intervals).tolist()

        return params

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
        num_samples: int,
        forecast_start_date: pd.Timestamp,
        item_id: Optional[str],
        info: Dict,
    ) -> Union[QuantileForecast, SampleForecast]:
        if self.method_name in UNIVARIATE_QUANTILE_FORECAST_METHODS:
            quantile_forecasts_dict = forecast_dict["quantiles"]

            return QuantileForecast(
                forecast_arrays=np.array(
                    list(quantile_forecasts_dict.values())
                ),
                forecast_keys=list(quantile_forecasts_dict.keys()),
                start_date=forecast_start_date,
                item_id=item_id,
            )
        else:
            if self.method_name in UNIVARIATE_POINT_FORECAST_METHODS:
                # Handling special cases outside of R is better,
                # since it is more visible and is easier to change.

                # Repeat mean forecasts `num_samples` times.
                samples = np.reshape(
                    forecast_dict["mean"] * num_samples,
                    (num_samples, self.prediction_length),
                )
            else:
                samples = np.array(forecast_dict["samples"])

            expected_shape = (
                num_samples,
                self.prediction_length,
            )
            assert (
                samples.shape == expected_shape
            ), f"Expected shape {expected_shape} but found {samples.shape}"

            return SampleForecast(
                samples,
                start_date=forecast_start_date,
                info=info,
                item_id=item_id,
            )
