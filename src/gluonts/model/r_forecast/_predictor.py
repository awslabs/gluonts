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

import os
from pathlib import Path
from typing import Dict, Optional, List, Union, Iterator

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality

# https://stackoverflow.com/questions/25329955/check-if-r-is-installed-from-python
from subprocess import Popen, PIPE

proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
R_IS_INSTALLED = proc.wait() == 0

try:
    import rpy2.robjects.packages as rpackages
    from rpy2 import rinterface, robjects
    from rpy2.rinterface import RRuntimeError
except ImportError as e:
    rpy2_error_message = str(e)
    RPY2_IS_INSTALLED = False
else:
    RPY2_IS_INSTALLED = True

USAGE_MESSAGE = """
The RForecastPredictor is a thin wrapper for calling the R forecast package.
In order to use it you need to install R and run

pip install 'rpy2>=2.9.*,<3.*'

R -e 'install.packages(c("forecast", "nnfor"),\
repos="https://cloud.r-project.org")'
"""

SAMPLE_FORECAST_METHODS = ["ets", "arima"]
QUANTILE_FORECAST_METHODS = ["tbats", "thetaf", "stlar"]
POINT_FORECAST_METHODS = ["croston", "mlp"]
SUPPORTED_METHODS = (
    SAMPLE_FORECAST_METHODS
    + QUANTILE_FORECAST_METHODS
    + POINT_FORECAST_METHODS
)


class RForecastPredictor(RepresentablePredictor):
    """
    Wrapper for calling the `R forecast package.

    <http://pkg.robjhyndman.com/forecast/>`_.

    The `RForecastPredictor` is a thin wrapper for calling the R forecast
    package.  In order to use it you need to install R and run::

        pip install 'rpy2>=2.9.*,<3.*'
        R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

    Parameters
    ----------
    freq
        The granularity of the time series (e.g. '1H')
    prediction_length
        Number of time points to be predicted.
    method
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
        Note that currently only `output_type = 'samples'` is supported.
    """  # noqa: E501

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        method_name: str = "ets",
        period: Optional[int] = None,
        trunc_length: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        if not R_IS_INSTALLED:
            raise ImportError("R is not Installed! \n " + USAGE_MESSAGE)

        if not RPY2_IS_INSTALLED:
            raise ImportError(rpy2_error_message + USAGE_MESSAGE)

        self._robjects = robjects
        self._rinterface = rinterface
        self._rinterface.initr()
        self._rpackages = rpackages

        this_dir = os.path.dirname(os.path.realpath(__file__))
        this_dir = this_dir.replace("\\", "/")  # for windows
        r_files = [
            n[:-2] for n in os.listdir(f"{this_dir}/R/") if n[-2:] == ".R"
        ]

        for n in r_files:
            try:
                path = Path(this_dir, "R", f"{n}.R")
                robjects.r(f'source("{path}")'.replace("\\", "\\\\"))
            except RRuntimeError as er:
                raise RRuntimeError(str(er) + USAGE_MESSAGE) from er

        assert method_name in SUPPORTED_METHODS, (
            f"method {method_name} is not supported please use one of"
            f" {SUPPORTED_METHODS}"
        )

        self.method_name = method_name

        self._stats_pkg = rpackages.importr("stats")
        self._r_method = robjects.r[method_name]

        self.prediction_length = prediction_length
        self.period = period if period is not None else get_seasonality(freq)
        self.trunc_length = trunc_length

        self.params = {
            "prediction_length": self.prediction_length,
            "output_types": ["samples"],
            "frequency": self.period,
        }
        if params is not None:
            self.params.update(params)

    def _unlist(self, l):
        if type(l).__name__.endswith("Vector"):
            return [self._unlist(x) for x in l]
        else:
            return l

    def _run_r_forecast(self, d, params, save_info):
        buf = []

        def save_to_buf(x):
            buf.append(x)

        def dont_save(x):
            pass

        f = save_to_buf if save_info else dont_save

        # save output from the R console in buf
        self._rinterface.set_writeconsole_regular(f)
        self._rinterface.set_writeconsole_warnerror(f)

        make_ts = self._stats_pkg.ts
        r_params = self._robjects.vectors.ListVector(params)
        vec = self._robjects.FloatVector(d["target"])
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

            # Post-processing quantiles on then Python side for the convenience
            # of asserting and debugging.
            upper_quantiles = [
                str(from_interval_to_level(interval, side="upper"))
                for interval in params["intervals"]
            ]

            lower_quantiles = [
                str(from_interval_to_level(interval, side="lower"))
                for interval in params["intervals"]
            ]

            # Median forecasts would be available at two places: Lower 0 and
            # Higher 0 (0-prediction interval)
            forecast_dict["quantiles"] = dict(
                zip(
                    lower_quantiles + upper_quantiles[1:],
                    forecast_dict["lower_quantiles"]
                    + forecast_dict["upper_quantiles"][1:],
                )
            )

            # `QuantileForecast` allows "mean" as the key; we store them as
            # well since they can differ from median.
            forecast_dict["quantiles"].update(
                {"mean": forecast_dict.pop("mean")}
            )

        self._rinterface.set_writeconsole_regular(
            self._rinterface.consolePrint
        )
        self._rinterface.set_writeconsole_warnerror(
            self._rinterface.consolePrint
        )
        return forecast_dict, buf

    def predict(
        self,
        dataset: Dataset,
        num_samples: int = 100,
        intervals: Optional[List] = None,
        save_info: bool = False,
        **kwargs,
    ) -> Iterator[Union[SampleForecast, QuantileForecast]]:
        if self.method_name in POINT_FORECAST_METHODS:
            print(
                "Overriding `output_types` to `mean` since"
                f" {self.method_name} is a point forecast method."
            )
        elif self.method_name in QUANTILE_FORECAST_METHODS:
            print(
                "Overriding `output_types` to `quantiles` since "
                f"{self.method_name} is a quantile forecast method."
            )

        for data in dataset:
            if self.trunc_length:
                shift_by = max(data["target"].shape[0] - self.trunc_length, 0)
                data["start"] = data["start"] + shift_by
                data["target"] = data["target"][-self.trunc_length :]

            params = self.params.copy()
            params["num_samples"] = num_samples

            if self.method_name in POINT_FORECAST_METHODS:
                params["output_types"] = ["mean"]
            elif self.method_name in QUANTILE_FORECAST_METHODS:
                params["output_types"] = ["quantiles", "mean"]
                if intervals is None:
                    # This corresponds to quantiles: 0.05 to 0.95 in steps of
                    # 0.05.
                    params["intervals"] = list(range(0, 100, 10))
                else:
                    params["intervals"] = np.sort(intervals).tolist()

            forecast_dict, console_output = self._run_r_forecast(
                data, params, save_info=save_info
            )

            if self.method_name in QUANTILE_FORECAST_METHODS:
                quantile_forecasts_dict = forecast_dict["quantiles"]

                yield QuantileForecast(
                    forecast_arrays=np.array(
                        list(quantile_forecasts_dict.values())
                    ),
                    forecast_keys=list(quantile_forecasts_dict.keys()),
                    start_date=forecast_start(data),
                    item_id=data.get("item_id", None),
                )
            else:
                if self.method_name in POINT_FORECAST_METHODS:
                    # Handling special cases outside of R is better, since it
                    # is more visible and is easier to change. Repeat mean
                    # forecasts `num_samples` times.
                    samples = np.reshape(
                        forecast_dict["mean"] * params["num_samples"],
                        (params["num_samples"], self.prediction_length),
                    )
                else:
                    samples = np.array(forecast_dict["samples"])

                expected_shape = (
                    params["num_samples"],
                    self.prediction_length,
                )
                assert (
                    samples.shape == expected_shape
                ), f"Expected shape {expected_shape} but found {samples.shape}"
                info = (
                    {"console_output": "\n".join(console_output)}
                    if save_info
                    else None
                )
                yield SampleForecast(
                    samples,
                    forecast_start(data),
                    info=info,
                    item_id=data.get("item_id", None),
                )
