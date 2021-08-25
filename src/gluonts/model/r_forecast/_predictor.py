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
from typing import Dict, Iterator, Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import forecast_start
from gluonts.time_feature import get_seasonality

USAGE_MESSAGE = """
The RForecastPredictor is a thin wrapper for calling the R forecast package.
In order to use it you need to install R and run

pip install 'rpy2>=2.9.*,<3.*'

R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'
"""


class RForecastPredictor(RepresentablePredictor):
    """
    Wrapper for calling the `R forecast package
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
        super().__init__(freq=freq, prediction_length=prediction_length)

        try:
            import rpy2.robjects.packages as rpackages
            from rpy2 import rinterface, robjects
            from rpy2.rinterface import RRuntimeError
        except ImportError as e:
            raise ImportError(str(e) + USAGE_MESSAGE) from e

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

        supported_methods = [
            "ets",
            "arima",
            "tbats",
            "croston",
            "mlp",
            "thetaf",
        ]
        assert (
            method_name in supported_methods
        ), f"method {method_name} is not supported please use one of {supported_methods}"

        self.method_name = method_name

        self._stats_pkg = rpackages.importr("stats")
        self._r_method = robjects.r[method_name]

        self.prediction_length = prediction_length
        self.freq = freq
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
        # FOR NOW ONLY SAMPLES...
        # if "quantiles" in forecast_dict:
        #     forecast_dict["quantiles"] = dict(zip(params["quantiles"], forecast_dict["quantiles"]))

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
        save_info: bool = False,
        **kwargs,
    ) -> Iterator[SampleForecast]:
        for entry in dataset:
            if isinstance(entry, dict):
                data = entry
            else:
                data = entry.data
                if self.trunc_length:
                    data = data[-self.trunc_length :]

            params = self.params.copy()
            params["num_samples"] = num_samples

            forecast_dict, console_output = self._run_r_forecast(
                data, params, save_info=save_info
            )

            samples = np.array(forecast_dict["samples"])
            expected_shape = (params["num_samples"], self.prediction_length)
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
                self.freq,
                info=info,
                item_id=entry.get("item_id", None),
            )
