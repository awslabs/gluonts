# Standard library imports
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.evaluation import get_seasonality
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor

ForecastResult = namedtuple("ForecastResult", "forecast output")


class RForecastPredictor(RepresentablePredictor):
    """
    Wrapper for calling the `R forecast package
    <http://pkg.robjhyndman.com/forecast/>`_.

    The `RForecastPredictor` is a thin wrapper for calling the R forecast
    package.  In order to use it you need to install R and run::

        pip install rpy2
        R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

    Parameters
    ----------
    method
        The method from rforecast to be used one of
        "ets", "arima", "tbats", "croston", "mlp".
    prediction_length
        Number of time points to be predicted.
    freq
        The granularity of the time series (e.g. '1H')
    period
        The period to be used (this is called `frequency` in the R forecast
        package), result to a tentative reasonable default if not specified
        (for instance 24 for hourly freq '1H')
    num_samples
        Number of samples to draw.
    trunc_length
        Maximum history length to feed to the model (some models become slow
        with very long series).
    params
        Parameters to be used when calling the forecast method default.
        Note that currently only `output_type = 'samples'` is supported.
    """

    supported_methods = frozenset({"ets", "arima", "tbats", "croston", "mlp"})

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        method_name: str = "ets",
        period: int = None,
        num_eval_samples: int = 100,
        trunc_length: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> None:

        assert (
            method_name in self.supported_methods
        ), f"method {method_name} is not supported please use one of {self.supported_methods}"

        self.r = RAPI.get()
        self.method = self.r.obj[method_name]
        self.stats_ts = self.r.obj.packages.importr('stats').ts

        self.freq = freq
        self.trunc_length = trunc_length

        params = params if params is not None else {}

        self.params = dict(
            prediction_length=self.prediction_length,
            output_types=['samples'],
            num_samples=self.num_samples,
            frequency=period if period is not None else get_seasonality(freq),
            **params,
        )

    def _unlist(self, l):
        if l.__class__.__name__.endswith("Vector"):
            return list(map(self._unlist, l))
        else:
            return l

    def _forecast(self, target, params, save_info: bool):
        with self.r.capture_output(save_info) as console_output:
            # FOR NOW ONLY SAMPLES...
            # if "quantiles" in forecast_dict:
            #     forecast_dict["quantiles"] = dict(zip(params["quantiles"], forecast_dict["quantiles"]))
            target = self.r.obj.FloatVector(target)
            params = self.r.obj.vectors.ListVector(params)

            ts = self.stats_ts(target, frequency=self.period)
            forecast = self.method(ts, params)

            return ForecastResult(
                dict(zip(forecast.names, map(self._unlist, forecast))),
                console_output,
            )

    def _target(self, data):
        target = data["target"]
        if self.trunc_length:
            return target[-self.trunc_length :]
        else:
            return target

    def _params(self, num_samples):
        params = dict(self.params)
        if num_samples is not None:
            params['num_samples'] = num_samples

        return params

    def _forecast_start(self, data):
        start = pd.Timestamp(data['start'], freq=self.freq)
        return start + len(data["target"])

    def predict(
        self, dataset: Dataset, num_samples=None, save_info=False, **kwargs
    ) -> Iterator[SampleForecast]:
        for entry in dataset:
            assert isinstance(entry, dict)

            target = self._target(entry)
            params = self._params(num_samples)
            result = self._forecast(target, params, save_info)
            samples = np.array(result.forecast['samples'])

            expected_shape = params['num_samples'], params['prediction_length']
            assert (
                samples.shape == expected_shape
            ), f"Expected shape {expected_shape} but found {samples.shape}"

            if result.output is not None:
                info = {'console_output': '\n'.join(result.output)}
            else:
                info = None
            forecast_start = self._forecast_start(entry)
            yield SampleForecast(
                samples, forecast_start, forecast_start.freqstr, info=info
            )
