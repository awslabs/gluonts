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

from typing import Dict, Optional, Union, List

import mxnet as mx
import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.model.forecast import (
    Forecast,
    SampleForecast,
    Quantile,
    QuantileForecast,
)
from gluonts.mx.distribution import Distribution


class DistributionForecast(Forecast):
    """
    A `Forecast` object that uses a GluonTS distribution directly.
    This can for instance be used to represent marginal probability
    distributions for each time point -- although joint distributions are
    also possible, e.g. when using MultiVariateGaussian).

    Parameters
    ----------
    distribution
        Distribution object. This should represent the entire prediction
        length, i.e., if we draw `num_samples` samples from the distribution,
        the sample shape should be

            samples = trans_dist.sample(num_samples)
            samples.shape -> (num_samples, prediction_length)

    start_date
        start of the forecast
    freq
        forecast frequency
    info
        additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.
    """

    @validated()
    def __init__(
        self,
        distribution: Distribution,
        start_date: pd.Timestamp,
        freq: str,
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ) -> None:
        self.distribution = distribution
        self.shape = (
            self.distribution.batch_shape + self.distribution.event_shape
        )
        self.prediction_length = self.shape[0]
        self.item_id = item_id
        self.info = info

        assert isinstance(
            start_date, pd.Timestamp
        ), "start_date should be a pandas Timestamp object"
        self.start_date = start_date

        assert isinstance(freq, str), "freq should be a string"
        self.freq = freq
        self._mean = None

    @property
    def mean(self) -> np.ndarray:
        """
        Forecast mean.
        """
        if self._mean is not None:
            return self._mean
        else:
            self._mean = self.distribution.mean.asnumpy()
            return self._mean

    @property
    def mean_ts(self) -> pd.Series:
        """
        Forecast mean, as a pandas.Series object.
        """
        return pd.Series(self.mean, index=self.index)

    def quantile(self, level: Union[float, str]) -> np.ndarray:
        level = Quantile.parse(level).value
        q = self.distribution.quantile(mx.nd.array([level])).asnumpy()[0]
        return q

    def to_sample_forecast(self, num_samples: int = 200) -> SampleForecast:
        return SampleForecast(
            samples=self.distribution.sample(num_samples),
            start_date=self.start_date,
            freq=self.freq,
            item_id=self.item_id,
            info=self.info,
        )

    def to_quantile_forecast(self, quantiles: List[Union[float, str]]):
        return QuantileForecast(
            forecast_arrays=np.array([self.quantile(q) for q in quantiles]),
            forecast_keys=quantiles,
            start_date=self.start_date,
            freq=self.freq,
            item_id=self.item_id,
            info=self.info,
        )
