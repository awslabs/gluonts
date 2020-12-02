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

from typing import Dict, Optional, Union, cast

import mxnet as mx
import numpy as np
import pandas as pd
from pandas import to_timedelta

from gluonts.model.forecast import Config, Forecast, OutputType


class PointProcessSampleForecast(Forecast):
    """
    Sample forecast object used for temporal point process inference.
    Differs from standard forecast objects as it does not implement
    fixed length samples. Each sample has a variable length, that is
    kept in a separate :code:`valid_length` attribute.

    Importantly, PointProcessSampleForecast does not implement some
    methods (such as :code:`quantile` or :code:`plot`) that are available
    in discrete time forecasts.

    Parameters
    ----------
    samples
        A multidimensional array of samples, of shape
        (number_of_samples, max_pred_length, target_dim). The target_dim is
        equal to 2, where the first dimension contains the inter-arrival times
        and the second - categorical marks.
    valid_length
        An array of integers denoting the valid lengths of each sample
        in :code:`samples`. That is, :code:`valid_length[0] == 2` implies
        that only the first two entries of :code:`samples[0, ...]` are
        valid "points".
    start_date
        Starting timestamp of the sample
    freq
        The time unit of interarrival times
    prediction_interval_length
        The length of the prediction interval for which samples were drawn.
    item_id
        Item ID, if available.
    info
        Optional dictionary of additional information.
    """

    prediction_interval_length: float

    # not used
    prediction_length = cast(int, None)
    mean = None
    _index = None

    def __init__(
        self,
        samples: Union[mx.nd.NDArray, np.ndarray],
        valid_length: Union[mx.nd.NDArray, np.ndarray],
        start_date: pd.Timestamp,
        freq: str,
        prediction_interval_length: float,
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ) -> None:
        assert isinstance(
            samples, (np.ndarray, mx.nd.NDArray)
        ), "samples should be either a numpy or an mxnet array"
        assert (
            samples.ndim == 2 or samples.ndim == 3
        ), f"samples should be a 2-dimensional or 3-dimensional array. Dimensions found: {samples.ndim}"

        assert isinstance(
            valid_length, (np.ndarray, mx.nd.NDArray)
        ), "samples should be either a numpy or an mxnet array"
        assert (
            valid_length.ndim == 1
        ), "valid_length should be a 1-dimensional array"
        assert (
            valid_length.shape[0] == samples.shape[0]
        ), "valid_length and samples should have compatible dimensions"

        self.samples, self.valid_length = (
            x if isinstance(x, np.ndarray) else x.asnumpy()
            for x in (samples, valid_length)
        )

        self._dim = samples.ndim
        self.item_id = item_id
        self.info = info

        assert isinstance(
            start_date, pd.Timestamp
        ), "start_date should be a pandas Timestamp object"
        self.start_date = start_date

        assert isinstance(freq, str), "freq should be a string"
        self.freq = freq

        assert (
            prediction_interval_length > 0
        ), "prediction_interval_length must be greater than 0"
        self.prediction_interval_length = prediction_interval_length

        self.end_date = (
            start_date
            + to_timedelta(1, self.freq) * prediction_interval_length
        )

    def dim(self) -> int:
        return self._dim

    @property
    def index(self) -> pd.DatetimeIndex:
        raise AttributeError(
            "Datetime index not defined for point process samples"
        )

    def as_json_dict(self, config: "Config") -> dict:
        result = super().as_json_dict(config)

        if OutputType.samples in config.output_types:
            result["samples"] = self.samples.tolist()
            result["valid_length"] = self.valid_length.tolist()

        return result

    def __repr__(self):
        return ", ".join(
            [
                f"PointProcessSampleForecast({self.samples!r})",
                f"{self.valid_length!r}",
                f"{self.start_date!r}",
                f"{self.end_date!r}",
                f"{self.freq!r}",
                f"item_id={self.item_id!r}",
                f"info={self.info!r})",
            ]
        )

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        raise NotImplementedError(
            "Quantile function is not defined for point process samples"
        )

    def plot(self, **kwargs):
        raise NotImplementedError(
            "Plotting not implemented for point process samples"
        )
