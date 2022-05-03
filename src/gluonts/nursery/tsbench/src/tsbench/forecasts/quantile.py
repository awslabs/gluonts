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

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import numpy.typing as npt
import pandas as pd
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality


@dataclass
class QuantileForecasts:
    """
    A type-safe wrapper for a list of quantile forecasts, stored as NumPy
    arrays.
    """

    values: npt.NDArray[
        np.float32
    ]  # [num_time_series, num_quantiles, prediction_length]
    start_dates: npt.NDArray[np.float32]
    item_ids: npt.NDArray[np.float32]
    freq: pd.DateOffset
    quantiles: list[str]

    @property
    def prediction_length(self) -> int:
        """
        Returns the prediction length of the quantile forecasts.
        """
        return self.values.shape[-1]

    @property
    def seasonality(self) -> int:
        """
        Returns the seasonality of the forecasts (i.e. how many steps to go
        back to arrive at the value of the previous period).
        """
        return get_seasonality(self.freq.freqstr)  # type: ignore

    # ---------------------------------------------------------------------------------------------
    # DATA ACCESS

    def get(self, index: int) -> QuantileForecast:
        """
        Returns the quantile forecast at the specified index.

        This method should typically only be used for visualizing single
        forecasts.
        """
        return QuantileForecast(
            forecast_arrays=self.values[index],
            start_date=pd.Timestamp(self.start_dates[index], freq=self.freq),
            freq=self.freq.freqstr,  # type: ignore
            item_id=self.item_ids[index],
            forecast_keys=self.quantiles,
        )

    @property
    def median(self) -> npt.NDArray[np.float32]:
        """
        Returns the median forecasts for all time series.

        NumPy array of shape [N, T] (N: number of forecasts, T: forecast
        horizon).
        """
        i = self.quantiles.index("0.5")
        return self.values[:, i]

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: npt.ArrayLike) -> QuantileForecasts:  # type: ignore
        return QuantileForecasts(
            values=self.values[index],
            start_dates=self.start_dates[index],
            item_ids=self.item_ids[index],
            freq=self.freq,
            quantiles=self.quantiles,
        )

    # ---------------------------------------------------------------------------------------------
    # STORAGE

    @classmethod
    def load(cls, path: Path) -> QuantileForecasts:
        """
        Loads the quantile forecasts from the specified path.

        Args:
            path: The path from where to load the forecasts.
        """
        try:
            with (path / "values.npy").open("rb") as f:
                values = np.load(f)
        except FileNotFoundError:
            # This is a bug in the publicly available evaluations for MQ-RNN.
            # Should not be called otherwise.
            with (path / "values.npz").open("rb") as f:
                values = np.load(f)
        with (path / "metadata.npz").open("rb") as f:
            metadata = np.load(f, allow_pickle=True).item()
        return QuantileForecasts(
            values=values,
            start_dates=metadata["start_dates"],
            item_ids=metadata["item_ids"],
            freq=metadata["freq"],
            quantiles=metadata["quantiles"],
        )

    def save(self, path: Path) -> None:
        """
        Saves the forecasts to the specified path.

        Args:
            path: The path of the file where to save the forecasts to.
        """
        assert path.is_dir()

        if not path.exists():
            path.mkdir()
        with (path / "values.npy").open("wb+") as f:
            np.save(f, self.values)
        with (path / "metadata.npz").open("wb+") as f:
            np.save(
                f,
                dict(
                    start_dates=self.start_dates,
                    item_ids=self.item_ids,
                    freq=self.freq,
                    quantiles=self.quantiles,
                ),
            )
