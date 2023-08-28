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
from typing import List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TimeSeries:
    """
    A time series contains the time series data along with metadata about the time series as well
    as static and dynamic features.
    """

    dataset_name: str
    start_date: Optional[pd.Timestamp]
    values: torch.Tensor
    item_id: Optional[str] = None
    feat_static_cat: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------------------------------------
    @property
    def end_date(self) -> pd.Timestamp:
        return self.start_date + (self.__len__() - 1) * self.start_date.freq

    @property
    def mean(self) -> torch.Tensor:
        return torch.mean(self.values, dim=0)

    @property
    def std(self) -> torch.Tensor:
        return torch.std(self.values, dim=0)

    def standardize(self, m: torch.Tensor, std: torch.Tensor):
        return TimeSeries(
            dataset_name=self.dataset_name,
            item_id=self.item_id,
            start_date=self.start_date,
            values=(self.values - m) / std,
            feat_static_cat=self.feat_static_cat,
            scale=torch.cat([m, std]),
        )

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, sequence: slice) -> TimeSeries:
        assert (
            sequence.start >= 0
        ), "Time series cannot be sliced prior to start."
        assert (
            sequence.step is None or sequence.step == 1
        ), "Time series cannot be sliced with a step size other than 1."

        return TimeSeries(
            dataset_name=self.dataset_name,
            item_id=self.item_id,
            start_date=self.start_date + sequence.start * self.start_date.freq,
            values=self.values[sequence],
            feat_static_cat=self.feat_static_cat,
            scale=self.scale,
        )


class TimeSeriesDataset(Dataset[TimeSeries]):
    """
    A dataset which provides time series.
    """

    def __init__(
        self,
        series: List[TimeSeries],
        prediction_length: int,
        freq: str,
        standardize: bool = True,
    ):
        """
        Args:
            series: The multivariate time series.
        """
        self.series = series
        self.prediction_length = prediction_length
        self.freq = freq
        self.standardize = standardize
        if self.standardize:
            self.means = torch.stack([s.mean for s in self.series], dim=0)
            self.stds = torch.stack([s.std for s in self.series], dim=0)

    def rescale_dataset(self, series: torch.Tensor):
        """
        Redo standardization. The series must contain the same time series in the same order as the dataset.
        """
        return (
            (series * self.stds.unsqueeze(2)) + self.means.unsqueeze(2)
            if self.standardize
            else series
        )

    @property
    def number_of_time_steps(self):
        return sum([len(s) for s in self.series])

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> TimeSeries:
        if self.standardize:
            return self.series[index].standardize(
                self.means[index], self.stds[index]
            )
        return self.series[index]
