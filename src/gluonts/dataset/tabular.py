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

from typing import Iterator, List, Union
from datetime import datetime

import pandas as pd

from gluonts.dataset.common import Dataset, DataEntry, ProcessDataEntry
from gluonts.dataset.field_names import FieldName


class DataFrameDataset(Dataset):
    """
    A pd.DataFrame-based dataset type.

    This dataset uses the long format for each variable. Target time series values,
    for example, are stacked on top of each other rather than side-by-side. The
    same is true for other dynamic or categorical features. An example of a dataframe
    is given below.


    Parameters
    ----------
    data: pd.DataFrame
        pd.DataFrame containing at least timestamp, target and item_id columns.
    target: str or List[str]
        Name of the column that contains the target time series.
        For multivariate targets `target` is expecting a list of column names.
    timestamp: str
        Name of the column that contains the timestamp information.
    item_id: str
        Name of the column that, when grouped by, gives the different time series.
    freq: str
        Frequency of observations in the time series. Must be a valid pandas frequency.
    feat_dynamic_real: List[str]
        List of column names that contain dynamic real features.
    feat_dynamic_cat: List[str]
        List of column names that contain dynamic categorical features.
    feat_static_real: List[str]
        List of column names that contain static real features.
    feat_static_cat: List[str]
        List of column names that contain static categorical features.
    past_feat_dynamic_real: List[str]
        List of column names that contain dynamic real features only for the history.
    prediction_length: int
        For target and past dynamic features last `prediction_length` elements are
        removed when iterating over the data set. This becomes important when the
        predictor is called.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        timestamp: str,
        item_id: str,
        freq: str,
        feat_dynamic_real: List[str] = None,
        feat_dynamic_cat: List[str] = None,
        feat_static_real: List[str] = None,
        feat_static_cat: List[str] = None,
        past_feat_dynamic_real: List[str] = None,
        prediction_length: int = None,
    ) -> None:
        self.data = data.copy()
        self.target = target
        self.timestamp = timestamp
        self.item_id = item_id
        self.freq = freq
        self.prediction_length = prediction_length
        self.feat_dynamic_real = feat_dynamic_real
        self.feat_dynamic_cat = feat_dynamic_cat
        self.feat_static_real = feat_static_real
        self.feat_static_cat = feat_static_cat
        self.past_feat_dynamic_real = past_feat_dynamic_real

        self.one_dim_target = not isinstance(self.target, list)
        self.process = ProcessDataEntry(
            freq, one_dim_target=self.one_dim_target
        )

    def _dataentry(self, item_id: str, df: pd.DataFrame) -> DataEntry:
        check_timestamps(df.loc[:, self.timestamp].to_list(), freq=self.freq)
        dataentry = dataframeTS_to_dataentry(
            data=df,
            target=self.target,
            timestamp=self.timestamp,
            feat_dynamic_real=self.feat_dynamic_real,
            feat_dynamic_cat=self.feat_dynamic_cat,
            feat_static_real=self.feat_static_real,
            feat_static_cat=self.feat_static_cat,
            past_feat_dynamic_real=self.past_feat_dynamic_real,
        )
        dataentry["item_id"] = item_id
        return self.process(dataentry)

    def __iter__(self) -> Iterator[DataEntry]:
        def func(x):
            dataentry = self._dataentry(x[0], x[1])
            if not self.prediction_length:
                return dataentry
            return prepare_prediction_data(dataentry, self.prediction_length)

        return map(func, self.data.groupby(self.item_id))

    def __len__(self) -> int:
        return len(self.data.loc[:, self.item_id].unique())


def dataframeTS_to_dataentry(
    data: pd.DataFrame,
    target: Union[str, List[str]],
    timestamp: str,
    feat_dynamic_real: List[str] = None,
    feat_dynamic_cat: List[str] = None,
    feat_static_real: List[str] = None,
    feat_static_cat: List[str] = None,
    past_feat_dynamic_real: List[str] = None,
) -> DataEntry:
    """
    Converts a single time series (uni- or multi-variate) that is given in
    a pd.DataFrame format to a DataEntry.

    Parameters
    ----------
    data: pd.DataFrame
        pd.DataFrame containing at least timestamp, target and item_id columns.
    target: str or List[str]
        Name of the column that contains the target time series.
        For multivariate targets `target` is expecting a list of column names.
    timestamp: str
        Name of the column that contains the timestamp information.
    feat_dynamic_real: List[str]
        List of column names that contain dynamic real features.
    feat_dynamic_cat: List[str]
        List of column names that contain dynamic categorical features.
    feat_static_real: List[str]
        List of column names that contain static real features.
    feat_static_cat: List[str]
        List of column names that contain static categorical features.
    past_feat_dynamic_real: List[str]
        List of column names that contain dynamic real features only for the history.

    Returns
    -------
    DataEntry
        A dictionary with at least `target` and `start` field.
    """
    dataentry = {
        FieldName.START: data.loc[:, timestamp].iloc[0],
    }

    def set_field(fieldname, col_names, f=lambda x: x):
        if isinstance(col_names, list):
            dataentry[fieldname] = [
                f(data.loc[:, n].to_list()) for n in col_names
            ]

    if isinstance(target, str):
        dataentry[FieldName.TARGET] = data.loc[:, target].to_list()
    else:
        set_field(FieldName.TARGET, target)
    set_field(FieldName.FEAT_DYNAMIC_REAL, feat_dynamic_real)
    set_field(FieldName.FEAT_DYNAMIC_CAT, feat_dynamic_cat)
    set_field(FieldName.FEAT_STATIC_REAL, feat_static_real, lambda x: x[0])
    set_field(FieldName.FEAT_STATIC_CAT, feat_static_cat, lambda x: x[0])
    set_field(FieldName.PAST_FEAT_DYNAMIC_REAL, past_feat_dynamic_real)
    return dataentry


def prepare_prediction_data(
    dataentry: DataEntry, prediction_length: int
) -> DataEntry:
    """
    Removes `prediction_length` values from `target` and `past_feat_dynamic_real`.
    Works in univariate and multivariate case.

    >>> prepare_prediction_data({"target": [1., 2., 3., 4.]}, prediction_length=2)
    {'target': [1.0, 2.0]}
    """

    def cut(
        x: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        if not isinstance(x[0], list):
            return x[:-prediction_length]
        elif len(x) == 1:
            return [cut(x[0])]  # type: ignore
        else:
            return [cut(x[0]), *cut(x[1:])]  # type: ignore

    dataentry[FieldName.TARGET] = cut(dataentry[FieldName.TARGET])
    if FieldName.PAST_FEAT_DYNAMIC_REAL in dataentry:
        dataentry[FieldName.PAST_FEAT_DYNAMIC_REAL] = cut(
            dataentry[FieldName.PAST_FEAT_DYNAMIC_REAL]
        )
    return dataentry


def check_timestamps(
    timestamps: List[Union[str, datetime]], freq: str
) -> None:
    """
    Checks if `timestamps` are monotonically increasing and evenly space with
    frequency `freq`.

    >>> timestamps = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    >>> check_timestamps(timestamps, freq="2H")
    """
    assert all(
        pd.to_datetime(timestamps)
        == pd.date_range(
            start=timestamps[0], freq=freq, periods=len(timestamps)
        )
    )
