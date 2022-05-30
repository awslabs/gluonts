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
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, List, Union, Dict

import pandas as pd

from gluonts.dataset.common import Dataset, DataEntry, ProcessDataEntry
from gluonts.dataset.field_names import FieldName


@dataclass
class DataFramesDataset(Dataset):
    """
    A pandas.DataFrame-based dataset type.

    This class is constructed with a collection of pandas.DataFrame-objects
    where each ``DataFrame`` is representing one time series.
    A ``target`` and a ``timestamp`` columns are essential. Furthermore,
    static/dynamic real/categorical features can be specified.

    Parameters
    ----------
    dataframes: Union[
        List[pandas.DataFrame], Dict[str, pandas.DataFrame], pandas.DataFrame
    ]
        List or Dict of ``DataFrame``s or a single ``DataFrame`` containing
        at least ``timestamp`` and ``target`` columns. If a Dict is provided,
        the key will be the associated ``item_id``.
    target: str or List[str]
        Name of the column that contains the ``target`` time series.
        For multivariate targets, a list of column names should be provided.
    timestamp: str
        Name of the column that contains the timestamp information.
    freq: str
        Frequency of observations in the time series. Must be a valid pandas
        frequency.
    feat_dynamic_real: List[str]
        List of column names that contain dynamic real features.
    feat_dynamic_cat: List[str]
        List of column names that contain dynamic categorical features.
    feat_static_real: List[str]
        List of column names that contain static real features.
    feat_static_cat: List[str]
        List of column names that contain static categorical features.
    past_feat_dynamic_real: List[str]
        List of column names that contain dynamic real features only for the
        history.
    ignore_last_n_targets: int
        For target and past dynamic features last ``ignore_last_n_targets``
        elements are removed when iterating over the data set. This becomes
        important when the predictor is called.
    """

    dataframes: Union[
        List[pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame
    ]
    target: Union[str, List[str]]
    timestamp: str
    freq: str
    feat_dynamic_real: List[str] = field(default_factory=list)
    feat_dynamic_cat: List[str] = field(default_factory=list)
    feat_static_real: List[str] = field(default_factory=list)
    feat_static_cat: List[str] = field(default_factory=list)
    past_feat_dynamic_real: List[str] = field(default_factory=list)
    ignore_last_n_targets: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.target, list) and len(self.target) == 1:
            self.target = self.target[0]
        self.one_dim_target = not isinstance(self.target, list)
        self.process = ProcessDataEntry(
            self.freq, one_dim_target=self.one_dim_target
        )
        # store data internally as List[Tuple[str, pandas.DataFrame]]
        # if str is not empty it will be set in ``DataEntry`` as ``item_id``.
        if isinstance(self.dataframes, dict):
            self._dataframes = list(self.dataframes.items())
        elif isinstance(self.dataframes, list):
            self._dataframes = [("", df) for df in self.dataframes]
        else:  # case single dataframe
            self._dataframes = [("", self.dataframes)]

    def _dataentry(self, item_id: str, df: pd.DataFrame) -> DataEntry:
        assert check_timestamps(
            df.loc[:, self.timestamp].to_list(), freq=self.freq
        ), "``timestamps`` are not monotonically increasing or evenly spaced."
        dataentry = as_dataentry(
            data=df,
            target=self.target,
            timestamp=self.timestamp,
            feat_dynamic_real=self.feat_dynamic_real,
            feat_dynamic_cat=self.feat_dynamic_cat,
            feat_static_real=self.feat_static_real,
            feat_static_cat=self.feat_static_cat,
            past_feat_dynamic_real=self.past_feat_dynamic_real,
        )
        if item_id:
            dataentry["item_id"] = item_id
        return dataentry

    def __iter__(self) -> Iterator[DataEntry]:
        for item_id, df in self._dataframes:
            dataentry = self.process(self._dataentry(item_id, df))
            if self.ignore_last_n_targets:
                dataentry = prepare_prediction_data(
                    dataentry, self.ignore_last_n_targets
                )
            yield dataentry

    def __len__(self) -> int:
        return len(self._dataframes)

    @classmethod
    def from_long_dataframe(
        cls, dataframe: pd.DataFrame, item_id: str, **kwargs
    ) -> DataFramesDataset:
        """
        Construct ``DataFramesDataset`` out of a long dataframe.
        A long dataframe uses the long format for each variable. Target time
        series values, for example, are stacked on top of each other rather
        than side-by-side. The same is true for other dynamic or categorical
        features.

        Parameters
        ----------
        dataframe: pandas.DataFrame
            pandas.DataFrame containing at least ``timestamp``, ``target`` and
            ``item_id`` columns.
        item_id: str
            Name of the column that, when grouped by, gives the different time
            series.
        **kwargs:
            Additional arguments. Same as of DataFramesDataset class.

        Returns
        -------
        DataFramesDataset
            Gluonts dataset based on ``pandas.DataFrame``s.
        """
        return cls(dataframes=dict(list(dataframe.groupby(item_id))), **kwargs)


def as_dataentry(
    data: pd.DataFrame,
    target: Union[str, List[str]],
    timestamp: str,
    feat_dynamic_real: List[str] = [],
    feat_dynamic_cat: List[str] = [],
    feat_static_real: List[str] = [],
    feat_static_cat: List[str] = [],
    past_feat_dynamic_real: List[str] = [],
) -> DataEntry:
    """
    Convert a single time series (uni- or multi-variate) that is given in
    a pandas.DataFrame format to a DataEntry.

    Parameters
    ----------
    data: pandas.DataFrame
        pandas.DataFrame containing at least ``timestamp``, ``target`` and
        ``item_id`` columns.
    target: str or List[str]
        Name of the column that contains the ``target`` time series.
        For multivariate targets ``target`` is expecting a list of column
        names.
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
        List of column names that contain dynamic real features only for
        the history.

    Returns
    -------
    DataEntry
        A dictionary with at least ``target`` and ``start`` field.
    """
    dataentry = {
        FieldName.START: data.loc[:, timestamp].iloc[0],
    }

    def set_field(fieldname, col_names, f=lambda x: x):
        if col_names:
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
    dataentry: DataEntry, ignore_last_n_targets: int
) -> DataEntry:
    """
    Remove ``ignore_last_n_targets`` values from ``target`` and
    ``past_feat_dynamic_real``.  Works in univariate and multivariate case.

    >>> prepare_prediction_data(
    >>>    {"target": np.array([1., 2., 3., 4.])}, ignore_last_n_targets=2
    >>> )
    {'target': array([1., 2.])}
    """
    entry = deepcopy(dataentry)
    for fname in [FieldName.TARGET, FieldName.PAST_FEAT_DYNAMIC_REAL]:
        if fname in entry:
            entry[fname] = entry[fname][..., :-ignore_last_n_targets]
    return entry


def check_timestamps(
    timestamps: List[Union[str, datetime]], freq: str
) -> bool:
    """
    Check if ``timestamps`` are monotonically increasing and evenly space with
    frequency ``freq``.

    >>> ts = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    >>> check_timestamps(ts, freq="2H")
    True
    """
    return all(
        pd.to_datetime(timestamps)
        == pd.date_range(
            start=timestamps[0], freq=freq, periods=len(timestamps)
        )
    )
