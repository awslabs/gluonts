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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast, Dict, Iterator, List, Optional, Union

import pandas as pd
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from toolz import valmap

from gluonts.dataset.common import DataEntry, ProcessDataEntry
from gluonts.dataset.field_names import FieldName


@dataclass
class PandasDataset:
    """
    A pandas.DataFrame-based dataset type.

    This class is constructed with a collection of pandas.DataFrame-objects
    where each ``DataFrame`` is representing one time series.
    A ``target`` and a ``timestamp`` columns are essential. Furthermore,
    static/dynamic real/categorical features can be specified.

    Parameters
    ----------
    dataframes
        Single ``pd.DataFrame``/``pd.Series`` or a collection as list or dict
        containing at least ``timestamp`` and ``target`` values.
        If a Dict is provided, the key will be the associated ``item_id``.
    target
        Name of the column that contains the ``target`` time series.
        For multivariate targets, a list of column names should be provided.
    timestamp
        Name of the column that contains the timestamp information.
    freq
        Frequency of observations in the time series. Must be a valid pandas
        frequency.
    feat_dynamic_real
        List of column names that contain dynamic real features.
    feat_dynamic_cat
        List of column names that contain dynamic categorical features.
    feat_static_real
        List of column names that contain static real features.
    feat_static_cat
        List of column names that contain static categorical features.
    past_feat_dynamic_real
        List of column names that contain dynamic real features only for the
        history.
    ignore_last_n_targets
        For target and past dynamic features last ``ignore_last_n_targets``
        elements are removed when iterating over the data set. This becomes
        important when the predictor is called.
    """

    dataframes: Union[
        pd.DataFrame,
        pd.Series,
        List[pd.DataFrame],
        List[pd.Series],
        Dict[str, pd.DataFrame],
        Dict[str, pd.Series],
    ]
    target: Union[str, List[str]] = "target"
    timestamp: Optional[str] = None
    freq: Optional[str] = None
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

        if is_series(self.dataframes):
            self.dataframes = series_to_dataframe(self.dataframes)
        # store data internally as List[Tuple[str, pandas.DataFrame]]
        # if str is not empty it will be set in ``DataEntry`` as ``item_id``.
        if isinstance(self.dataframes, dict):
            self._dataframes = list(self.dataframes.items())
        elif isinstance(self.dataframes, list):
            self._dataframes = [(None, df) for df in self.dataframes]
        else:  # case single dataframe
            self._dataframes = [(None, self.dataframes)]

        for i, (item_id, df) in enumerate(self._dataframes):
            if self.timestamp:
                df = df.set_index(keys=self.timestamp)

            if not isinstance(df.index, pd.PeriodIndex):
                df.index = pd.to_datetime(df.index)
                df = df.to_period(freq=self.freq)

            df.sort_index(inplace=True)

            assert is_uniform(df.index), (
                "Dataframe index is not uniformly spaced. "
                "If your dataframe contains data from multiple series in the "
                'same column ("long" format), consider constructing the '
                "dataset with `PandasDataset.from_long_dataframe` instead."
            )

            self._dataframes[i] = (item_id, df)

        if not self.freq:  # infer frequency from index
            self.freq = self._dataframes[0][1].index.freqstr

        self.process = ProcessDataEntry(
            cast(str, self.freq), one_dim_target=self.one_dim_target
        )

    def _dataentry(
        self, item_id: Optional[str], df: pd.DataFrame
    ) -> DataEntry:
        dataentry = as_dataentry(
            data=df,
            target=self.target,
            feat_dynamic_real=self.feat_dynamic_real,
            feat_dynamic_cat=self.feat_dynamic_cat,
            feat_static_real=self.feat_static_real,
            feat_static_cat=self.feat_static_cat,
            past_feat_dynamic_real=self.past_feat_dynamic_real,
        )
        if item_id is not None:
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
    ) -> "PandasDataset":
        """
        Construct ``PandasDataset`` out of a long dataframe.
        A long dataframe uses the long format for each variable. Target time
        series values, for example, are stacked on top of each other rather
        than side-by-side. The same is true for other dynamic or categorical
        features.

        Parameters
        ----------
        dataframe
            pandas.DataFrame containing at least ``timestamp``, ``target`` and
            ``item_id`` columns.
        item_id
            Name of the column that, when grouped by, gives the different time
            series.
        **kwargs
            Additional arguments. Same as of PandasDataset class.

        Returns
        -------
        PandasDataset
            Gluonts dataset based on ``pandas.DataFrame``s.
        """
        return cls(dataframes=dict(list(dataframe.groupby(item_id))), **kwargs)


def series_to_dataframe(
    series: Union[pd.Series, List[pd.Series], Dict[str, pd.Series]]
) -> Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]]:
    def to_df(series):
        assert isinstance(
            series.index, DatetimeIndexOpsMixin
        ), "series index has to be a DatetimeIndex."
        return series.to_frame(name="target")

    if isinstance(series, list):
        return list(map(to_df, series))
    elif isinstance(series, dict):
        return valmap(to_df, series)
    return to_df(series)


def is_series(series: Any) -> bool:
    """
    return True if ``series`` is ``pd.Series`` or a collection of
    ``pd.Series``.
    """
    if isinstance(series, list):
        return is_series(series[0])
    elif isinstance(series, dict):
        return is_series(list(series.values()))
    return isinstance(series, pd.Series)


def as_dataentry(
    data: pd.DataFrame,
    target: Union[str, List[str]],
    timestamp: Optional[str] = None,
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
    data
        pandas.DataFrame containing at least ``timestamp``, ``target`` and
        ``item_id`` columns.
    target
        Name of the column that contains the ``target`` time series.
        For multivariate targets ``target`` is expecting a list of column
        names.
    timestamp
        Name of the column that contains the timestamp information.
        If ``None`` the index of ``data`` is assumed to be the time.
    feat_dynamic_real
        List of column names that contain dynamic real features.
    feat_dynamic_cat
        List of column names that contain dynamic categorical features.
    feat_static_real
        List of column names that contain static real features.
    feat_static_cat
        List of column names that contain static categorical features.
    past_feat_dynamic_real
        List of column names that contain dynamic real features only for
        the history.

    Returns
    -------
    DataEntry
        A dictionary with at least ``target`` and ``start`` field.
    """
    start = data.loc[:, timestamp].iloc[0] if timestamp else data.index[0]
    dataentry = {FieldName.START: start}

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


def is_uniform(index: pd.PeriodIndex) -> bool:
    """
    Check if ``index`` contains monotonically increasing periods, evenly spaced
    with frequency ``index.freq``.

    >>> ts = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    >>> is_uniform(pd.DatetimeIndex(ts).to_period("2H"))
    True
    >>> ts = ["2021-01-01 00:00", "2021-01-01 04:00"]
    >>> is_uniform(pd.DatetimeIndex(ts).to_period("2H"))
    False
    """
    other = pd.period_range(index[0], periods=len(index), freq=index.freq)
    return (other == index).all()
