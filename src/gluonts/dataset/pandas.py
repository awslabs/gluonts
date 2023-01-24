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

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from toolz import first

from gluonts.dataset.common import DataEntry
from gluonts.itertools import Map, StarMap, SizedIterable


@dataclass
class PandasDataset:
    """
    A dataset type based on ``pandas.DataFrame``.

    This class is constructed with a collection of ``pandas.DataFrame``
    objects where each ``DataFrame`` is representing one time series.
    Both ``target`` and ``timestamp`` columns are essential. Dynamic features
    of a series can be specified with together with the series' ``DataFrame``,
    while static features can be specified in a separate ``DataFrame`` object
    via the ``static_features`` argument.

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
    past_feat_dynamic_real
        List of column names that contain dynamic real features only available
        in the past.
    static_features
        ``pd.DataFrame`` containing static features for the series. The index
        should contain the key of the series in the ``dataframes`` argument.
    future_length
        For target and past dynamic features last ``future_length``
        elements are removed when iterating over the data set.
    unchecked
        Whether consistency checks on indexes should be skipped.
        (Default: ``False``)
    assume_sorted
        Whether to assume that indexes are sorted by time, and skip sorting.
        (Default: ``False``)
    """

    dataframes: Union[
        pd.DataFrame,
        pd.Series,
        Iterable[pd.DataFrame],
        Iterable[pd.Series],
        Iterable[Tuple[Any, pd.DataFrame]],
        Iterable[Tuple[Any, pd.Series]],
        Dict[str, pd.DataFrame],
        Dict[str, pd.Series],
    ]
    target: Union[str, List[str]] = "target"
    feat_dynamic_real: Optional[List[str]] = None
    past_feat_dynamic_real: Optional[List[str]] = None
    timestamp: Optional[str] = None
    freq: Optional[str] = None
    static_features: pd.DataFrame = pd.DataFrame()
    future_length: int = 0
    unchecked: bool = False
    assume_sorted: bool = False
    dtype: Type = np.float32
    _static_reals: pd.DataFrame = field(init=False)
    _static_cats: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if isinstance(self.dataframes, dict):
            pairs = self.dataframes.items()
        elif isinstance(self.dataframes, (pd.Series, pd.DataFrame)):
            pairs = [(None, self.dataframes)]
        else:
            assert isinstance(self.dataframes, SizedIterable)
            pairs = Map(pair_with_item_id, self.dataframes)

        self._data_entries = StarMap(self._pair_to_dataentry, pairs)

        if self.feat_dynamic_real is None:
            self.feat_dynamic_real = pd.DataFrame()

        if self.past_feat_dynamic_real is None:
            self.past_feat_dynamic_real = pd.DataFrame()

        if self.freq is None:
            assert (
                self.timestamp is None
            ), "You need to provide `freq` along with `timestamp`"
            self.freq = infer_freq(first(pairs)[1].index)

        self._static_reals = (
            self.static_features.select_dtypes("number").astype(self.dtype).T
        )

        self._static_cats = (
            self.static_features.select_dtypes("category")
            .apply(lambda col: col.cat.codes)
            .T
        )

    @property
    def num_feat_static_cat(self):
        return len(self._static_cats)

    @property
    def num_feat_static_real(self):
        return len(self._static_reals)

    @property
    def num_feat_dynamic_real(self):
        return len(self.feat_dynamic_real)

    @property
    def num_past_feat_dynamic_real(self):
        return len(self.past_feat_dynamic_real)

    def _pair_to_dataentry(self, item_id, df) -> DataEntry:
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.target)

        if self.timestamp:
            df.index = pd.PeriodIndex(df[self.timestamp], freq=self.freq)

        if not isinstance(df.index, pd.PeriodIndex):
            df = df.to_period(freq=self.freq)

        if not self.assume_sorted:
            df.sort_index(inplace=True)

        if not self.unchecked:
            assert is_uniform(df.index), (
                "Dataframe index is not uniformly spaced. "
                "If your dataframe contains data from multiple series in the "
                'same column ("long" format), consider constructing the '
                "dataset with `PandasDataset.from_long_dataframe` instead."
            )

        entry = {
            "start": df.index[0],
        }

        target = df[self.target].values.T
        target = target[: len(target) - self.future_length]
        entry["target"] = target

        if item_id is not None:
            entry["item_id"] = item_id

        if self.num_feat_static_cat:
            entry["feat_static_cat"] = self._static_cats[item_id].values

        if self.num_feat_static_real:
            entry["feat_static_real"] = self._static_reals[item_id].values

        if self.num_feat_dynamic_real:
            entry["feat_dynamic_real"] = df[self.feat_dynamic_real].values.T

        if self.num_past_feat_dynamic_real:
            past_feat_dynamic_real = df[self.past_feat_dynamic_real].values.T
            entry["past_feat_dynamic_real"] = past_feat_dynamic_real[
                : len(past_feat_dynamic_real) - self.future_length
            ]

        return entry

    def __iter__(self):
        yield from self._data_entries
        self.unchecked = True

    def __len__(self) -> int:
        return len(self._data_entries)

    def __str__(self) -> str:
        info = ", ".join(
            [
                f"size={len(self)}",
                f"freq={self.freq}",
                f"num_feat_dynamic_real={self.num_feat_dynamic_real}",
                f"num_past_feat_dynamic_real={self.num_past_feat_dynamic_real}",
                f"num_feat_static_real={self.num_feat_static_real}",
                f"num_feat_static_cat={self.num_feat_static_cat}",
            ]
        )
        return f"PandasDataset<{info}>"

    @classmethod
    def from_long_dataframe(
        cls,
        dataframe: pd.DataFrame,
        item_id: str,
        static_feature_columns: Optional[List[str]] = None,
        static_features: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> "PandasDataset":
        """
        Construct ``PandasDataset`` out of a long data frame.

        A long dataframe contains time series data (both the target series and
        covariates) about multiple items at once. An ``item_id`` column is used
        to distinguish the items and ``group_by`` accordingly.

        Static features can be included in the long data frame as well (with
        constant value), or be given as a separate data frame indexed by the
        ``item_id`` values.

        Parameters
        ----------
        dataframe
            pandas.DataFrame containing at least ``timestamp``, ``target`` and
            ``item_id`` columns.
        item_id
            Name of the column that, when grouped by, gives the different time
            series.
        static_feature_columns
            Columns in ``dataframe`` containing static features.
        static_features
            Dedicated ``DataFrame`` for static features. If both ``static_features``
            and ``static_feature_columns`` are specified, then the two sets of features
            are appended together.
        **kwargs
            Additional arguments. Same as of PandasDataset class.

        Returns
        -------
        PandasDataset
            Gluonts dataset based on ``pandas.DataFrame``s.
        """
        if not isinstance(dataframe.index, DatetimeIndexOpsMixin):
            dataframe.index = pd.to_datetime(dataframe.index)

        if static_feature_columns is not None:
            other_static_features = (
                dataframe[[item_id] + static_feature_columns]
                .drop_duplicates()
                .set_index(item_id)
            )
            assert len(other_static_features) == len(
                dataframe[item_id].unique()
            )
        else:
            other_static_features = pd.DataFrame()

        return cls(
            dataframes=dataframe.groupby(item_id),
            static_features=pd.concat(
                [static_features, other_static_features], axis=1
            ),
            **kwargs,
        )


def pair_with_item_id(obj: Union[Tuple, pd.DataFrame, pd.Series]):
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return (None, obj)
    raise ValueError("input must be a pair, or a pandas Series or DataFrame.")


def infer_freq(index: pd.Index) -> str:
    if isinstance(index, pd.PeriodIndex):
        return index.freqstr

    freq = pd.infer_freq(index)
    # pandas likes to infer the `start of x` frequency, however when doing
    # df.to_period("<x>S"), it fails, so we avoid using it. It's enough to
    # remove the trailing S, e.g `MS` -> `M
    if len(freq) > 1 and freq.endswith("S"):
        return freq[:-1]

    return freq


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
