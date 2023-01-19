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

from typing import Any, Dict, Iterable, Union, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from toolz import first

from gluonts.dataset.common import DataEntry
from gluonts.itertools import SizedIterable, Map


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
    ignore_last_n_targets
        For target and past dynamic features last ``ignore_last_n_targets``
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
    static_features: Optional[pd.DataFrame] = None
    ignore_last_n_targets: int = 0
    unchecked: bool = False
    assume_sorted: bool = False
    _pairs: Iterable[Tuple[Any, Union[pd.Series, pd.DataFrame]]] = field(
        init=False
    )
    _static_reals: Optional[pd.DataFrame] = None
    _static_cats: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if isinstance(self.dataframes, dict):
            self._pairs = self.dataframes.items()
        elif isinstance(self.dataframes, (pd.Series, pd.DataFrame)):
            self._pairs = [pair_with_item_id(self.dataframes)]
        else:
            assert isinstance(self.dataframes, SizedIterable)
            self._pairs = Map(pair_with_item_id, self.dataframes)

        assert isinstance(self._pairs, SizedIterable)

        if self.freq is None:
            assert (
                self.timestamp is None
            ), "You need to provide `freq` along with `timestamp`"
            self.freq = infer_freq(first(self._pairs)[1].index)

        if self.static_features is not None:
            (
                self._static_reals,
                self._static_cats,
            ) = split_numerical_categorical(self.static_features)

        if self._static_reals is not None:
            self._static_reals = self._static_reals.astype(np.float32)

        if self._static_cats is not None:
            self._static_cats = category_to_int(self._static_cats)

        self._data_entries = Map(self._pair_to_dataentry, self._pairs)

    @property
    def num_feat_static_cat(self):
        return 0 if self._static_cats is None else self._static_cats.shape[1]

    @property
    def num_feat_static_real(self):
        return 0 if self._static_reals is None else self._static_reals.shape[1]

    @property
    def num_feat_dynamic_real(self):
        return (
            0
            if self.feat_dynamic_real is None
            else len(self.feat_dynamic_real)
        )

    @property
    def num_past_feat_dynamic_real(self):
        return (
            0
            if self.past_feat_dynamic_real is None
            else len(self.past_feat_dynamic_real)
        )

    @property
    def cardinalities(self):
        if self._static_cats is None:
            return []
        return [
            max(self._static_cats[c]) + 1 for c in self._static_cats.columns
        ]

    def _pair_to_dataentry(self, pair) -> DataEntry:
        item_id, df = pair

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
            "target": remove_last_n(
                self.ignore_last_n_targets,
                df[self.target].values.transpose(),
            ),
        }

        if item_id is not None:
            entry["item_id"] = item_id

        if self._static_cats is not None:
            entry["feat_static_cat"] = self._static_cats.loc[item_id].values

        if self._static_reals is not None:
            entry["feat_static_real"] = self._static_reals.loc[item_id].values

        if self.feat_dynamic_real is not None:
            entry["feat_dynamic_real"] = df[
                self.feat_dynamic_real
            ].values.transpose()

        if self.past_feat_dynamic_real is not None:
            entry["past_feat_dynamic_real"] = remove_last_n(
                self.ignore_last_n_targets,
                df[self.past_feat_dynamic_real].values.transpose(),
            )

        return entry

    def __iter__(self):
        yield from self._data_entries
        self.unchecked = True

    def __len__(self) -> int:
        return len(self._data_entries)

    def __str__(self) -> str:
        return (
            f"PandasDataset<"
            f"size={len(self)}, "
            f"freq={self.freq}, "
            f"num_dynamic_real={self.num_feat_dynamic_real}, "
            f"num_past_dynamic_real={self.num_past_feat_dynamic_real}, "
            f"num_static_real={self.num_feat_static_real}, "
            f"num_static_cat={self.num_feat_static_cat}, "
            f"cardinalities={self.cardinalities}>"
        )

    @classmethod
    def from_long_dataframe(
        cls, dataframe: pd.DataFrame, item_id: str, **kwargs
    ) -> "PandasDataset":
        """
        Construct ``PandasDataset`` out of a long dataframe. A long dataframe
        uses the long format for each variable. Target time series values, for
        example, are stacked on top of each other rather than side-by-side. The
        same is true for other dynamic or categorical features.

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
        if not isinstance(dataframe.index, DatetimeIndexOpsMixin):
            dataframe.index = pd.to_datetime(dataframe.index)
        return cls(dataframes=dataframe.groupby(item_id), **kwargs)


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


def remove_last_n(n: int, array: np.ndarray) -> np.ndarray:
    """
    Return a new array with last ``n`` elements removed from the
    trailing axis, if ``n`` is positive, and the array itself otherwise.
    """
    if n <= 0:
        return array
    return array[..., :-n]


def split_numerical_categorical(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Splits the given DataFrame into two: one containing numerical columns,
    the other containing categorical columns.

    Columns of other types are excluded.
    """
    numerical = {}
    categorical = {}

    for col in df.columns:
        if df[col].dtype == "category":
            categorical[col] = df[col]
        elif is_numeric_dtype(df[col]):
            numerical[col] = df[col]

    return (
        pd.DataFrame.from_dict(numerical) if len(numerical) else None,
        pd.DataFrame.from_dict(categorical) if len(categorical) else None,
    )


def category_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a ``DataFrame`` with categorical columns replaced by integer codes.

    Codes are obtained via ``pd.Categorical.codes``, and range from 0 to N,
    where N is the number of categories for the column of interest.
    """
    data = {
        col: (
            pd.Categorical(df[col]).codes
            if df[col].dtype == "category"
            else df[col]
        )
        for col in df.columns
    }
    return pd.DataFrame(data=data, index=df.index)


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
