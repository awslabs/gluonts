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

from typing import Dict, Union, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from toolz import first

from gluonts.dataset.common import DataEntry
from gluonts.itertools import SizedIterable

# NOTE
# - feat_static_real -> gone, use dedicated dataframe for static features
# - feat_static_cat -> gone, same as above
# - feat_dynamic_cat -> gone, too painful to do now, let's think about it later
# - TODO have dtype somewhere?
# - TODO have assertions for stuff being the right dtype?
# - TODO add the "ignore_last_n_targets" in some way (I don't like it)
# - TODO benchmark
# - TODO adjust tests, especially add tests for static features


@dataclass
class PandasDataset:
    dataframes: Union[
        pd.DataFrame,
        pd.Series,
        List[pd.DataFrame],
        List[pd.Series],
        Dict[str, pd.DataFrame],
        Dict[str, pd.Series],
    ]
    target: Union[str, List[str]] = "target"  # TODO meh
    feat_dynamic_real: Optional[List[str]] = None
    timestamp: Optional[str] = None
    freq: Optional[str] = None
    static_features: Optional[pd.DataFrame] = None
    unchecked: bool = True
    assume_sorted: bool = False
    _static_reals: Optional[pd.DataFrame] = None
    _static_cats: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if isinstance(self.dataframes, dict):
            pass
        elif isinstance(self.dataframes, (pd.DataFrame, pd.Series)):
            self.dataframes = {0: self.dataframes}
        elif isinstance(self.dataframes, SizedIterable):
            self.dataframes = dict(enumerate(self.dataframes))

        if self.freq is None:
            assert (
                self.timestamp is None
            ), "You need to provide `freq` along with `timestamp`"
            self.freq = infer_freq(first(self.dataframes.items())[1].index)

        if self.static_features is not None:
            (
                self._static_reals,
                self._static_cats,
            ) = split_numerical_categorical(self.static_features)

        if self._static_reals is not None:
            self._static_reals = self._static_reals.astype(np.float32)

        if self._static_cats is not None:
            self._static_cats = category_to_int(self._static_cats)

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
    def cardinalities(self):
        if self._static_cats is None:
            return []
        return [
            max(self._static_cats[c]) + 1 for c in self._static_cats.columns
        ]

    def __len__(self) -> int:
        return len(self.dataframes)

    def __str__(self) -> str:
        return (
            f"PandasDataset<"
            f"size={len(self)}, "
            f"freq={self.freq}, "
            f"num_dynamic_real={self.num_feat_dynamic_real}, "
            f"num_static_real={self.num_feat_static_real}, "
            f"num_static_cat={self.num_feat_static_cat}, "
            f"cardinalities={self.cardinalities}>"
        )

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

        start = df.index[0]
        target = df[self.target].values.transpose()

        entry = {
            "item_id": item_id,
            "start": start,
            "target": target,
        }

        if self._static_cats is not None:
            entry["feat_static_cat"] = self._static_cats.loc[item_id].values

        if self._static_reals is not None:
            entry["feat_static_real"] = self._static_reals.loc[item_id].values

        if self.feat_dynamic_real is not None:
            entry["feat_dynamic_real"] = df[
                self.feat_dynamic_real
            ].values.transpose()

        return entry

    def __iter__(self):
        for pair in self.dataframes.items():
            yield self._pair_to_dataentry(pair)

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
        return cls(
            dataframes={k: v for k, v in dataframe.groupby(item_id)}, **kwargs
        )


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


def split_numerical_categorical(df: pd.DataFrame):
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


def category_to_int(df):
    """
    Turns categorical columns from the given DataFrame into integer ones.
    """
    for col in df.columns:
        if df[col].dtype == "category":
            df[col] = pd.Categorical(df[col]).codes
    return df


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
