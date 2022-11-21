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
from typing import (
    Any,
    cast,
    Dict,
    Iterator,
    Iterable,
    List,
    Optional,
    Union,
    Tuple,
)

import pandas as pd
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from toolz import first

from gluonts.dataset.common import DataEntry, ProcessDataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Map, SizedIterable


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
    timestamp: Optional[str] = None
    freq: Optional[str] = None
    feat_dynamic_real: List[str] = field(default_factory=list)
    feat_dynamic_cat: List[str] = field(default_factory=list)
    feat_static_real: List[str] = field(default_factory=list)
    feat_static_cat: List[str] = field(default_factory=list)
    past_feat_dynamic_real: List[str] = field(default_factory=list)
    ignore_last_n_targets: int = 0
    unchecked: bool = False
    assume_sorted: bool = False
    _pairs: Iterable[Tuple[Any, Union[pd.Series, pd.DataFrame]]] = field(
        init=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.target, list) and len(self.target) == 1:
            self.target = self.target[0]
        self.one_dim_target = not isinstance(self.target, list)

        if isinstance(self.dataframes, dict):
            self._pairs = self.dataframes.items()
        elif isinstance(self.dataframes, (pd.Series, pd.DataFrame)):
            self._pairs = [pair_with_item_id(self.dataframes)]
        else:
            assert isinstance(self.dataframes, SizedIterable)
            self._pairs = Map(pair_with_item_id, self.dataframes)

        assert isinstance(self._pairs, SizedIterable)

        if self.freq is None:
            self.freq = infer_freq(first(self._pairs)[1].index)

        self.process = ProcessDataEntry(
            cast(str, self.freq), one_dim_target=self.one_dim_target
        )

        self._data_entries = Map(self._pair_to_dataentry, self._pairs)

    def _pair_to_dataentry(
        self, pair: Tuple[Any, Union[pd.Series, pd.DataFrame]]
    ) -> DataEntry:
        item_id, df = pair

        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.target)

        if self.timestamp:
            df.index = pd.PeriodIndex(df[self.timestamp], freq=self.freq)

        if not self.assume_sorted:
            df.sort_index(inplace=True)

        if not self.unchecked:
            if not isinstance(df.index, pd.PeriodIndex):
                df = df.to_period(freq=self.freq)
            assert is_uniform(df.index), (
                "Dataframe index is not uniformly spaced. "
                "If your dataframe contains data from multiple series in the "
                'same column ("long" format), consider constructing the '
                "dataset with `PandasDataset.from_long_dataframe` instead."
            )

        data_entry = as_dataentry(
            data=df,
            target=self.target,
            feat_dynamic_real=self.feat_dynamic_real,
            feat_dynamic_cat=self.feat_dynamic_cat,
            feat_static_real=self.feat_static_real,
            feat_static_cat=self.feat_static_cat,
            past_feat_dynamic_real=self.past_feat_dynamic_real,
        )

        if item_id is not None:
            data_entry["item_id"] = item_id

        return self.process(data_entry)

    def __iter__(self) -> Iterator[DataEntry]:
        for entry in self._data_entries:
            if self.ignore_last_n_targets:
                entry = prepare_prediction_data(
                    entry, self.ignore_last_n_targets
                )
            yield entry

        self.unchecked = True

    def __len__(self) -> int:
        return len(self._data_entries)

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
        if not isinstance(dataframe.index, DatetimeIndexOpsMixin):
            dataframe.index = pd.to_datetime(dataframe.index)
        return cls(dataframes=dataframe.groupby(item_id), **kwargs)


def pair_with_item_id(obj: Union[Tuple, pd.DataFrame, pd.Series]):
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return (None, obj)
    raise ValueError("input must be a pair, or a pandas Series or DataFrame.")


def infer_freq(index: pd.Index):
    if isinstance(index, pd.PeriodIndex):
        return index.freqstr
    return pd.infer_freq(index)


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
        if len(col_names) > 0:
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
