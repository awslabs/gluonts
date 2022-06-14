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

"""
Train/test splitter
~~~~~~~~~~~~~~~~~~~

This module defines strategies to split a whole dataset into train and test
subsets.

For uniform datasets, where all time-series start and end at the same point in
time `OffsetSplitter` can be used::

    splitter = OffsetSplitter(prediction_length=24, split_offset=24)
    train, test = splitter.split(whole_dataset)

For all other datasets, the more flexible `DateSplitter` can be used::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Period('2018-01-31', freq='D')
    )
    train, test = splitter.split(whole_dataset)

The module also supports rolling splits::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Period('2018-01-31', freq='D')
    )
    train, test = splitter.rolling_split(whole_dataset, windows=7)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, cast

import numpy as np
import pandas as pd
import pydantic

from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName


class TimeSeriesSlice(pydantic.BaseModel):
    """
    Like DataEntry, but all time-related fields are of type pd.Series and is
    indexable, e.g `ts_slice['2018':]`.
    """

    class Config:
        arbitrary_types_allowed = True

    target: pd.Series
    item: str

    feat_static_cat: List[int] = []
    feat_static_real: List[float] = []

    feat_dynamic_cat: List[pd.Series] = []
    feat_dynamic_real: List[pd.Series] = []

    @classmethod
    def from_data_entry(
        cls, item: DataEntry, freq: Optional[str] = None
    ) -> "TimeSeriesSlice":
        if freq is None:
            freq = item["start"].freq

        index = pd.period_range(
            start=item["start"], freq=freq, periods=len(item["target"])
        )

        feat_dynamic_cat = [
            pd.Series(cat, index=index)
            for cat in list(item.get("feat_dynamic_cat", []))
        ]

        feat_dynamic_real = [
            pd.Series(real, index=index)
            for real in list(item.get("feat_dynamic_real", []))
        ]

        feat_static_cat = list(item.get("feat_static_cat", []))

        feat_static_real = list(item.get("feat_static_real", []))

        return TimeSeriesSlice(
            target=pd.Series(item["target"], index=index),
            item=item[FieldName.ITEM_ID],
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
        )

    def to_data_entry(self) -> DataEntry:
        ret = {
            FieldName.START: self.start,
            FieldName.ITEM_ID: self.item,
            FieldName.TARGET: self.target.values,
        }

        if self.feat_static_cat:
            ret[FieldName.FEAT_STATIC_CAT] = self.feat_static_cat
        if self.feat_static_real:
            ret[FieldName.FEAT_STATIC_REAL] = self.feat_static_real
        if self.feat_dynamic_cat:
            ret[FieldName.FEAT_DYNAMIC_CAT] = [
                cast(np.ndarray, cat.values).tolist()
                for cat in self.feat_dynamic_cat
            ]
        if self.feat_dynamic_real:
            ret[FieldName.FEAT_DYNAMIC_REAL] = [
                cast(np.ndarray, real.values).tolist()
                for real in self.feat_dynamic_real
            ]

        return ret

    @property
    def start(self) -> pd.Period:
        return self.target.index[0]

    @property
    def end(self) -> pd.Period:
        return self.target.index[-1]

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, slice_: slice) -> "TimeSeriesSlice":
        feat_dynamic_real = []
        feat_dynamic_cat = []

        if self.feat_dynamic_real is not None:
            feat_dynamic_real = [
                feat[slice_] for feat in self.feat_dynamic_real
            ]

        if self.feat_dynamic_cat is not None:
            feat_dynamic_cat = [feat[slice_] for feat in self.feat_dynamic_cat]

        target = self.target[slice_]

        assert all([len(target) == len(feat) for feat in feat_dynamic_real])
        assert all([len(target) == len(feat) for feat in feat_dynamic_cat])

        return TimeSeriesSlice(
            target=target,
            item=self.item,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
        )


class TrainTestSplit(pydantic.BaseModel):
    train: List[DataEntry] = []
    test: List[DataEntry] = []

    def _add_train_slice(self, train_slice: TimeSeriesSlice) -> None:
        # is there any data left for training?
        if train_slice:
            self.train.append(train_slice.to_data_entry())

    def _add_test_slice(self, test_slice: TimeSeriesSlice) -> None:
        self.test.append(test_slice.to_data_entry())


class AbstractBaseSplitter(ABC):
    """
    Base class for all other splitter.

    Args:
        param prediction_length:
            The prediction length which is used to train themodel.

        max_history:
            If given, all entries in the *test*-set have a max-length of
            `max_history`. This can be used to produce smaller file-sizes.
    """

    @abstractmethod
    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        pass

    @abstractmethod
    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        pass

    def _trim_history(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        if getattr(self, "max_history") is not None:
            return item[-getattr(self, "max_history") :]
        else:
            return item

    def split(self, items: List[DataEntry]) -> TrainTestSplit:
        split = TrainTestSplit()

        for item in map(TimeSeriesSlice.from_data_entry, items):

            train = self._train_slice(item)
            test = self._trim_history(self._test_slice(item))

            split._add_train_slice(train)

            prediction_length = getattr(self, "prediction_length")

            _check_split_length(train.end, test.end, prediction_length)
            split._add_test_slice(test)

        return split

    def rolling_split(
        self,
        items: List[DataEntry],
        windows: int,
        distance: Optional[int] = None,
    ) -> TrainTestSplit:
        # distance defaults to prediction_length
        if distance is None:
            distance = getattr(self, "prediction_length")
        assert distance is not None

        split = TrainTestSplit()

        for item in map(TimeSeriesSlice.from_data_entry, items):
            train = self._train_slice(item)
            split._add_train_slice(train)

            for window in range(windows):
                offset = window * distance
                test = self._trim_history(
                    self._test_slice(item, offset=offset)
                )
                prediction_length = getattr(self, "prediction_length")

                _check_split_length(train.end, test.end, prediction_length)
                split._add_test_slice(test)

        return split


def _check_split_length(
    train_end: pd.Period, test_end: pd.Period, prediction_length: int
) -> None:
    msg = (
        "Not enough observations after the split point to construct"
        " the test instance; consider using longer time series,"
        " or splitting at an earlier point."
    )
    assert train_end + prediction_length <= test_end, msg


class OffsetSplitter(pydantic.BaseModel, AbstractBaseSplitter):
    """
    A splitter that slices training and test data based on a fixed integer
    offset.

    Parameters
    ----------
    prediction_length
        Length of the prediction interval in test data.
    split_offset
        Offset determining where the training data ends.
        A positive offset indicates how many observations since the start of
        each series should be in the training slice; a negative offset
        indicates how many observations before the end of each series should
        be excluded from the training slice. Please make sure that the number
        of excluded values is enough for the test case, i.e., at least
        ``prediction_length`` (for ``rolling_split`` multiple of
        ``prediction_length``) values are left off.
    max_history
        If given, all entries in the *test*-set have a max-length of
        `max_history`. This can be used to produce smaller file-sizes.
    """

    prediction_length: int
    split_offset: int
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        return item[: self.split_offset]

    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        offset_ = self.split_offset + offset + self.prediction_length
        if self.split_offset < 0 and offset_ >= 0:
            offset_ += len(item)
        return item[:offset_]


class DateSplitter(AbstractBaseSplitter, pydantic.BaseModel):
    """
    A splitter that slices training and test data based on a
    ``pandas.Period``.

    Training entries obtained from this class will be limited to observations
    up to (including) the given ``split_date``.

    Parameters
    ----------
    prediction_length
        Length of the prediction interval in test data.
    split_date
        Period determining where the training data ends. Please make sure
        at least ``prediction_length`` (for ``rolling_split`` multiple of
        ``prediction_length``) values are left over after the ``split_date``.
    max_history
        If given, all entries in the *test*-set have a max-length of
        `max_history`. This can be used to produce smaller file-sizes.
    """

    class Config:
        arbitrary_types_allowed = True

    prediction_length: int
    split_date: pd.Period
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        # the train-slice includes everything up to (including) the split date
        return item[: self.split_date]

    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        return item[
            : self.split_date
            + (self.prediction_length + offset) * item.start.freq
        ]
