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
    split = splitter.split(whole_dataset)

For all other datasets, the more flexible `DateSplitter` can be used::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Timestamp('2018-01-31', freq='D')
    )
    split = splitter.split(whole_dataset)

The module also supports rolling splits::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Timestamp('2018-01-31', freq='D')
    )
    split = splitter.rolling_split(whole_dataset, windows=7)
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
import pydantic

from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.support.pandas import frequency_add as ts_add

import pandas as pd
import numpy as np


class TimeSeriesSlice:
    def __init__(
        self,
        freq,
        index,
        target,
        feat_static_cat,
        feat_static_real,
        feat_dynamic_cat,
        feat_dynamic_real,
    ):
        self.freq = freq
        self.index = index
        self.target = target
        self.feat_static_cat = feat_static_cat
        self.feat_static_real = feat_static_real
        self.feat_dynamic_cat = feat_dynamic_cat
        self.feat_dynamic_real = feat_dynamic_real

    @classmethod
    def from_data_entry(cls, item: dict) -> "TimeSeriesSlice":
        freq = item["start"].freq
        index = pd.date_range(
            start=item["start"],
            freq=freq,
            periods=len(item["target"]),
        )

        feat_dynamic_cat = np.atleast_2d(item.get("feat_dynamic_cat", []))
        feat_dynamic_real = np.atleast_2d(item.get("feat_dynamic_real", []))

        feat_static_cat = np.asarray(item.get("feat_static_cat", []))
        feat_static_real = np.asarray(item.get("feat_static_real", []))

        return TimeSeriesSlice(
            freq=freq,
            index=np.asarray(index),
            target=np.asarray(item["target"]),
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
        )

    def _get_idx(self, val):
        if val is None:
            return val

        if isinstance(val, int):
            return val

        return np.searchsorted(self.index, val)

    def __getitem__(self, s):
        start = s.start
        stop = s.stop

        if isinstance(start, pd.Timestamp):
            start = start.to_numpy()

        if isinstance(stop, pd.Timestamp):
            stop = stop.to_numpy()

        slice_ = slice(self._get_idx(start), self._get_idx(stop))

        return TimeSeriesSlice(
            freq=self.freq,
            index=self.index[slice_],
            target=self.target[slice_],
            feat_dynamic_cat=self.feat_dynamic_cat[slice_],
            feat_dynamic_real=self.feat_dynamic_real[slice_],
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
        )

    def __len__(self):
        return len(self.index)

    def to_data_entry(self) -> DataEntry:
        ret = {
            FieldName.START: self.start,
            # FieldName.ITEM_ID: self.item,
            FieldName.TARGET: self.target,
        }

        if self.feat_static_cat:
            ret[FieldName.FEAT_STATIC_CAT] = self.feat_static_cat
        if self.feat_static_real:
            ret[FieldName.FEAT_STATIC_REAL] = self.feat_static_real
        if self.feat_dynamic_cat:
            ret[FieldName.FEAT_DYNAMIC_CAT] = [
                cat.values.tolist() for cat in self.feat_dynamic_cat
            ]
        if self.feat_dynamic_real:
            ret[FieldName.FEAT_DYNAMIC_REAL] = [
                real.values.tolist() for real in self.feat_dynamic_real
            ]

        return ret

    @property
    def start(self):
        return pd.Timestamp(self.index[0], freq=self.freq)

    @property
    def end(self):
        return pd.Timestamp(self.index[-1], freq=self.freq)


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
    """Base class for all other splitter.

    Args:
        param prediction_length:
            The prediction length which is used to train themodel.

        max_history:
            If given, all entries in the *test*-set have a max-length of
            `max_history`. This can be used to produce smaller file-sizes.
    """

    prediction_length: int
    max_history: Optional[int]

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

            # assert train.end + train.end.freq * prediction_length <= test.end

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
            distance = self.prediction_length
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
                split._add_test_slice(test)

                assert ts_add(train.end, self.prediction_length) <= test.end

        return split


class OffsetSplitter(pydantic.BaseModel, AbstractBaseSplitter):
    "Requires uniform data."

    prediction_length: int
    split_offset: int
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        return item[: self.split_offset]

    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        offset_ = self.split_offset + offset + self.prediction_length
        assert offset_ <= len(item)
        return item[:offset_]


class DateSplitter(AbstractBaseSplitter, pydantic.BaseModel):
    prediction_length: int
    split_date: pd.Timestamp
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        # the train-slice includes everything up to (including) the split date
        return item[: self.split_date]

    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        return item[: ts_add(self.split_date, self.prediction_length + offset)]
