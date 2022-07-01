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
        split_date=pd.Period('2018-01-31', freq='D'), 
        windows=7
    )
    train, test = splitter.split(whole_dataset)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydantic

from gluonts.dataset import Dataset, DataEntry
from gluonts.dataset.field_names import FieldName


class TimeSeriesSlice(pydantic.BaseModel):
    """
    Like DataEntry, but all time-related fields are of type pd.Series and is
    indexable, e.g `ts_slice['2018':]`.
    """

    class Config:
        arbitrary_types_allowed = True

    target: pd.Series
    item_id: Optional[str] = None

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

        item_id = item.get(FieldName.ITEM_ID, None)

        return TimeSeriesSlice(
            target=pd.Series(item["target"], index=index),
            item_id=item_id,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
        )

    def to_data_entry(self) -> DataEntry:
        ret = {
            FieldName.START: self.start,
            FieldName.TARGET: self.target.values,
        }

        if self.item_id:
            ret[FieldName.ITEM_ID] = self.item_id
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
            item_id=self.item_id,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
        )


class AbstractBaseSplitter(ABC):
    """
    Base class for all other splitter.
    """

    @abstractmethod
    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        pass

    @abstractmethod
    def _test_slice(
        self, item: TimeSeriesSlice, prediction_length: int, offset: int = 0
    ) -> Tuple[TimeSeriesSlice, TimeSeriesSlice]:
        pass

    def _trim_history(
        self, item: TimeSeriesSlice, max_history: Optional[int]
    ) -> TimeSeriesSlice:
        if max_history is not None:
            return item[-max_history:]
        else:
            return item

    def split(self, dataset: Dataset):
        test_data = TestTemplate(dataset=dataset, splitter=self)
        train_data = TrainingDataset(dataset=dataset, splitter=self)

        return train_data, test_data

    def _generate_train_slices(self, items: List[DataEntry]):
        for item in map(TimeSeriesSlice.from_data_entry, items):
            train = self._train_slice(item)

            yield train.to_data_entry()

    def _generate_test_slices(
        self,
        items: Dataset,
        prediction_length: int,
        windows: int = 1,
        distance: Optional[int] = None,
        max_history: Optional[int] = None,
    ) -> Generator[Tuple[DataEntry, DataEntry], None, None]:
        if distance is None:
            distance = prediction_length

        for item in map(TimeSeriesSlice.from_data_entry, items):
            train = self._train_slice(item)

            for window in range(windows):
                offset = window * distance
                test = self._test_slice(
                    item, prediction_length=prediction_length, offset=offset
                )

                _check_split_length(
                    train.end, test[1].end, train.end.freq * prediction_length
                )

                input = self._trim_history(
                    test[0], max_history
                ).to_data_entry()

                label = test[1].to_data_entry()

                yield input, label


def _check_split_length(
    train_end: pd.Period, test_end: pd.Period, prediction_length: int
) -> None:
    msg = (
        "Not enough observations after the split point to construct"
        " the test instance; consider using longer time series,"
        " or splitting at an earlier point."
    )
    assert train_end + prediction_length <= test_end, msg


@dataclass
class TestTemplate:
    """
    A class used for generating test data.

    Parameters
    ----------
    dataset:
        Whole dataset used for testing.
    splitter:
        A specific splitter that knows how to slices training and
        test data.
    """

    dataset: Dataset
    splitter: AbstractBaseSplitter

    def generate_instances(
        self,
        prediction_length: int,
        windows=1,
        distance=None,
        max_history=None,
    ) -> Generator[Tuple[DataEntry, DataEntry], None, None]:
        """
        Generate an iterator of test dataset, which includes input part and
        label part.

        Parameters
        ----------
        prediction_length
            Length of the prediction interval in test data.
        windows
            Indicates how many test windows to generate for each original
            dataset entry.
        distance
            This is rather the difference between the start of each test
            window generated, for each of the original dataset entries.
        max_history
            If given, all entries in the *test*-set have a max-length of
            `max_history`. This can be used to produce smaller file-sizes.
        """

        yield from self.splitter._generate_test_slices(
            self.dataset,
            prediction_length=prediction_length,
            windows=windows,
            distance=distance,
            max_history=max_history,
        )


@dataclass
class TrainingDataset:
    dataset: Iterable[DataEntry]
    splitter: AbstractBaseSplitter

    def __iter__(self):
        return self.splitter._generate_train_slices(self.dataset)


@dataclass
class OffsetSplitter(AbstractBaseSplitter):
    """
    A splitter that slices training and test data based on a fixed integer
    offset.

    Parameters
    ----------
    split_offset
        Offset determining where the training data ends.
        A positive offset indicates how many observations since the start of
        each series should be in the training slice; a negative offset
        indicates how many observations before the end of each series should
        be excluded from the training slice. Please make sure that the number
        of excluded values is enough for the test case, i.e., at least
        ``prediction_length`` (for ``rolling_split`` multiple of
        ``prediction_length``) values are left off.
    """

    split_offset: int

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        return item[: self.split_offset]

    def _test_slice(
        self, item: TimeSeriesSlice, prediction_length: int, offset: int = 0
    ) -> Tuple[TimeSeriesSlice, TimeSeriesSlice]:
        offset_ = self.split_offset + offset
        if self.split_offset < 0 and offset_ >= 0:
            offset_ += len(item)
        if offset_ + prediction_length:
            return (
                item[:offset_],
                item[offset_ : offset_ + prediction_length],
            )
        else:
            return (item[:offset_], item[offset_:])


@dataclass
class DateSplitter(AbstractBaseSplitter):
    """
    A splitter that slices training and test data based on a
    ``pandas.Period``.

    Training entries obtained from this class will be limited to observations
    up to (including) the given ``split_date``.

    Parameters
    ----------
    split_date
        Period determining where the training data ends. Please make sure
        at least ``prediction_length`` (for ``rolling_split`` multiple of
        ``prediction_length``) values are left over after the ``split_date``.
    """

    split_date: pd.Period

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        # the train-slice includes everything up to (including) the split date
        return item[: self.split_date]

    def _test_slice(
        self, item: TimeSeriesSlice, prediction_length: int, offset: int = 0
    ) -> Tuple[TimeSeriesSlice, TimeSeriesSlice]:
        return (
            item[: self.split_date + offset * item.start.freq],
            item[
                self.split_date
                + (offset + 1) * item.start.freq : self.split_date
                + (prediction_length + offset) * item.start.freq
            ],
        )


def split(dataset, *, offset=None, date=None):
    # You need to provide `offset` or `date`, but not both
    assert (offset is None) != (date is None)
    if offset is not None:
        splitter = OffsetSplitter(offset)
    else:
        splitter = DateSplitter(date)
    return splitter.split(dataset)
