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
        split_date=pd.Timestamp('2018-01-31', freq='D')
    )
    train, test = splitter.split_dataset(whole_dataset)

The module also supports rolling splits::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Timestamp('2018-01-31', freq='D')
    )
    train, test = splitter.rolling_split_dataset(whole_dataset, windows=7)
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import List, Optional

# Third-party imports
import pandas as pd
import pydantic

# First-party imports
from gluonts.dataset.common import TimeSeriesItem


class TimeSeriesSlice(pydantic.BaseModel):
    """Like TimeSeriesItem, but all time-related fields are of type pd.Series
    and is indexable, e.g `ts_slice['2018':]`.
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
    def from_time_series_item(
        cls, item: TimeSeriesItem, freq: Optional[str] = None
    ) -> "TimeSeriesSlice":
        if freq is None:
            freq = item.start.freq

        index = pd.date_range(
            start=item.start, freq=freq, periods=len(item.target)
        )

        feat_dynamic_cat = [
            pd.Series(cat, index=index) for cat in item.feat_dynamic_cat
        ]

        feat_dynamic_real = [
            pd.Series(real, index=index) for real in item.feat_dynamic_real
        ]

        return TimeSeriesSlice(
            target=pd.Series(item.target, index=index),
            item=item.item,
            feat_static_cat=item.feat_static_cat,
            feat_static_real=item.feat_static_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
        )

    def to_time_series_item(self) -> TimeSeriesItem:
        return TimeSeriesItem(
            start=self.start,
            target=self.target.values,
            item=self.item,
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
            feat_dynamic_cat=[cat.values for cat in self.feat_dynamic_cat],
            feat_dynamic_real=[real.values for real in self.feat_dynamic_real],
        )

    @property
    def start(self):
        return self.target.index[0]

    @property
    def end(self):
        return self.target.index[-1]

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, slice_: slice) -> "TimeSeriesSlice":
        feat_dynamic_real = None
        feat_dynamic_cat = None

        if self.feat_dynamic_real is not None:
            feat_dynamic_real = [feat for feat in self.feat_dynamic_real]

        if self.feat_dynamic_cat is not None:
            feat_dynamic_cat = [feat for feat in self.feat_dynamic_cat]

        return TimeSeriesSlice(
            item=self.item,
            target=self.target[slice_],
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
        )


class TrainTestSplit(pydantic.BaseModel):
    train: List[TimeSeriesItem] = []
    test: List[TimeSeriesItem] = []

    def _add_train_slice(self, train_slice: TimeSeriesSlice) -> None:
        # is there any data left for training?
        if train_slice:
            self.train.append(train_slice.to_time_series_item())

    def _add_test_slice(self, test_slice: TimeSeriesSlice) -> None:
        self.test.append(test_slice.to_time_series_item())


class AbstractBaseSplitter(ABC):
    """Base class for all other splitter.

    Args:
        param prediction_length:
            The prediction length which is used to train themodel.

        max_history:
            If given, all entries in the *test*-set have a max-length of
            `max_history`. This can be sued to produce smaller file-sizes.
    """

    # @property
    # @abstractmethod
    # def prediction_length(self) -> int:
    #     pass

    # @property
    # @abstractmethod
    # def max_history(self) -> Optional[int]:
    #     pass

    @abstractmethod
    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        pass

    @abstractmethod
    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        pass

    def _trim_history(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        if getattr(self, 'max_history') is not None:
            return item[: -getattr(self, 'max_history')]
        else:
            return item

    def split(self, items: List[TimeSeriesItem]) -> TrainTestSplit:
        split = TrainTestSplit()

        for item in map(TimeSeriesSlice.from_time_series_item, items):
            train = self._train_slice(item)
            test = self._trim_history(self._test_slice(item))

            split._add_train_slice(train)

            assert len(test) - len(train) >= getattr(self, 'prediction_length')
            split._add_test_slice(test)

        return split

    def rolling_split(
        self,
        items: List[TimeSeriesItem],
        windows: int,
        distance: Optional[int] = None,
    ) -> TrainTestSplit:
        # distance defaults to prediction_length
        if distance is None:
            distance = getattr(self, 'prediction_length')
        assert distance is not None

        split = TrainTestSplit()

        for item in map(TimeSeriesSlice.from_time_series_item, items):
            train = self._train_slice(item)
            split._add_train_slice(train)

            for window in range(windows):
                offset = window * distance
                test = self._trim_history(
                    self._test_slice(item, offset=offset)
                )

                assert len(test) - len(train) >= getattr(self, 'max_history')
                split._add_test_slice(test)

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
        return item[: self.split_date + self.prediction_length + offset]
