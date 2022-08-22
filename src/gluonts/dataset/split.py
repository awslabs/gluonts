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

.. testsetup:: *

    import pandas as pd
    import numpy as np
    from gluonts.dataset.split import OffsetSplitter, DateSplitter
    whole_dataset = [
        {"start": pd.Period("2018-01-01", freq="D"), "target": np.arange(50)},
        {"start": pd.Period("2018-01-01", freq="D"), "target": np.arange(50)},
    ]

This module defines strategies to split a whole dataset into train and test
subsets. The :func:`split` function can also be used to trigger their logic.

For uniform datasets, where all time-series start and end at the same point in
time :class:`OffsetSplitter` can be used::

.. testcode::

    splitter = OffsetSplitter(offset=7)
    train, test_template = splitter.split(whole_dataset)

For all other datasets, the more flexible :class:`DateSplitter` can be used::

.. testcode::

    splitter = DateSplitter(
        date=pd.Period('2018-01-31', freq='D')
    )
    train, test_template = splitter.split(whole_dataset)

In the above examples, the ``train`` output is a regular ``Dataset`` that can
be used for training purposes; ``test_template`` can generate test instances
as follows::

.. testcode::

    test_dataset = test_template.generate_instances(
        prediction_length=7,
        windows=2,
    )

The ``windows`` argument controls how many test windows to generate from each
entry in the original dataset. Each window will begin after the split point,
and so will not contain any training data. By default, windows are
non-overlapping, but this can be controlled with the ``distance`` optional
argument.

.. testcode::

    test_dataset = test_template.generate_instances(
        prediction_length=7,
        windows=2,
        distance=3, # windows are three time steps apart from each other
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, Generator, List, Optional, Tuple

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

    def split(
        self, dataset: Dataset
    ) -> Tuple["TrainingDataset", "TestTemplate"]:
        return (
            TrainingDataset(dataset=dataset, splitter=self),
            TestTemplate(dataset=dataset, splitter=self),
        )

    def _generate_train_slices(
        self, dataset: Dataset
    ) -> Generator[DataEntry, None, None]:
        for entry in map(TimeSeriesSlice.from_data_entry, dataset):
            train = self._train_slice(entry)

            yield train.to_data_entry()

    def _generate_test_slices(
        self,
        dataset: Dataset,
        prediction_length: int,
        windows: int = 1,
        distance: Optional[int] = None,
        max_history: Optional[int] = None,
    ) -> Generator[Tuple[DataEntry, DataEntry], None, None]:
        if distance is None:
            distance = prediction_length

        for entry in map(TimeSeriesSlice.from_data_entry, dataset):
            train = self._train_slice(entry)

            for window in range(windows):
                offset = window * distance
                test = self._test_slice(
                    entry, prediction_length=prediction_length, offset=offset
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
class OffsetSplitter(AbstractBaseSplitter):
    """
    A splitter that slices training and test data based on a fixed integer
    offset.

    Parameters
    ----------
    offset
        Offset determining where the training data ends.
        A positive offset indicates how many observations since the start of
        each series should be in the training slice; a negative offset
        indicates how many observations before the end of each series should
        be excluded from the training slice.
    """

    offset: int

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        return item[: self.offset]

    def _test_slice(
        self, item: TimeSeriesSlice, prediction_length: int, offset: int = 0
    ) -> Tuple[TimeSeriesSlice, TimeSeriesSlice]:
        offset_ = self.offset + offset
        if self.offset < 0 and offset_ >= 0:
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
    up to (including) the given ``date``.

    Parameters
    ----------
    date
        ``pandas.Period`` determining where the training data ends.
    """

    date: pd.Period

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        # the train-slice includes everything up to (including) the split date
        return item[: self.date]

    def _test_slice(
        self, item: TimeSeriesSlice, prediction_length: int, offset: int = 0
    ) -> Tuple[TimeSeriesSlice, TimeSeriesSlice]:
        return (
            item[: self.date + offset * item.start.freq],
            item[
                self.date
                + (offset + 1) * item.start.freq : self.date
                + (prediction_length + offset) * item.start.freq
            ],
        )


@dataclass
class TestDataset:
    """
    An iterable type used for wrapping test data.

    Elements of a ``TestDataset`` are pairs ``(input, label)``, where
    ``input`` is input data for models, while ``label`` is the future
    ground truth that models are supposed to predict.

    Parameters
    ----------
    dataset:
        Whole dataset used for testing.
    splitter:
        A specific splitter that knows how to slices training and
        test data.
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

    dataset: Dataset
    splitter: AbstractBaseSplitter
    prediction_length: int
    windows: int = 1
    distance: Optional[int] = None
    max_history: Optional[int] = None

    def __iter__(self) -> Generator[Tuple[DataEntry, DataEntry], None, None]:
        yield from self.splitter._generate_test_slices(
            dataset=self.dataset,
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.distance,
            max_history=self.max_history,
        )

    @property
    def input(self) -> Generator[DataEntry, None, None]:
        """
        Iterable over the ``input`` portion of the test data.
        """
        for input, _ in self:
            yield input

    @property
    def label(self) -> Generator[DataEntry, None, None]:
        """
        Iterable over the ``label`` portion of the test data.
        """
        for _, label in self:
            yield label


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

    def generate_instances(self, **kwargs) -> TestDataset:
        """
        Generate an iterator of test dataset, which includes input part and
        label part.

        Keyword arguments are the same as for :class:`TestDataset`.
        """
        return TestDataset(self.dataset, self.splitter, **kwargs)


@dataclass
class TrainingDataset:
    dataset: Dataset
    splitter: AbstractBaseSplitter

    def __iter__(self) -> Generator[DataEntry, None, None]:
        return self.splitter._generate_train_slices(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)


def split(
    dataset: Dataset, *, offset: int = None, date: pd.Period = None
) -> Tuple[TrainingDataset, TestTemplate]:
    assert (offset is None) != (
        date is None
    ), "You need to provide ``offset`` or ``date``, but not both."
    if offset is not None:
        return OffsetSplitter(offset).split(dataset)
    else:
        return DateSplitter(date).split(dataset)
