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

For uniform datasets, where all time series start and end at the same point in
time :class:`OffsetSplitter` can be used:

.. testcode::

    splitter = OffsetSplitter(offset=7)
    train, test_template = splitter.split(whole_dataset)

For all other datasets, the more flexible :class:`DateSplitter` can be used:

.. testcode::

    splitter = DateSplitter(
        date=pd.Period('2018-01-31', freq='D')
    )
    train, test_template = splitter.split(whole_dataset)

In the above examples, the ``train`` output is a regular ``Dataset`` that can
be used for training purposes; ``test_template`` can generate test instances
as follows:

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
from typing import Generator, Optional, Tuple

import pandas as pd

from gluonts.dataset import Dataset, DataEntry
from gluonts.dataset.field_names import FieldName


def to_positive_slice(slice_: slice, length: int) -> slice:
    """
    Return an equivalent slice with positive bounds, given the
    length of the sequence it will apply to.
    """
    start, stop = slice_.start, slice_.stop
    if start is not None and start < 0:
        start += length
        assert start >= 0
    if stop is not None and stop < 0:
        stop += length
        assert stop >= 0
    return slice(start, stop, slice_.step)


def to_integer_slice(slice_: slice, start: pd.Period) -> slice:
    """
    Returns an equivalent slice with integer bounds, given the
    start timestamp of the sequence it will apply to.
    """
    start_is_int = isinstance(slice_.start, (int, type(None)))
    stop_is_int = isinstance(slice_.stop, (int, type(None)))

    if start_is_int and stop_is_int:
        return slice_

    if isinstance(slice_.start, pd.Period):
        start_offset = (slice_.start - start).n
        assert start_offset >= 0
    elif start_is_int:
        start_offset = slice_.start
    else:
        raise ValueError(
            "Can only use None, int, or pd.Period for slicing, got type "
            f"{type(slice_.start)}"
        )

    if isinstance(slice_.stop, pd.Period):
        stop_offset = (slice_.stop - start).n + 1
        assert stop_offset >= 0
    elif stop_is_int:
        stop_offset = slice_.stop
    else:
        raise ValueError(
            "Can only use None, int, or pd.Period for slicing, got type "
            f"{type(slice_.stop)}"
        )

    return slice(start_offset, stop_offset)


def slice_data_entry(
    entry: DataEntry, slice_: slice, prediction_length: int = 0
) -> DataEntry:
    slice_ = to_positive_slice(
        to_integer_slice(slice_, entry[FieldName.START]),
        entry[FieldName.TARGET].shape[-1],
    )

    if slice_.stop is not None:
        slice_extended = slice(
            slice_.start, slice_.stop + prediction_length, slice_.step
        )
    else:
        slice_extended = slice_

    sliced_entry = dict(entry)

    if slice_.start is not None:
        offset = slice_.start
        if offset < 0:
            offset += entry["target"].shape[-1]
        sliced_entry[FieldName.START] += offset

    # TODO fix
    if len(sliced_entry[FieldName.TARGET].shape) == 1:
        sliced_entry[FieldName.TARGET] = sliced_entry[FieldName.TARGET][slice_]
    else:
        sliced_entry[FieldName.TARGET] = sliced_entry[FieldName.TARGET][
            :, slice_
        ]

    if FieldName.FEAT_DYNAMIC_REAL in sliced_entry:
        sliced_entry[FieldName.FEAT_DYNAMIC_REAL] = sliced_entry[
            FieldName.FEAT_DYNAMIC_REAL
        ][:, slice_extended]

    if FieldName.FEAT_DYNAMIC_CAT in sliced_entry:
        sliced_entry[FieldName.FEAT_DYNAMIC_CAT] = sliced_entry[
            FieldName.FEAT_DYNAMIC_CAT
        ][:, slice_extended]

    if FieldName.PAST_FEAT_DYNAMIC_REAL in sliced_entry:
        sliced_entry[FieldName.PAST_FEAT_DYNAMIC_REAL] = sliced_entry[
            FieldName.PAST_FEAT_DYNAMIC_REAL
        ][:, slice_]

    return sliced_entry


@dataclass
class TimeSeriesSlice:
    entry: DataEntry
    prediction_length: int = 0

    def to_data_entry(self) -> DataEntry:
        return self.entry

    @property
    def start(self) -> pd.Period:
        return self.entry[FieldName.START]

    @property
    def end(self) -> pd.Period:
        return self.start + len(self) - 1

    def __len__(self) -> int:
        return len(self.entry[FieldName.TARGET])

    def __getitem__(self, slc: slice) -> DataEntry:
        return slice_data_entry(
            self.entry, slc, prediction_length=self.prediction_length
        )


class AbstractBaseSplitter(ABC):
    """
    Base class for all other splitter.
    """

    @abstractmethod
    def training_entry(self, entry: DataEntry) -> DataEntry:
        pass

    @abstractmethod
    def test_pair(
        self, entry: DataEntry, prediction_length: int, offset: int = 0
    ) -> Tuple[DataEntry, DataEntry]:
        pass

    def split(
        self, dataset: Dataset
    ) -> Tuple["TrainingDataset", "TestTemplate"]:
        return (
            TrainingDataset(dataset=dataset, splitter=self),
            TestTemplate(dataset=dataset, splitter=self),
        )

    def generate_training_entries(
        self, dataset: Dataset
    ) -> Generator[DataEntry, None, None]:
        yield from map(self.training_entry, dataset)

    def generate_test_pairs(
        self,
        dataset: Dataset,
        prediction_length: int,
        windows: int = 1,
        distance: Optional[int] = None,
        max_history: Optional[int] = None,
    ) -> Generator[Tuple[DataEntry, DataEntry], None, None]:
        if distance is None:
            distance = prediction_length

        for entry in dataset:
            for window in range(windows):
                offset = window * distance
                test = self.test_pair(
                    entry, prediction_length=prediction_length, offset=offset
                )

                if max_history is not None:
                    yield TimeSeriesSlice(test[0])[-max_history:], test[1]
                else:
                    yield test[0], test[1]


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

    def training_entry(self, entry: DataEntry) -> DataEntry:
        return TimeSeriesSlice(entry)[: self.offset]

    def test_pair(
        self, entry: DataEntry, prediction_length: int, offset: int = 0
    ) -> Tuple[DataEntry, DataEntry]:
        offset_ = self.offset + offset
        if self.offset < 0 and offset_ >= 0:
            offset_ += len(entry)
        if offset_ + prediction_length:
            return (
                TimeSeriesSlice(entry, prediction_length)[:offset_],
                TimeSeriesSlice(entry, prediction_length)[
                    offset_ : offset_ + prediction_length
                ],
            )
        else:
            return (
                TimeSeriesSlice(entry, prediction_length)[:offset_],
                TimeSeriesSlice(entry, prediction_length)[offset_:],
            )


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

    def training_entry(self, entry: DataEntry) -> DataEntry:
        return TimeSeriesSlice(entry)[: self.date]

    def test_pair(
        self, entry: DataEntry, prediction_length: int, offset: int = 0
    ) -> Tuple[DataEntry, DataEntry]:
        date = self.date.asfreq(entry[FieldName.START].freq)
        return (
            TimeSeriesSlice(entry, prediction_length)[: date + offset],
            TimeSeriesSlice(entry, prediction_length)[
                date + (offset + 1) : date + (prediction_length + offset)
            ],
        )


@dataclass
class TestData:
    """
    An iterable type used for wrapping test data.

    Elements of a ``TestData`` object are pairs ``(input, label)``, where
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
        yield from self.splitter.generate_test_pairs(
            dataset=self.dataset,
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.distance,
            max_history=self.max_history,
        )

    def __len__(self):
        return len(self.dataset) * self.windows

    @property
    def input(self) -> "InputDataset":
        return InputDataset(self)

    @property
    def label(self) -> "LabelDataset":
        return LabelDataset(self)


@dataclass
class InputDataset:
    test_data: TestData

    def __len__(self):
        return len(self.test_data)

    def __iter__(self):
        for input, _label in self.test_data:
            yield input


@dataclass
class LabelDataset:
    test_data: TestData

    def __len__(self):
        return len(self.test_data)

    def __iter__(self):
        for _input, label in self.test_data:
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

    def generate_instances(
        self,
        prediction_length: int,
        windows: int = 1,
        distance: Optional[int] = None,
        max_history: Optional[int] = None,
    ) -> TestData:
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
        return TestData(
            self.dataset,
            self.splitter,
            prediction_length,
            windows,
            distance,
            max_history,
        )


@dataclass
class TrainingDataset:
    dataset: Dataset
    splitter: AbstractBaseSplitter

    def __iter__(self) -> Generator[DataEntry, None, None]:
        return self.splitter.generate_training_entries(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)


def split(
    dataset: Dataset, *, offset: Optional[int] = None, date: pd.Period = None
) -> Tuple[TrainingDataset, TestTemplate]:
    assert (offset is None) != (
        date is None
    ), "You need to provide ``offset`` or ``date``, but not both."
    if offset is not None:
        return OffsetSplitter(offset).split(dataset)
    else:
        return DateSplitter(date).split(dataset)
