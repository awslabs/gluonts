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

# Standard library imports
import abc
from functools import reduce
from typing import Callable, Iterable, Iterator, List

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.runtime_params import GLUONTS_MAX_IDLE_TRANSFORMS


class Transformation(metaclass=abc.ABCMeta):
    """
    Base class for all Transformations.

    A Transformation processes works on a stream (iterator) of dictionaries.
    """

    @abc.abstractmethod
    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        pass

    def estimate(self, data_it: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return data_it  # default is to pass through without estimation

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def __lshift__(self, dataset: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return TransformedDataset(dataset, self)

    def __rrshift__(self, dataset: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return self << dataset


class TransformedDataset(Iterable[DataEntry]):
    """
    A dataset that corresponds to applying a list of transformations to each
    element in the base_dataset.
    This only supports SimpleTransformations, which do the same thing at
    prediction and training time.


    Parameters
    ----------
    base_dataset
        Dataset to transform
    transformations
        List of transformations to apply
    """

    def __init__(
        self, dataset: Iterator[DataEntry], transformation: Transformation
    ) -> None:
        self.dataset = dataset
        self.transformation = transformation

    def __iter__(self) -> Iterator[DataEntry]:
        yield from self.transformation(self.dataset, is_train=True)

    def __len__(self):
        return sum(1 for _ in self)


class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    @validated()
    def __init__(self, trans: List[Transformation]) -> None:
        self.transformations = []
        for transformation in trans:
            # flatten chains
            if isinstance(transformation, Chain):
                self.transformations.extend(transformation.transformations)
            else:
                self.transformations.append(transformation)

    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        tmp = data_it
        for t in self.transformations:
            tmp = t(tmp, is_train)
        return tmp

    def estimate(self, data_it: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return reduce(
            lambda x, y: y.estimate(x), self.transformations, data_it
        )


class Identity(Transformation):
    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        return data_it


class MapTransformation(Transformation):
    """
    Base class for Transformations that returns exactly one result per input in the stream.
    """

    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator:
        for data_entry in data_it:
            try:
                yield self.map_transform(data_entry.copy(), is_train)
            except Exception as e:
                raise e

    @abc.abstractmethod
    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        pass


class SimpleTransformation(MapTransformation):
    """
    Element wise transformations that are the same in train and test mode
    """

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return self.transform(data)

    @abc.abstractmethod
    def transform(self, data: DataEntry) -> DataEntry:
        pass

    def chain(self, other: Transformation) -> Chain:
        if isinstance(other, SimpleTransformation):
            return SimpleChain([self, other])
        else:
            return Chain([self, other])

    def __lt__(self, data: DataEntry) -> DataEntry:
        # `self < {"foo": "bar"}`
        return self.transform(data)


class SimpleChain(Chain, SimpleTransformation):
    """Like chain, but where all elements are of type SimpleTransformation."""

    def transform(self, data: DataEntry) -> DataEntry:
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data


class AdhocTransform(SimpleTransformation):
    """
    Applies a function as a transformation
    This is called ad-hoc, because it is not serializable.
    It is OK to use this for experiments and outside of a model pipeline that
    needs to be serialized.
    """

    def __init__(self, func: Callable[[DataEntry], DataEntry]) -> None:
        self.func = func

    def transform(self, data: DataEntry) -> DataEntry:
        return self.func(data.copy())


class FlatMapTransformation(Transformation):
    """
    Transformations that yield zero or more results per input, but do not combine
    elements from the input stream.
    """

    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator:
        num_idle_transforms = 0
        for data_entry in data_it:
            num_idle_transforms += 1
            try:
                for result in self.flatmap_transform(
                    data_entry.copy(), is_train
                ):
                    num_idle_transforms = 0
                    yield result
            except Exception as e:
                raise e
            if num_idle_transforms > GLUONTS_MAX_IDLE_TRANSFORMS:
                raise Exception(
                    f"Reached maximum number of idle transformation calls.\n"
                    f"This means the transformation looped over "
                    f"GLUONTS_MAX_IDLE_TRANSFORMS={GLUONTS_MAX_IDLE_TRANSFORMS} "
                    f"inputs without returning any output.\n"
                    f"This occurred in the following transformation:\n{self}"
                )

    @abc.abstractmethod
    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pass


class FilterTransformation(FlatMapTransformation):
    def __init__(self, condition: Callable[[DataEntry], bool]) -> None:
        self.condition = condition

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        if self.condition(data):
            yield data
