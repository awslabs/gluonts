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

import abc
from typing import Callable, Iterable, Iterator, List

from gluonts.core.component import equals
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.env import env


class Transformation(metaclass=abc.ABCMeta):
    """
    Base class for all Transformations.

    A Transformation processes works on a stream (iterator) of dictionaries.
    """

    @abc.abstractmethod
    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterable[DataEntry]:
        pass

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def apply(
        self, dataset: Dataset, is_train: bool = True
    ) -> "TransformedDataset":
        return TransformedDataset(dataset, self, is_train=is_train)


class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    @validated()
    def __init__(self, trans: List[Transformation]) -> None:
        self.transformations: List[Transformation] = []
        for transformation in trans:
            # flatten chains
            if isinstance(transformation, Chain):
                self.transformations.extend(transformation.transformations)
            else:
                self.transformations.append(transformation)

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterable[DataEntry]:
        tmp = data_it
        for t in self.transformations:
            tmp = t(tmp, is_train)
        return tmp


class TransformedDataset(Dataset):
    """
    A dataset that corresponds to applying a list of transformations to each
    element in the base_dataset. This only supports SimpleTransformations,
    which do the same thing at prediction and training time.

    Parameters
    ----------
    base_dataset
        Dataset to transform
    transformations
        List of transformations to apply
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transformation: Transformation,
        is_train=True,
    ) -> None:
        self.base_dataset = base_dataset
        self.transformation = transformation
        self.is_train = is_train

    def __len__(self):
        # NOTE this is unsafe when transformations are run with is_train = True
        # since some transformations may not be deterministic
        # (instance splitter)
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[DataEntry]:
        yield from self.transformation(
            self.base_dataset, is_train=self.is_train
        )


class Identity(Transformation):
    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterable[DataEntry]:
        return data_it


class MapTransformation(Transformation):
    """
    Base class for Transformations that returns exactly one result per input in
    the stream.
    """

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
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
    Element wise transformations that are the same in train and test mode.
    """

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return self.transform(data)

    @abc.abstractmethod
    def transform(self, data: DataEntry) -> DataEntry:
        pass


class AdhocTransform(SimpleTransformation):
    """
    Applies a function as a transformation This is called ad-hoc, because it is
    not serializable.

    It is OK to use this for experiments and outside of a model pipeline that
    needs to be serialized.
    """

    def __init__(self, func: Callable[[DataEntry], DataEntry]) -> None:
        self.func = func

    def transform(self, data: DataEntry) -> DataEntry:
        return self.func(data.copy())


class FlatMapTransformation(Transformation):
    """
    Transformations that yield zero or more results per input, but do not
    combine elements from the input stream.
    """

    @validated()
    def __init__(self):
        self.max_idle_transforms = env.max_idle_transforms

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterator:
        num_idle_transforms = 0
        for data_entry in data_it:
            num_idle_transforms += 1
            for result in self.flatmap_transform(data_entry.copy(), is_train):
                num_idle_transforms = 0
                yield result

            if (
                # negative values disable the check
                self.max_idle_transforms > 0
                and num_idle_transforms > self.max_idle_transforms
            ):
                raise Exception(
                    "Reached maximum number of idle transformation"
                    " calls.\nThis means the transformation looped over"
                    f" {self.max_idle_transforms} inputs without returning any"
                    " output.\nThis occurred in the following"
                    f" transformation:\n{self}"
                )

    @abc.abstractmethod
    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pass


class FilterTransformation(FlatMapTransformation):
    def __init__(self, condition: Callable[[DataEntry], bool]) -> None:
        super().__init__()
        self.condition = condition

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        if self.condition(data):
            yield data


# The __init__ in FilterTransformation is not validated but the __init__ in the
# parent class (FlatMapTransformation) is. So now the code
# (equals_default_impl) validate the arguments in FilterTransformation, which
# is an empty dict for all the FilterTransformation. We can not make __init__
# FilterTransformation as validated as we may use lambda function as the
# argument
@equals.register(FilterTransformation)
def equals_filter_transformation(
    this: FilterTransformation, that: FilterTransformation
):
    return this.condition.__code__.co_code == that.condition.__code__.co_code
