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

import typing
from dataclasses import dataclass

import pandas as pd
import numpy as np


T = typing.TypeVar("T")


class Type:
    def apply(self, data):
        raise NotImplementedError


class GenericType(Type, typing.Generic[T]):
    pass


@dataclass
class AnyType(GenericType[T]):
    def apply(self, data) -> T:
        return data


Any = AnyType()


@dataclass
class Default(GenericType[T]):
    value: T
    base: typing.Optional[Type] = None

    def __post_init__(self):
        if self.base is not None:
            self.value = self.base.apply(self.value)

    def apply(self, data) -> T:
        return self.value


@dataclass
class Array(GenericType[T]):
    """Array type with fixed number of dimensions, but optional dtype and time
    dimension.

    This class ensures that the handled output data, will have `ndim` number of
    dimensions. If specified, `dtype` will be applied to the input to force a
    consistent type, e.g. ``np.float32``. `time_axis` is just a marker,
    indicating which axis notes the time-axis, useful for splitting. If
    `time_axis` is none, the array is time invariant.
    """

    ndim: int
    dtype: typing.Optional[typing.Type[T]] = None
    time_axis: typing.Optional[int] = None
    past_only: bool = False

    def apply(self, data):
        arr = np.asarray(data, dtype=self.dtype)

        if arr.ndim != self.ndim:
            raise ValueError("Dimensions do not match.")

        return arr

    def split(self, data, idx, future_length: int = 0):
        if self.time_axis is None:
            return data

        # if self.past_only:
        #     data = np.pad(data, (0, future_length), constant_values=np.nan)

        sl = [None] * self.ndim

        sl[self.time_axis] = slice(None, idx)
        left = data[tuple(sl)]

        sl[self.time_axis] = slice(idx, None)
        right = data[tuple(sl)]

        return left, right

    def time_dim(self, data):
        return data.shape[self.time_axis]


@dataclass
class Period:
    freq: typing.Optional[str] = None

    def apply(self, data):
        return pd.Period(data, freq=self.freq)
