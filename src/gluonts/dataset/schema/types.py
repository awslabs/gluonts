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
    def __call__(self, data):
        raise NotImplementedError


class GenericType(Type, typing.Generic[T]):
    pass


@dataclass
class Default(GenericType[T]):
    value: T
    base: typing.Optional[Type] = None

    def __post_init__(self):
        if self.base is not None:
            self.value = self.base(self.value)

    def __call__(self, data) -> T:
        return self.value


@dataclass
class Array(GenericType[T]):
    """Array type with fixed number of dimensions, but optional dtype and time
    dimension.

    This class ensures that the handled output data, will have `ndim` number of
    dimensions. If specified, `dtype` will be applied to the input to force a
    consistent type, e.g. ``np.float32``. `time_dim` is just a marker,
    indicating which axis notes the time-axis, useful for splitting. If
    `time_dim` is none, the array is time invariant.
    """

    ndim: int
    dtype: typing.Optional[typing.Type[T]] = None
    time_dim: typing.Optional[int] = None

    def __call__(self, data):
        arr = np.asarray(data, dtype=self.dtype)

        if arr.ndim != self.ndim:
            raise ValueError("Dimensions do not match.")

        return arr


@dataclass
class Period:
    freq: typing.Optional[str] = None

    def __call__(self, data):
        return pd.Period(data, freq=self.freq)
