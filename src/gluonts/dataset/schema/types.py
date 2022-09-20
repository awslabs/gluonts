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
class Default(GenericType[T]):
    value: T
    base: Type = None

    def __post_init__(self):
        if self.base is not None:
            self.value = self.base.apply(self.value)

    def apply(self, data):
        return self.value


@dataclass
class Array(GenericType[T]):
    dtype: typing.Type[T]
    ndim: int
    time_dim: typing.Optional[int] = None

    def apply(self, data):
        arr = np.asarray(data, dtype=self.dtype)

        if arr.ndim < self.ndim:
            to_expand = self.ndim - arr.ndim
            new_shape = (1,) * to_expand + arr.shape
            arr = arr.reshape(new_shape)
        elif arr.ndim > self.ndim:
            raise ValueError("Too many dimensions.")

        return arr


@dataclass
class Numeric(Type):
    pass


@dataclass
class WrappedType(Type):
    wrapped_type: typing.ClassVar[T]

    def apply(self, data):
        return self.wrapped_type(data)


@dataclass
class String(Type):
    wrapped_type: typing.ClassVar = str


@dataclass
class Timestamp(Type):
    pass


@dataclass
class Period(WrappedType):
    freq: str = None

    def apply(self, data):
        return pd.Period(data, freq=self.freq)
