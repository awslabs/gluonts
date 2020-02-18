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

import numpy as np
from typing import TypeVar


T = TypeVar("T")


def as_numpy_array(xs, dtype, ndim):
    array = np.asarray(xs, dtype=dtype)
    assert (
        ndim == array.ndim
    ), f"Dimension mismatch, got {array.ndim} expected {ndim}."
    return array


class Generic:
    def __class_getitem__(cls, *params):
        class GenericAlias(cls):
            __args__ = tuple(params)

            arg_names = ", ".join(param.__name__ for param in params)
            __qualname__ = f"{cls.__name__}[{arg_names}]"

        return GenericAlias


class Array1D(Generic[T]):
    @classmethod
    def __get_validators__(cls):
        assert cls.__args__
        dtype = cls.__args__[0]
        return [lambda v: as_numpy_array(v, dtype, 1)]


class Array2D(Generic[T]):
    @classmethod
    def __get_validators__(cls):
        assert cls.__args__
        dtype = cls.__args__[0]
        return [lambda v: as_numpy_array(v, dtype, 2)]


class T(Array1D):
    pass


class C(Array1D):
    pass


class CT(Array2D):
    pass


class TC(Array2D):
    pass
