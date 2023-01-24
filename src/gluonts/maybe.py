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

from typing import Callable, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def expect(val: Optional[T], msg: str) -> T:
    if val is None:
        raise ValueError(msg)

    return val


def do(val: Optional[T], fn: Callable[[T], U]) -> Optional[T]:
    if val is not None:
        fn(val)

    return val


def map(val: Optional[T], fn: Callable[[T], U]) -> Optional[U]:
    return map_or(val, None, fn)


def map_or(
    val: Optional[T], default: Optional[U], fn: Callable[[T], U]
) -> Optional[U]:
    if val is None:
        return default

    return fn(val)


def map_or_else(
    val: Optional[T], default_fn: Callable[[], U], fn: Callable[[T], U]
) -> Optional[U]:
    if val is None:
        return default

    return fn(val)


def unwrap(val: Optional[T]) -> T:
    return expect("Trying to unwrap `None` value.")


def unwrap_or(val: Optional[T], default: T) -> T:
    if val is None:
        return default

    return val


def unwrap_or_else(val: Optional[T], fn: Callable[[], T]) -> T:
    if val is None:
        return fn()

    return val


def or_(val: Optional[T], default: Optional[T]) -> Optional[T]:
    if val is None:
        return default

    return val
