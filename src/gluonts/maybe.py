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
    """
    ::
        >>> expect(1, "My message")
        1
        >>> expect(None, "My message")
        Traceback (most recent call last):
            ...
        ValueError: My message
    """
    if val is None:
        raise ValueError(msg)

    return val


def do(val: Optional[T], fn: Callable[[T], U]) -> Optional[T]:
    """
    Apply `fn` to `val` then return `val`, if `val is not None.

    ::
        >>> do("a", print)
        a
        'a'
        >>> do(None, print)
    """
    if val is not None:
        fn(val)

    return val


def map(val: Optional[T], fn: Callable[[T], U]) -> Optional[U]:
    """
    ::
        >>> map(1, lambda x: x + 1)
        2
        >>> map(None, lambda x: x + 1)
    """
    return map_or(val, None, fn)


def map_or(
    val: Optional[T], default: Optional[U], fn: Callable[[T], U]
) -> Optional[U]:
    """
    ::
        >>> map_or(["x"], 0, len)
        1
        >>> map_or(None, 0, len)
        0
    """
    if val is None:
        return default

    return fn(val)


def map_or_else(
    val: Optional[T], default_fn: Callable[[], U], fn: Callable[[T], U]
) -> Optional[U]:
    """
    ::
        >>> map_or_else(1, list, lambda n: [n])
        [1]
        >>> map_or_else(None, list, lambda n: [n])
        []
    """
    if val is None:
        return default_fn()

    return fn(val)


def unwrap(val: Optional[T]) -> T:
    """
    ::
        >>> unwrap(1)
        1
        >>> unwrap(None)
        Traceback (most recent call last):
            ...
        ValueError: Trying to unwrap `None` value.
    """
    return expect(val, "Trying to unwrap `None` value.")


def unwrap_or(val: Optional[T], default: T) -> T:
    """
    ::
        >>> unwrap_or(1, 2)
        1
        >>> unwrap_or(None, 2)
        2
    """
    if val is None:
        return default

    return val


def unwrap_or_else(val: Optional[T], fn: Callable[[], T]) -> T:
    """
    ::
        >>> unwrap_or_else([1, 2, 3], list)
        [1, 2, 3]
        >>> unwrap_or_else(None, list)
        []
    """
    if val is None:
        return fn()

    return val


def or_(val: Optional[T], default: Optional[T]) -> Optional[T]:
    if val is None:
        return default

    return val
