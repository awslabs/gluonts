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
This module contains functions that work on `Optional` values. In contrast to
other approaches, this does not wrap values into a dedicated type, but works
on normal Python values, which are of type `Optional[T]`.

Thus, some functions are implemented identically but have different type
signatures. For example, both `map` and `and_then` both just apply a function
to a value if it is not `None`, but the result of `map` is `T` and the result
of `and_then` is `Optional[T]`.

The names are taken from Rust, see:
https://doc.rust-lang.org/stable/std/option/enum.Option.html

"""


from typing import Callable, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def expect(self: Optional[T], msg: str) -> T:
    """
    ::
        >>> expect(1, "My message")
        1
        >>> expect(None, "My message")
        Traceback (most recent call last):
            ...
        ValueError: My message
    """
    if self is None:
        raise ValueError(msg)

    return self


def do(self: Optional[T], fn: Callable[[T], U]) -> Optional[T]:
    """
    Apply `fn` to `self` then return `self`, if `self is not None.

    ::
        >>> do("a", print)
        a
        'a'
        >>> do(None, print)
    """
    if self is not None:
        fn(self)

    return self


def map(self: Optional[T], fn: Callable[[T], U]) -> Optional[U]:
    """
    ::
        >>> map(1, lambda x: x + 1)
        2
        >>> map(None, lambda x: x + 1)
    """
    return map_or(self, None, fn)


def map_or(self: Optional[T], default: U, fn: Callable[[T], U]) -> U:
    """
    ::
        >>> map_or(["x"], 0, len)
        1
        >>> map_or(None, 0, len)
        0
    """
    if self is None:
        return default

    return fn(self)


def map_or_else(
    self: Optional[T], default_fn: Callable[[], U], fn: Callable[[T], U]
) -> U:
    """
    ::
        >>> map_or_else(1, list, lambda n: [n])
        [1]
        >>> map_or_else(None, list, lambda n: [n])
        []
    """
    if self is None:
        return default_fn()

    return fn(self)


def unwrap(self: Optional[T]) -> T:
    """
    ::
        >>> unwrap(1)
        1
        >>> unwrap(None)
        Traceback (most recent call last):
            ...
        ValueError: Trying to unwrap `None` value.
    """
    return expect(self, "Trying to unwrap `None` value.")


def unwrap_or(self: Optional[T], default: T) -> T:
    """
    ::
        >>> unwrap_or(1, 2)
        1
        >>> unwrap_or(None, 2)
        2
    """
    if self is None:
        return default

    return self


def unwrap_or_else(self: Optional[T], fn: Callable[[], T]) -> T:
    """
    ::
        >>> unwrap_or_else([1, 2, 3], list)
        [1, 2, 3]
        >>> unwrap_or_else(None, list)
        []
    """
    if self is None:
        return fn()

    return self


def and_(self: Optional[T], other: Optional[U]) -> Optional[U]:
    """
    ::
        >>> and_(1, 2)
        2
        >>> and_(1, None)
        >>> and_(None, 2)
    """
    if self is None:
        return None

    return other


def and_then(self: Optional[T], fn: Callable[[T], Optional[U]]) -> Optional[U]:
    """
    ::
        >>> and_then([42], lambda xs: xs[0] if xs else None)
        42
        >>> and_then(None, lambda xs: xs[0] if xs else None)
    """
    return map(self, fn)  # type: ignore


def or_(self: Optional[T], default: Optional[T]) -> Optional[T]:
    if self is None:
        return default

    return self


def or_else(self: Optional[T], fn: Callable[[], Optional[T]]) -> Optional[T]:
    """Like `unwrap_or_else`, except that it returns an optional value.

    ::
        >>> or_else([42], list)
        [42]
        >>> or_else(None, list)
        []
    """
    return unwrap_or_else(self, fn)  # type: ignore


def contains(self: Optional[T], other: U) -> bool:
    """
    ::
        >>> contains(1, 1)
        True
        >>> contains(1, 2)
        False
        >>> contains(None, 3)
        False
    """
    if self is None:
        return False

    return self == other


def filter(self: Optional[T], pred: Callable[[T], bool]) -> Optional[T]:
    """
    ::
        >>> is_even = lambda n: n % 2 == 0
        >>> filter(1, is_even)
        >>> filter(2, is_even)
        2
        >>> filter(None, is_even)
    """
    if self is None or not pred(self):
        return None

    return self


def xor(self: Optional[T], other: Optional[T]) -> Optional[T]:
    """
    ::
        >>> xor(1, None)
        1
        >>> xor(None, 2)
        2
        >>> xor(1, 2)
        >>> xor(None, None)
    """

    if self is None:
        return other

    if other is None:
        return self

    return None


def flatten(self: Optional[Optional[T]]) -> Optional[T]:
    return self  # type: ignore
