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

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

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
    return map_or(val, fn, None)


def map_or(val: Optional[T], fn: Callable[[T], U], default: U) -> U:
    """
    ::
        >>> map_or(["x"], len, 0)
        1
        >>> map_or(None, len, 0)
        0
    """
    if val is None:
        return default

    return fn(val)


def map_or_else(
    val: Optional[T],
    fn: Callable[[T], U],
    factory: Callable[[], U],
) -> U:
    """
    ::
        >>> map_or_else(1, lambda n: [n], list)
        [1]
        >>> map_or_else(None, lambda n: [n], list)
        []
    """
    if val is None:
        return factory()

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


def unwrap_or_else(val: Optional[T], factory: Callable[[], T]) -> T:
    """
    ::
        >>> unwrap_or_else([1, 2, 3], list)
        [1, 2, 3]
        >>> unwrap_or_else(None, list)
        []
    """
    if val is None:
        return factory()

    return val


def and_(val: Optional[T], other: Optional[U]) -> Optional[U]:
    """
    ::
        >>> and_(1, 2)
        2
        >>> and_(1, None)
        >>> and_(None, 2)
    """
    if val is None:
        return None

    return other


def and_then(val: Optional[T], fn: Callable[[T], Optional[U]]) -> Optional[U]:
    """
    ::
        >>> and_then([42], lambda xs: xs[0] if xs else None)
        42
        >>> and_then(None, lambda xs: xs[0] if xs else None)
    """
    return map(val, fn)  # type: ignore


def or_(val: Optional[T], default: Optional[T]) -> Optional[T]:
    if val is None:
        return default

    return val


def or_else(
    val: Optional[T], factory: Callable[[], Optional[T]]
) -> Optional[T]:
    """Like `unwrap_or_else`, except that it returns an optional value.

    ::
        >>> or_else([42], list)
        [42]
        >>> or_else(None, list)
        []
    """
    return unwrap_or_else(val, factory)  # type: ignore


def contains(val: Optional[T], other: U) -> bool:
    """
    ::
        >>> contains(1, 1)
        True
        >>> contains(1, 2)
        False
        >>> contains(None, 3)
        False
    """
    if val is None:
        return False

    return val == other


def filter(val: Optional[T], pred: Callable[[T], bool]) -> Optional[T]:
    """
    ::
        >>> is_even = lambda n: n % 2 == 0
        >>> filter(1, is_even)
        >>> filter(2, is_even)
        2
        >>> filter(None, is_even)
    """
    if val is None or not pred(val):
        return None

    return val


def xor(val: Optional[T], other: Optional[T]) -> Optional[T]:
    """
    ::
        >>> xor(1, None)
        1
        >>> xor(None, 2)
        2
        >>> xor(1, 2)
        >>> xor(None, None)
    """

    if val is None:
        return other

    if other is None:
        return val

    return None


def flatten(val: Optional[Optional[T]]) -> Optional[T]:
    return val  # type: ignore


@dataclass
class Maybe(Generic[T]):
    val: Optional[T]

    def expect(self, msg: str) -> T:
        return expect(self.val, msg)

    def do(self, fn: Callable[[T], U]) -> Optional[T]:
        return do(self.val, fn)

    def map(self, fn: Callable[[T], U]) -> Optional[U]:
        """
        ::
            >>> Maybe(1).map(lambda x: x + 1)
            2
            >>> Maybe(None).map(lambda x: x + 1)
        """
        return map(self.val, fn)

    def map_or(self, fn: Callable[[T], U], default: U) -> U:
        """
        ::
            >>> Maybe(["x"]).map_or(len, 0)
            1
            >>> Maybe(None).map_or(len, 0)
            0
        """
        return map_or(self.val, fn, default)

    def map_or_else(
        self,
        fn: Callable[[T], U],
        factory: Callable[[], U],
    ) -> U:
        """
        ::
            >>> Maybe(1).map_or_else(lambda n: [n], list)
            [1]
            >>> Maybe(None).map_or_else(lambda n: [n], list)
            []
        """
        return map_or_else(self.val, fn, factory)

    def unwrap(self) -> T:
        """
        ::
            >>> Maybe(1).unwrap()
            1
            >>> Maybe(None).unwrap()
            Traceback (most recent call last):
                ...
            ValueError: Trying to unwrap `None` value.
        """
        return unwrap(self.val)

    def unwrap_or(self, default: T) -> T:
        """
        ::
            >>> Maybe(1).unwrap_or(2)
            1
            >>> Maybe(None).unwrap_or(2)
            2
        """
        return unwrap_or(self.val, default)

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        """
        ::
            >>> Maybe([1, 2, 3]).unwrap_or_else(list)
            [1, 2, 3]
            >>> Maybe(None).unwrap_or_else(list)
            []
        """
        return unwrap_or_else(self.val, fn)

    def and_(self, other: Optional[U]) -> Optional[U]:
        """
        ::
            >>> Maybe(1).and_(2)
            2
            >>> Maybe(1).and_(None)
            >>> Maybe(None).and_(2)
        """
        return and_(self.val, other)

    def __and__(self, other: Optional[U]) -> Optional[U]:
        """
        ::
            >>> Maybe(1) & 2
            2
            >>> Maybe(1) & None
            >>> Maybe(None) & 2
        """
        return and_(self.val, other)

    def and_then(self, fn: Callable[[T], Optional[U]]) -> Optional[U]:
        """
        ::
            >>> Maybe([42]).and_then(lambda xs: xs[0] if xs else None)
            42
            >>> Maybe([]).and_then(lambda xs: xs[0] if xs else None)
            >>> Maybe(None).and_then(lambda xs: xs[0] if xs else None)
        """
        return and_then(self.val, fn)

    def or_(self, default: Optional[T]) -> Optional[T]:
        return or_(self.val, default)

    def __or__(self, default: Optional[T]) -> Optional[T]:
        return or_(self.val, default)

    def or_else(self, factory: Callable[[], Optional[T]]) -> Optional[T]:
        """Like `unwrap_or_else`, except that it returns an optional value.

        ::
            >>> Maybe([42]).or_else(list)
            [42]
            >>> Maybe(None).or_else(list)
            []
        """
        return or_else(self.val, factory)

    def contains(self, other: U) -> bool:
        """
        ::
            >>> Maybe(1).contains(1)
            True
            >>> Maybe(1).contains(2)
            False
            >>> Maybe(None).contains(3)
            False
        """
        return contains(self.val, other)

    def filter(self, pred: Callable[[T], bool]) -> Optional[T]:
        """
        ::
            >>> is_even = lambda n: n % 2 == 0
            >>> Maybe(1).filter(is_even)
            >>> Maybe(2).filter(is_even)
            2
            >>> Maybe(None).filter(is_even)
        """
        return filter(self.val, pred)

    def xor(self, other: Optional[T]) -> Optional[T]:
        """
        ::
            >>> Maybe(1).xor(None)
            1
            >>> Maybe(None).xor(2)
            2
            >>> Maybe(1).xor(2)
            >>> Maybe(None).xor(None)
        """
        return xor(self.val, other)

    def __xor__(self, other: Optional[T]) -> Optional[T]:
        """
        ::
            >>> Maybe(1) ^ None
            1
            >>> Maybe(None) ^ 2
            2
            >>> Maybe(1) ^ 2
            >>> Maybe(None) ^ None
        """
        return xor(self.val, other)

    def flatten(self: Maybe[Optional[T]]) -> Optional[T]:
        return flatten(self.val)
