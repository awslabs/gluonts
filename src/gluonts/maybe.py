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
This module contains functions that work on ``Optional`` values. It supports
wrapping of values into a dedicated type (``Maybe``), but also works on normal
Python values, which are of type ``Optional[T]``.

Each function is implemented twice, as a simple function and as a method on
``maybe.Maybe``::

    maybe.Some(1).map(fn) -> Maybe[T]

    maybe.map(1, fn) -> Optional[T]


Methods on ``Maybe`` return ``Maybe`` types, while functions return
``Optional`` values.

The names are taken from Rust, see:
https://doc.rust-lang.org/stable/std/option/enum.Option.html

.. note::
    The argument order for ``map_or`` and ``map_or_else`` is reversed
    compared to their Rust counterparts.

    ``do`` is not implemented in Rust but mimics ``toolz.do`` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    TypeVar,
    Tuple,
    List,
    Union,
    cast,
    Any,
)
from typing_extensions import final, ParamSpec, Concatenate

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")

P = ParamSpec("P")

OptionalOrMaybe = Union[Optional[T], "Maybe[T]"]


def box(val: OptionalOrMaybe[T]) -> Maybe[T]:
    """
    Turn ``Optional[T]`` into ``Maybe[T]``.
    """

    if isinstance(val, Maybe):
        return val

    if val is None:
        return Nothing

    return Some(val)


def unbox(val: OptionalOrMaybe[T]) -> Optional[T]:
    """
    Turn ``Optional[T]`` into ``Maybe[T]``.
    """

    if isinstance(val, Maybe):
        return val.unbox()

    return val


def flatten(val: Optional[Optional[T]]) -> Optional[T]:
    """
    Flatten nested optional value.

    Note: This just returns the value, but changes the type from
    ``Optional[Optional[T]]`` to ``Optional[T].``
    """

    return val  # type: ignore


def expect(val: OptionalOrMaybe[T], msg: str) -> T:
    """
    Ensure that ``val`` is not ``None``, raises a ``ValueError`` using ``msg``
    otherwise.

    >>> expect(1, "My message")
    1
    >>> expect(None, "My message")
    Traceback (most recent call last):
        ...
    ValueError: My message
    """
    return box(val).expect(msg)


def do(val: OptionalOrMaybe[T], fn: Callable[[T], U]) -> Optional[T]:
    """
    Apply ``fn`` to ``val`` then return ``val``, if ``val`` is not ``None``.

    >>> do("a", print)
    a
    'a'
    >>> do(None, print)
    """
    return box(val).do(fn).unbox()


def map(
    val: OptionalOrMaybe[T],
    fn: Callable[Concatenate[T, P], U],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Optional[U]:
    """
    Apply ``fn`` to ``val`` if ``val`` is not ``None``.

    >>> map(1, lambda x: x + 1)
    2
    >>> map(None, lambda x: x + 1)

    Allows to pass additional arguments that are passed to ``fn``:

    >>> map(10, divmod, 3)
    (3, 1)
    """
    return box(val).map(fn, *args, **kwargs).unbox()


def map_or(val: OptionalOrMaybe[T], fn: Callable[[T], U], default: U) -> U:
    """
    Apply ``fn`` to ``val`` if ``val`` is not ``None`` and return the result.
    In case of ``None`` the provided ``default`` is returned instead.

    This is similar to calling ``map`` and ``unwrap_or`` in succession.

    >>> map_or(["x"], len, 0)
    1
    >>> map_or(None, len, 0)
    0
    """
    return box(val).map_or(fn, default)


def map_or_else(
    val: OptionalOrMaybe[T],
    fn: Callable[[T], U],
    factory: Callable[[], U],
) -> U:
    """
    Similar to ``map_or``, except that the returned value is lazily evaluated.

    This is similar to calling ``map`` and ``unwrap_or_else`` in succession.

    >>> map_or_else(1, lambda n: [n], list)
    [1]
    >>> map_or_else(None, lambda n: [n], list)
    []
    """
    return box(val).map_or_else(fn, factory)


def unwrap(val: OptionalOrMaybe[T]) -> T:
    """
    Assert that the value is not ``None``.

    >>> unwrap(1)
    1
    >>> unwrap(None)
    Traceback (most recent call last):
        ...
    ValueError: Trying to unwrap `None` value.
    """
    return box(val).expect("Trying to unwrap `None` value.")


def unwrap_or(val: OptionalOrMaybe[T], default: T) -> T:
    """
    Get ``val`` if it is not ``None``, or ``default`` otherwise.

    >>> unwrap_or(1, 2)
    1
    >>> unwrap_or(None, 2)
    2
    """
    return box(val).unwrap_or(default)


def unwrap_or_else(val: OptionalOrMaybe[T], factory: Callable[[], T]) -> T:
    """
    Get ``val`` if it is not ``None``, or invoke ``factory`` to get a fallback.

    >>> unwrap_or_else([1, 2, 3], list)
    [1, 2, 3]
    >>> unwrap_or_else(None, list)
    []
    """

    return box(val).unwrap_or_else(factory)


def and_(val: OptionalOrMaybe[T], other: OptionalOrMaybe[U]) -> Optional[U]:
    """
    Like ``a and b`` in Python, except only considering ``None``.

    This is implement identical to ``unwrap_or``, but different with respect to
    types.

    >>> and_(1, 2)
    2
    >>> and_(1, None)
    >>> and_(None, 2)
    """
    return box(val).and_(other).unbox()


def and_then(
    val: OptionalOrMaybe[T],
    fn: Callable[Concatenate[T, P], OptionalOrMaybe[U]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Optional[U]:
    """
    Apply ``fn`` to ``val`` if it is not ``None`` and return the result.

    In contrast to ``map``, ``fn`` always returns an ``Optional``, which is
    consequently flattened.

    >>> and_then([42], lambda xs: xs[0] if xs else None)
    42
    >>> and_then(None, lambda xs: xs[0] if xs else None)
    """
    return box(val).and_then(fn, *args, **kwargs).unbox()


def or_(val: OptionalOrMaybe[T], default: Optional[T]) -> Optional[T]:
    """
    Like ``a or b`` in Python, except only considering ``None``.

    >>> or_(1, 2)
    1
    >>> or_(1, None)
    1
    >>> or_(None, 2)
    2
    """
    return box(val).or_(default).unbox()


def or_else(
    val: OptionalOrMaybe[T], factory: Callable[[], Optional[T]]
) -> Optional[T]:
    """
    Like ``unwrap_or_else``, except that it returns an optional value.

    >>> or_else([42], list)
    [42]
    >>> or_else(None, list)
    []
    """
    return box(val).or_else(factory).unbox()


def contains(val: OptionalOrMaybe[T], other: U) -> bool:
    """
    Check if ``val`` equals ``other``, always return ``False`` if ``val`` is
    ``None``.

    >>> contains(1, 1)
    True
    >>> contains(1, 2)
    False
    >>> contains(None, 3)
    False
    """
    return box(val).contains(other)


def filter(val: OptionalOrMaybe[T], pred: Callable[[T], bool]) -> Optional[T]:
    """
    Return ``None`` if ``val`` is ``None`` or if ``pred(val)`` does not return
    ``True``, otherwise return ``val``.

    >>> is_even = lambda n: n % 2 == 0
    >>> filter(1, is_even)
    >>> filter(2, is_even)
    2
    >>> filter(None, is_even)
    """
    return box(val).filter(pred).unbox()


def xor(val: OptionalOrMaybe[T], other: OptionalOrMaybe[T]) -> Optional[T]:
    """
    Return either ``val`` or ``other`` if the other is ``None``. Also return
    ``None`` if both are not ``None``.

    >>> xor(1, None)
    1
    >>> xor(None, 2)
    2
    >>> xor(1, 2)
    >>> xor(None, None)
    """
    return box(val).xor(other).unbox()


def iter(val: OptionalOrMaybe[T]) -> List[T]:
    """
    Wrap ``val`` into a list, if it is not ``None``.

    Allows to use for loops on optional values.
    """
    return box(val).iter()


def zip(
    val: OptionalOrMaybe[T], other: OptionalOrMaybe[U]
) -> Optional[Tuple[T, U]]:
    """
    Return tuple of ``(val, other)`` if neither is ``None``, otherwise return
    ``None``.
    """
    return box(val).zip(other).unbox()


def zip_with(
    val: OptionalOrMaybe[T], other: OptionalOrMaybe[U], fn: Callable[[T, U], R]
) -> Optional[R]:
    """
    Apply function to two optional values, if neither of them is ``None``:

    >>> add = lambda left, right: left + right
    >>> zip_with(1, 2, add)
    3
    >>> zip_with(1, None, add)
    >>> zip_with(None, 2, add)
    """
    return box(val).zip_with(other, fn).unbox()


class Maybe(ABC, Generic[T]):
    @abstractmethod
    def unbox(self) -> Optional[T]:
        """
        Turn ``Maybe[T]`` into ``Optional[T]``.

        >>> Some(1).unbox()
        1
        >>> Some(None).unbox() is None
        True
        """

    @abstractmethod
    def is_some(self) -> bool:
        pass

    def is_none(self) -> bool:
        return not self.is_some()

    @abstractmethod
    def expect(self, msg: str) -> T:
        """
        Ensure that ``val`` is not ``None``, raises a ``ValueError`` using
        ``msg`` otherwise.

        >>> Some(1).expect("My message")
        1
        >>> Nothing.expect("My message")
        Traceback (most recent call last):
            ...
        ValueError: My message
        """

    @abstractmethod
    def do(self, fn: Callable[[T], U]) -> Maybe[T]:
        """
        Apply ``fn`` to ``val`` then return ``val``, if ``val`` is not
        ``None``.

        >>> Some("a").do(print)
        a
        Some('a')
        >>> Nothing.do(print)
        Nothing
        """

    @abstractmethod
    def map(
        self, fn: Callable[Concatenate[T, P], U], *args, **kwargs
    ) -> Maybe[U]:
        """
        Apply ``fn`` to ``val`` if ``val`` is not ``None``.

        >>> Some(1).map(lambda x: x + 1)
        Some(2)
        >>> Nothing.map(lambda x: x + 1)
        Nothing

        Allows to pass additional arguments that are passed to ``fn``:

        >>> Some(10).map(divmod, 3)
        Some((3, 1))
        """

    @abstractmethod
    def map_or(self, fn: Callable[[T], U], default: U) -> U:
        """
        Apply ``fn`` to ``val`` if ``val`` is not ``None`` and return the
        result. In case of ``None`` the provided ``default`` is returned
        instead.

        This is similar to calling ``map`` and ``unwrap_or`` in succession.

        >>> Some(["x"]).map_or(len, 0)
        1
        >>> Nothing.map_or(len, 0)
        0
        """

    @abstractmethod
    def map_or_else(
        self,
        fn: Callable[[T], U],
        factory: Callable[[], U],
    ) -> U:
        """
        Similar to ``map_or``, except that the returned value is lazily
        evaluated.

        This is similar to calling ``map`` and ``unwrap_or_else`` in
        succession.

        >>> Some(1).map_or_else(lambda n: [n], list)
        [1]
        >>> Nothing.map_or_else(lambda n: [n], list)
        []
        """

    @abstractmethod
    def unwrap(self) -> T:
        """
        Assert that the value is not ``None``.

        >>> Some(1).unwrap()
        1
        >>> Nothing.unwrap()
        Traceback (most recent call last):
            ...
        ValueError: Trying to unwrap `None` value.
        """

    @abstractmethod
    def unwrap_or(self, default: T) -> T:
        """
        Get ``val`` if it is not ``None``, or ``default`` otherwise.

        >>> Some(1).unwrap_or(2)
        1
        >>> Nothing.unwrap_or(2)
        2
        """

    @abstractmethod
    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        """
        Get ``val`` if it is not ``None``, or invoke ``factory`` to get a
        fallback.

        >>> Some([1, 2, 3]).unwrap_or_else(list)
        [1, 2, 3]
        >>> Nothing.unwrap_or_else(list)
        []
        """

    @abstractmethod
    def and_(self, other: OptionalOrMaybe[U]) -> Maybe[U]:
        """
        Like ``a and b`` in Python, except only considering ``None``.

        This is implement identical to ``unwrap_or``, but different with
        respect to types.

        >>> Some(1).and_(2)
        Some(2)
        >>> Some(1).and_(None)
        Nothing
        >>> Nothing.and_(2)
        Nothing
        """

    def __and__(self, other: OptionalOrMaybe[U]) -> Maybe[U]:
        """
        Like ``a and b`` in Python, except only considering ``None``.

        This is implement identical to ``unwrap_or``, but different with
        respect to types.

        >>> Some(1) & 2
        Some(2)
        >>> Some(1) & None
        Nothing
        >>> Nothing & 2
        Nothing
        """
        return self.and_(other)

    @abstractmethod
    def and_then(
        self,
        fn: Callable[Concatenate[T, P], OptionalOrMaybe[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Maybe[U]:
        """
        Apply ``fn`` to ``val`` if it is not ``None`` and return the result.

        In contrast to ``map``, ``fn`` always returns an ``Optional``, which is
        consequently flattened.

        >>> Some([42]).and_then(lambda xs: xs[0] if xs else None)
        Some(42)
        >>> Some([]).and_then(lambda xs: xs[0] if xs else None)
        Nothing
        >>> Nothing.and_then(lambda xs: xs[0] if xs else None)
        Nothing
        """

    @abstractmethod
    def or_(self, default: Optional[T]) -> Maybe[T]:
        """
        Like ``a or b`` in Python, except only considering ``None``.

        >>> Some(1).or_(2)
        Some(1)
        >>> Some(1).or_(None)
        Some(1)
        >>> Nothing.or_(2)
        Some(2)
        """

    def __or__(self, default: Optional[T]) -> Maybe[T]:
        """
        Like ``a or b`` in Python, except only considering ``None``.

        >>> Some(1) | 2
        Some(1)
        >>> Some(1) | None
        Some(1)
        >>> Nothing | 2
        Some(2)
        """

        return self.or_(default)

    @abstractmethod
    def or_else(self, factory: Callable[[], Optional[T]]) -> Maybe[T]:
        """
        Like `unwrap_or_else`, except that it returns an optional value.

        >>> Some([42]).or_else(list)
        Some([42])
        >>> Nothing.or_else(list)
        Some([])
        """

    @abstractmethod
    def contains(self, other: U) -> bool:
        """
        Check if ``val`` equals ``other``, always return ``False`` if ``val``
        is ``None``.

        >>> Some(1).contains(1)
        True
        >>> Some(1).contains(2)
        False
        >>> Nothing.contains(3)
        False
        """

    @abstractmethod
    def filter(self, pred: Callable[[T], bool]) -> Maybe[T]:
        """
        Return ``None`` if ``val`` is ``None`` or if ``pred(val)`` does not
        return ``True``, otherwise return ``val``.

        >>> is_even = lambda n: n % 2 == 0
        >>> Some(1).filter(is_even)
        Nothing
        >>> Some(2).filter(is_even)
        Some(2)
        >>> Nothing.filter(is_even)
        Nothing
        """

    @abstractmethod
    def xor(self, other: OptionalOrMaybe[T]) -> Maybe[T]:
        """
        Return either ``val`` or ``other`` if the other is ``None``. Also
        return ``None`` if both are not ``None``.

        >>> xor(1, None)
        1
        >>> xor(None, 2)
        2
        >>> xor(1, 2)
        >>> xor(None, None)
        """

    def __xor__(self, other: OptionalOrMaybe[T]) -> Maybe[T]:
        return self.xor(other)

    @abstractmethod
    def iter(self) -> List[T]:
        """
        Wrap ``val`` into a list, if it is not ``None``.

        Allows to use for loops on optional values.
        """

    def __iter__(self):
        yield from self.iter()

    @abstractmethod
    def zip(self, other: OptionalOrMaybe[U]) -> Maybe[Tuple[T, U]]:
        """
        Abstract zip.
        """

    @abstractmethod
    def zip_with(
        self, other: OptionalOrMaybe[U], fn: Callable[[T, U], R]
    ) -> Maybe[R]:
        """Apply function to two optional values, if neither of them is
        ``None``:

        >>> add = lambda left, right: left + right
        >>> Some(1).zip_with(2, add)
        Some(3)
        >>> Some(1).zip_with(None, add)
        Nothing
        >>> Nothing.zip_with(2, add)
        Nothing
        """

    @abstractmethod
    def flatten(self: "Maybe[OptionalOrMaybe[T]]") -> Maybe[T]:
        """
        Flatten nested optional value.

        Note: This just returns the value, but changes the type from
        ``Optional[Optional[T]]`` to ``Optional[T].``
        """


@dataclass
@final
class Some(Maybe[T]):
    val: T

    def __repr__(self):
        return f"Some({self.val!r})"

    def unbox(self) -> Optional[T]:
        return self.val

    def is_some(self):
        return True

    def unwrap(self) -> T:
        return self.val

    def expect(self, msg: str) -> T:
        return self.val

    def do(self, fn: Callable[[T], U]) -> Maybe[T]:
        fn(self.val)

        return self

    def map(
        self, fn: Callable[Concatenate[T, P], U], *args, **kwargs
    ) -> Maybe[U]:
        return Some(fn(self.val, *args, **kwargs))

    def map_or(self, fn: Callable[[T], U], default: U) -> U:
        return self.map(fn).unwrap()

    def map_or_else(
        self,
        fn: Callable[[T], U],
        factory: Callable[[], U],
    ) -> U:
        return self.map(fn).unwrap()

    def unwrap_or(self, default: T) -> T:
        return self.unwrap()

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return self.unwrap()

    def and_(self, other: OptionalOrMaybe[U]) -> Maybe[U]:
        return box(other)

    def __and__(self, other: OptionalOrMaybe[U]) -> Maybe[U]:
        return self.and_(other)

    def and_then(
        self,
        fn: Callable[Concatenate[T, P], OptionalOrMaybe[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Maybe[U]:
        return box(fn(self.val, *args, **kwargs))

    def or_(self, default: Optional[T]) -> Maybe[T]:
        return self

    def __or__(self, default: Optional[T]) -> Maybe[T]:
        return self.or_(default)

    def or_else(self, factory: Callable[[], Optional[T]]) -> Maybe[T]:
        return self

    def contains(self, other: U) -> bool:
        return self.val == other

    def filter(self, pred: Callable[[T], bool]) -> Maybe[T]:
        if pred(self.val):
            return self

        return Nothing

    def xor(self, other: OptionalOrMaybe[T]) -> Maybe[T]:
        other = box(other)

        if other.is_none():
            return self

        return Nothing

    def iter(self) -> List[T]:
        return [self.val]

    def zip(self, other: OptionalOrMaybe[U]) -> Maybe[Tuple[T, U]]:
        other = box(other)
        if other.is_some():
            return Some((self.unwrap(), other.unwrap()))

        return Nothing

    def zip_with(
        self, other: OptionalOrMaybe[U], fn: Callable[[T, U], R]
    ) -> Maybe[R]:
        zipped = self.zip(other)

        if zipped.is_some():
            return box(fn(*zipped.unwrap()))

        return Nothing

    def flatten(self: "Maybe[OptionalOrMaybe[T]]") -> Maybe[T]:
        return box(self.unwrap())


@final
class _Nothing(Maybe[T]):
    def __repr__(self):
        return "Nothing"

    def unbox(self) -> Optional[T]:
        return None

    def is_some(self):
        return False

    def unwrap(self) -> T:
        self.expect("Trying to unwrap `None` value.")

        assert False

    def expect(self, msg: str) -> T:
        raise ValueError(msg)

    def do(self, fn: Callable[[T], U]) -> Maybe[T]:
        return self

    def map(
        self, fn: Callable[Concatenate[T, P], U], *args, **kwargs
    ) -> Maybe[U]:
        return Nothing

    def map_or(self, fn: Callable[[T], U], default: U) -> U:
        return default

    def map_or_else(
        self,
        fn: Callable[[T], U],
        factory: Callable[[], U],
    ) -> U:
        return factory()

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return fn()

    def and_(self, other: OptionalOrMaybe[U]) -> Maybe[U]:
        return Nothing

    def and_then(
        self,
        fn: Callable[Concatenate[T, P], OptionalOrMaybe[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Maybe[U]:
        return Nothing

    def or_(self, default: Optional[T]) -> Maybe[T]:
        return box(default)

    def or_else(self, factory: Callable[[], Optional[T]]) -> Maybe[T]:
        return box(factory())

    def contains(self, other: U) -> bool:
        return False

    def filter(self, pred: Callable[[T], bool]) -> Maybe[T]:
        return self

    def xor(self, other: OptionalOrMaybe[T]) -> Maybe[T]:
        return box(other)

    def iter(self) -> List[T]:
        return []

    def zip(self, other: OptionalOrMaybe[U]) -> Maybe[Tuple[T, U]]:
        return Nothing

    def zip_with(
        self, other: OptionalOrMaybe[U], fn: Callable[[T, U], R]
    ) -> Maybe[R]:
        return Nothing

    def flatten(self: "Maybe[OptionalOrMaybe[T]]") -> Maybe[T]:
        return cast(Maybe[T], self)


Nothing: _Nothing[Any] = _Nothing()
