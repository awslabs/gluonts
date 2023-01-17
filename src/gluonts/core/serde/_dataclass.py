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

import dataclasses
import inspect
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    cast,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

import pydantic
import pydantic.dataclasses

from gluonts.itertools import select

T = TypeVar("T")


class _EventualType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

        return cls._instance

    def __repr__(self):
        return "EVENTUAL"


EVENTUAL = cast(Any, _EventualType())


@dataclasses.dataclass
class Eventual(Generic[T]):
    value: T

    def set(self, value: T) -> None:
        self.value = value

    def set_default(self, value: T) -> None:
        if self.value is EVENTUAL:
            self.value = value

    def unwrap(self) -> T:
        if self.value is EVENTUAL:
            raise ValueError("UNWRAP")

        return self.value


# OrElse should have `Any` type, meaning that we can do
#     x: int = OrElse(...)
# without mypy complaining. We achieve this by nominally deriving from Any
# during type checking. Because it's not actually possible to derive from Any
# in Python, we inherit from object during runtime instead.
if TYPE_CHECKING:
    AnyLike = Any
else:
    AnyLike = object


@dataclasses.dataclass
class OrElse(AnyLike):
    """
    A default field for a dataclass, that uses a function to calculate the
    value if none was passed.

    The function can take arguments, which are other fields of the annotated
    dataclass, which can also be `OrElse` fields. In case of circular
    dependencies an error is thrown.

    ::

        @serde.dataclass
        class X:
            a: int
            b: int = serde.OrElse(lambda a: a + 1)
            c: int = serde.OrEsle(lambda c: c * 2)

        x = X(1)
        assert x.b == 2
        assert x.c == 4
    ```
    """

    fn: Callable
    parameters: dict = dataclasses.field(init=False)

    def __post_init__(self):
        self.parameters = set(inspect.signature(self.fn).parameters)

    def _call(self, env):
        kwargs = {attr: env[attr] for attr in self.parameters}
        return self.fn(**kwargs)


def dataclass(
    cls=None,
    *,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
):
    """
    Custom dataclass wrapper for serde.

    This works similar to the ``dataclasses.dataclass`` and
    ``pydantic.dataclasses.dataclass`` decorators. Similar to the ``pydantic``
    version, this does type checking of arguments during runtime.
    """

    # assert frozen
    assert init

    if cls is None:
        return _dataclass

    return _dataclass(cls, init, repr, eq, order, unsafe_hash, frozen)


def _dataclass(
    cls,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=True,
):
    class Config:
        arbitrary_types_allowed = True

    # make `cls` a dataclass
    pydantic.dataclasses.dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
        config=Config,
    )(cls)

    model_fields = {}
    orelse_fields = {}
    eventual_fields = set()

    # Collect init args to pass to pydantic model
    # Also, collect eventual and orelse fields
    for name, field in cls.__dataclass_fields__.items():
        if not field.init:
            continue

        if field.default is EVENTUAL:
            eventual_fields.add(name)
            continue

        elif isinstance(field.default, OrElse):
            orelse_fields[name] = field.default
            continue

        # now let's look at the type
        ty = field.type

        # skip ClassVar
        if (
            ty is ClassVar
            # `ClassVar[X].__origin__ is ClassVar`
            or getattr(ty, "__origin__", None) is ClassVar
        ):
            continue

        # if InitVar is used, we extract the inner value
        if ty is dataclasses.InitVar:
            ty = Any
        elif isinstance(ty, dataclasses.InitVar):
            ty = ty.type

        # do we have a default value set
        if not isinstance(field.default, dataclasses._MISSING_TYPE):
            model_fields[name] = ty, pydantic.Field(default=field.default)
        # do we have a default_factory set
        elif not isinstance(field.default_factory, dataclasses._MISSING_TYPE):
            model_fields[name] = ty, pydantic.Field(
                default_factory=field.default_factory
            )
        # no default
        else:
            model_fields[name] = ty, ...

    # figure out in which order to resolve `OrElse`, since they can depend on
    # each other -- however, no cycles!
    orelse_order = {}

    if orelse_fields:
        available_fields = set(cls.__dataclass_fields__)

        unhandled = dict(orelse_fields)

        while unhandled:
            for orelse_name, orelse in unhandled.items():
                params = orelse.parameters

                if params <= available_fields:
                    available_fields.add(orelse_name)
                    orelse_order[orelse_name] = orelse
                    del unhandled[orelse_name]
                    break
            else:
                # we made one pass over unhandled without resolving anything
                raise ValueError(
                    f"Can not resolve dependencies of {unhandled}"
                )

    model = pydantic.create_model(
        f"{cls.__name__}Model", __config__=Config, **model_fields
    )

    orig_init = cls.__init__

    def _init_(self, *args, **input_kwargs):
        if args:
            raise TypeError("serde.dataclass is always `kw_only`.")

        validated_model = model(**input_kwargs)
        # .dict() turns its values into dicts as well, so we just get the
        # attributes directly from the model
        init_kwargs = {
            key: getattr(validated_model, key)
            for key in validated_model.dict()
        }

        for orelse_name, orelse in orelse_order.items():
            value = orelse._call(init_kwargs)
            init_kwargs[orelse_name] = value

        if eventual_fields:
            eventual_kwargs = {
                key: Eventual(input_kwargs.get(key, EVENTUAL))
                for key in eventual_fields
            }

            self.__class__.__eventually__(
                SimpleNamespace(**init_kwargs),
                **eventual_kwargs,
            )

            for name, value in eventual_kwargs.items():
                eventual_kwargs[name] = value.unwrap()

            init_kwargs.update(eventual_kwargs)

        object.__setattr__(
            self,
            "__init_kwargs__",
            init_kwargs,
        )

        object.__setattr__(
            self,
            "__init_passed_kwargs__",
            {
                key: value
                for key, value in select(input_kwargs, init_kwargs).items()
            },
        )

        orig_init(self, **init_kwargs)

    cls.__init__ = _init_
    return cls


if TYPE_CHECKING:
    dataclass = dataclasses.dataclass  # type: ignore # noqa
