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

# from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    cast,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

import pydantic.dataclasses
from pydantic.fields import FieldInfo

T = TypeVar("T")


class _EventualType:
    __init_kwargs__: dict = {}


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
    """A default field for a dataclass, that uses a function to calculate the
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
    repr=False,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
):
    """Custom dataclass wrapper for serde.


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
    repr=False,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
):
    fields = []
    orelse_fields = {}
    # eventual_fields = {}
    optional_fields = []
    init_fields = []

    class Config:
        arbitrary_types_allowed = True

    dc = pydantic.dataclasses.dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
        config=Config,
    )(cls)

    for name in getattr(cls, "__annotations__", {}):
        init_fields.append(name)

    for name, field in getattr(dc, "__dataclass_fields__", {}).items():
        field: dataclasses.Field
        # skip if: field(init=False)
        if not field.init:
            init_fields.remove(name)
            continue

        # skip ClassVar
        # TODO: how to recognize ClassVar[int]
        if field.type is ClassVar:
            init_fields.remove(name)
            continue

        if (
            hasattr(field.type, "__args__")
            and len(field.type.__args__) == 2
            and field.type.__args__[-1] is type(None)
        ):
            # Check if exactly two arguments exists and one of them are None type
            optional_fields.append(name)

        fields.append(name)

        if isinstance(field.default, FieldInfo):
            """
            if field.default.default is EVENTUAL:
                eventual_fields[name] = Eventual(field.default.default)
            """

            if isinstance(field.default.default, OrElse):
                orelse_fields[name] = field.default.default

    orelse_order = {}

    if orelse_fields:
        available_fields = set(fields)

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

    orig_init = dc.__init__

    def _init_(self, *args, **kwargs):
        # assert len(args) <= len(fields), "Too many arguments"
        # assert len(args) + len(kwargs) <= len(fields)

        init_kwargs = {
            field_name: arg for arg, field_name in zip(args, init_fields)
        }
        init_kwargs.update(kwargs)

        for orelse_name, orelse in orelse_order.items():
            if orelse_name not in init_kwargs:
                value = orelse._call(init_kwargs)
                init_kwargs[orelse_name] = value

        """
        if eventual_fields:
            eventual_kwargs = {
                key: Eventual(getattr(validated_model, key, EVENTUAL))
                for key in eventual_fields
            }

            self.__class__.__eventually__(
                SimpleNamespace(**init_kwargs),
                **eventual_kwargs,
            )

            for name, value in eventual_kwargs.items():
                eventual_kwargs[name] = value.unwrap()

            init_kwargs.update(eventual_kwargs)
        """

        object.__setattr__(
            self,
            "__init_kwargs__",
            dict(init_kwargs),
        )

        kwargs_attr_dict = {}
        for field in list(init_kwargs.keys()):
            delete_field = False
            if field not in fields:
                kwargs_attr_dict[field] = init_kwargs[field]
                delete_field = True
            if init_kwargs[field] is None and field not in optional_fields:
                delete_field = True
            if delete_field:
                del init_kwargs[field]

        object.__setattr__(
            self,
            "kwargs",
            kwargs_attr_dict,
        )

        orig_init(self, **init_kwargs)

    def _repr_(self) -> str:
        from gluonts.core.serde import dump_code

        return dump_code(self)

    dc.__init__ = _init_
    dc.__repr__ = _repr_

    return dc
