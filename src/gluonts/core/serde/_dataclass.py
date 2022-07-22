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
    Type,
    TypeVar,
)

import pydantic.dataclasses
from pydantic import create_model

T = TypeVar("T")


class _EventualType:
    __init_kwargs__: dict = {}


EVENTUAL = cast(Any, _EventualType())


@dataclasses.dataclass
class Eventual(Generic[T]):
    value: T

    def insert(self, value: T) -> None:
        self.value = value

    def insert_default(self, value: T) -> None:
        if self.value is EVENTUAL:
            self.value = value

    def unwrap(self) -> T:
        if self.value is EVENTUAL:
            raise ValueError("UNWRAP")

        return self.value


if TYPE_CHECKING:
    AnyLike = Any
else:
    AnyLike = object


@dataclasses.dataclass
class OrElse(AnyLike):
    fn: Callable

    def get_parameters(self):
        return set(inspect.signature(self.fn).parameters)

    def call(self, values):
        kwargs = {attr: values[attr] for attr in self.get_parameters()}
        return self.fn(**kwargs)


def dataclass(cls):
    fields = {}
    orelse_fields = {}
    eventual_fields = set()

    for name, ty in getattr(cls, "__annotations__", {}).items():
        value = cls.__dict__.get(name, dataclasses.MISSING)

        # skip if: field(init=False)
        if isinstance(value, dataclasses.Field):
            if not value.init:
                continue
            default = value.default
        else:
            default = value

        if default is dataclasses.MISSING:
            default = ...

        # skip ClassVar
        if ty is ClassVar or getattr(cls, "__origin__", None) is ClassVar:
            continue

        # if InitVar is used, we extract the inner value
        if ty is dataclasses.InitVar:
            ty = Any
        elif isinstance(ty, dataclasses.InitVar):
            ty = ty.type

        if default is EVENTUAL:
            eventual_fields.add(name)

        if isinstance(default, OrElse):
            orelse_fields[name] = default

        fields[name] = ty, default

    orelse_order = {}

    if orelse_fields:
        available_fields = set(fields)

        unhandled = dict(orelse_fields)

        while unhandled:
            for orelse_name, orelse in unhandled.items():
                params = orelse.get_parameters()

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

    class Config:
        """
        `Config <https://pydantic-docs.helpmanual.io/#model-config>`_ for the
        Pydantic model inherited by all :func:`validated` initializers.

        Allows the use of arbitrary type annotations in initializer parameters.
        """

        arbitrary_types_allowed = True

    model = create_model(f"{cls.__name__}Model", __config__=Config, **fields)

    dc = pydantic.dataclasses.dataclass(config=Config)(cls)

    orig_init = dc.__init__

    def _init_(self, *args, **kwargs):
        # assert len(args) <= len(fields), "Too many arguments"
        # assert len(args) + len(kwargs) <= len(fields)

        input_kwargs = {
            field_name: arg for arg, field_name in zip(args, fields)
        }
        input_kwargs.update(kwargs)

        validated_model = model(**input_kwargs)
        init_kwargs = validated_model.dict()

        for orelse_name, orelse in orelse_order.items():
            value = orelse.call(init_kwargs)
            init_kwargs[orelse_name] = value

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
                for key, value in validated_model.dict().items()
                if key in input_kwargs
            },
        )

        orig_init(self, **init_kwargs)

    dc.__init__ = _init_
    return dc
