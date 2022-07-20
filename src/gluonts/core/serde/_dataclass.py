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
from typing import cast, Any, ClassVar, Type

import pydantic.dataclasses
from pydantic import create_model


class Infer:
    pass


INFER = cast(Any, Infer())


@dataclasses.dataclass
class InferOption:
    value: Any

    def map(self, value):
        if self.value is INFER:
            self.value = value

    def unwrap(self):
        assert self.value is not INFER
        return self.value


def dataclass(cls):
    fields = {}
    inferred_fields = set()

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

        if default is INFER:
            inferred_fields.add(name)
            continue

        fields[name] = ty, default

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

        inferred_kwargs = {
            key: InferOption(input_kwargs.get(key, INFER))
            for key in inferred_fields
        }
        self.__class__.__infer__(
            validated_model,
            **inferred_kwargs,
        )
        for name, value in inferred_kwargs.items():
            inferred_kwargs[name] = value.unwrap()

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
                for key, value in init_kwargs.items()
                if key in input_kwargs
            },
        )

        orig_init(self, **init_kwargs, **inferred_kwargs)

    dc.__init__ = _init_
    return dc
