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
from typing import TYPE_CHECKING, Any, ClassVar

import pydantic.dataclasses
from pydantic import create_model


def dataclass(cls):
    fields = {}

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

        fields[name] = ty, default

    model = create_model(f"{cls.__name__}Model", **fields)

    dc = pydantic.dataclasses.dataclass(cls)

    orig_init = dc.__init__

    def _init_(self, *args, **kwargs):
        assert len(args) <= len(fields), "Too many arguments"
        assert len(args) + len(kwargs) <= len(fields)

        model_kwargs = {
            field_name: arg for arg, field_name in zip(args, fields)
        }
        model_kwargs.update(kwargs)

        object.__setattr__(
            self, "__init_kwargs__", model(**model_kwargs).dict()
        )

        orig_init(self, **self.__init_kwargs__)

    dc.__init__ = _init_
    return dc
