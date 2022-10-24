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

__all__ = [
    # types
    "Type",
    "Default",
    "Array",
    "Period",
    # this module
    "Schema",
]

import functools
from dataclasses import dataclass, field, MISSING
from operator import attrgetter, methodcaller
from typing import Any, Dict

from gluonts.itertools import Map
from .types import Type, Default, Array, Period


@dataclass
class IfSet:
    condition: bool
    default: Any = MISSING

    def apply(self, schema, name, ty):
        return schema.add_if(self.condition, name, ty, default=self.default)


ignore = IfSet(False)


@dataclass
class Schema:
    fields: Dict[str, Type] = field(default_factory=dict)
    default_fields: Dict[str, Any] = field(init=False)

    @classmethod
    def common(
        cls,
        *,
        dtype,
        multivariate=False,
        feat_dynamic_real: IfSet = ignore,
        feat_static_cat: IfSet = ignore,
        feat_static_real: IfSet = ignore,
    ):
        schema = cls()

        if multivariate:
            ndim = 2
        else:
            ndim = 1

        schema.add("target", Array(dtype, ndim=ndim, time_dim=ndim - 1))
        schema.add("start", Period())

        feat_dynamic_real.apply(
            schema, "feat_dynamic_real", Array(dtype, ndim=2, time_dim=1)
        )
        feat_static_cat.apply(
            schema, "feat_static_cat", Array(dtype=dtype, ndim=1)
        )
        feat_static_real.apply(
            schema, "feat_static_real", Array(dtype=dtype, ndim=1)
        )

        return schema

    def __post_init__(self):
        self.default_fields = {}

        for name, ty in list(self.fields.items()):
            if isinstance(ty, Default):
                self.default_fields[name] = self.fields.pop(name).value

    def apply(self, entry):
        result = {
            field_name: ty.apply(entry[field_name])
            for field_name, ty in self.fields.items()
        }

        for name, default in self.default_fields.items():
            result[name] = default
        return result

    def add(self, name: str, ty: Type):
        if when is None or when:
            self.fields[name] = ty
        elif default is not MISSING:
            self.default_fields[name] = ty.apply(default)

        return self

    def add_if(self, condition: bool, name: str, ty: Type, default=MISSING):
        if not condition:
            if default is not MISSING:
                self.default_fields[name] = ty.apply(default)
            return self

        return self.add(name, ty)


def with_schema(*, method=None, schema=None, attribute=None):
    num_chosen = (
        (method is not None) + (schema is not None) + (attribute is not None)
    )

    if num_chosen != 1:
        raise ValueError(
            "Exactly one of `method`, `schema`, or `attribute` has to be specified."
        )

    if attribute:
        get_schema = attrgetter(attribute)
    elif method:
        get_schema = methodcaller(method)
    else:
        get_schema = lambda self: schema

    def decorator(train):
        @functools.wraps(train)
        def wrapper(
            self,
            training_data,
            validation_data=None,
        ):
            schema = get_schema(self)

            training_data = Map(schema.apply, training_data)
            if validation_data is not None:
                validation_data = Map(schema.apply, validation_data)

            return train(self, training_data, validation_data)

        return wrapper

    return decorator
