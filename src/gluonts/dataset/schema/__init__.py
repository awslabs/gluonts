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

from dataclasses import dataclass, field, MISSING
from typing import Any, Dict, Optional

import numpy as np

from .translate import Translator
from .types import Type, Array, Default, Period

__all__ = [
    # this module
    "Schema",
    # types
    "Type",
    "Array",
    "Default",
    "Period",
    # translate
    "Translator",
]


@dataclass
class Schema:
    fields: Dict[str, Type] = field(default_factory=dict)
    default_fields: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.default_fields = {}

        for name, ty in list(self.fields.items()):
            if isinstance(ty, Default):
                self.default_fields[name] = self.fields.pop(name).value

    def apply(self, entry) -> dict:
        result = {
            field_name: ty.apply(entry[field_name])
            for field_name, ty in self.fields.items()
        }

        for name, default in self.default_fields.items():
            result[name] = default

        return result

    def add(self, name: str, ty: Type):
        self.fields[name] = ty

        return self

    def add_if(self, condition: bool, name: str, ty: Type, default=None):
        if not condition:
            if default is not None:
                self.default_fields[name] = ty.apply(default)
            return self

        return self.add(name, ty)


@dataclass
class RequiredIf:
    condition: bool
    default: Optional[Any] = None

    def apply(self, schema, name, ty):
        return schema.add_if(self.condition, name, ty, default=self.default)


def common(
    *,
    dtype,
    freq=None,
    multivariate=False,
    feat_dynamic_real=RequiredIf(False),
    feat_static_cat=RequiredIf(False),
    feat_static_real=RequiredIf(False),
):
    if multivariate:
        ndim = 2
    else:
        ndim = 1

    schema = Schema()
    schema.add(
        "target",
        Array(dtype=dtype, ndim=ndim, time_axis=ndim - 1, past_only=True),
    )
    schema.add("start", Period(freq=freq))

    feat_dynamic_real.apply(
        schema, "feat_dynamic_real", Array(dtype=dtype, ndim=2, time_axis=1)
    )
    feat_static_cat.apply(
        schema, "feat_static_cat", Array(dtype=dtype, ndim=1)
    )
    feat_static_real.apply(
        schema, "feat_static_real", Array(dtype=dtype, ndim=1)
    )

    return schema


@dataclass
class TimeSeries:
    data: dict
    schema: Schema
    length: int
    future_length: int = 0

    @classmethod
    def load(cls, schema, data: dict, future_length: int = 0):
        data = schema.apply(data)

        past_lengths = [
            ty.time_dim(data[name]) + future_length
            for name, ty in schema.fields.items()
            if isinstance(ty, Array) and ty.past_only
        ]
        future_lengths = [
            ty.time_dim(data[name])
            for name, ty in schema.fields.items()
            if isinstance(ty, Array) and not ty.past_only
        ]

        all_lengths = np.array(past_lengths + future_lengths)

        assert np.all(all_lengths[0] == all_lengths)

        return cls(
            data=data,
            schema=schema,
            length=all_lengths[0],
            future_length=future_length,
        )

    def __len__(self):
        return self.length
