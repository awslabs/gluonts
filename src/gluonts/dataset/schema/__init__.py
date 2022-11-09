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
from itertools import chain
from typing import Any, Dict, List, Optional

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
    default_fields: Dict[str, Any] = field(default_factory=dict)

    past_only_arrays: List[str] = field(init=False, default_factory=list)
    past_future_arrays: List[str] = field(init=False, default_factory=list)
    static_values: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        for name, ty in self.fields.items():
            self.classify_field(name, ty)

    def copy(self):
        return Schema(self.fields.copy(), self.default_fields.copy())

    def classify_field(self, name, ty):
        if isinstance(ty, Array) and ty.time_axis is not None:
            if ty.past_only:
                self.past_only_arrays.append(name)
            else:
                self.past_future_arrays.append(name)
        else:
            self.static_values.append(name)

    def set(self, name: str, ty: Type):
        if isinstance(ty, Default):
            self.default_fields[name] = ty.value
        else:
            self.fields[name] = ty
            self.classify_field(name, ty)

        return self

    def set_if(self, condition: bool, name: str, ty: Type, default=None):
        if not condition:
            if default is not None:
                self.default_fields[name] = ty.apply(default)
            return self

        return self.set(name, ty)

    def extend(self, name, ty):
        return self.copy().set(name, ty)

    def validate(self, entry: dict, future_length: int = 0) -> "TimeSeries":
        result = {
            field_name: ty.apply(entry[field_name])
            for field_name, ty in self.fields.items()
        }

        for name, default in self.default_fields.items():
            result[name] = default

        lengths = []
        for name in chain(self.past_only_arrays, self.past_future_arrays):
            ty = self.fields[name]
            array = result[name]

            if array.type.past_only:
                lengths.append(array.time_length + future_length)
            else:
                lengths.append(array.time_length)

        length = lengths[0]
        assert np.all(np.array(lengths) == length)

        return TimeSeries(result, self, length, future_length)


@dataclass
class RequiredIf:
    condition: bool
    default: Optional[Any] = None

    def apply(self, schema, name, ty):
        return schema.set_if(self.condition, name, ty, default=self.default)


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
    schema.set(
        "target",
        Array(dtype=dtype, ndim=ndim, time_axis=ndim - 1, past_only=True),
    )
    schema.set("start", Period(freq=freq))

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

    def extend(self, name, ty, value):
        self.schema = self.schema.extend(name, ty)

        value = ty.apply(value)
        self.data[name] = value

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self.data[key]

    @property
    def past(self) -> dict:
        result = {}

        for field_name in self.schema.static_values:
            result[field_name] = self.data[field_name]

        for field_name in self.schema.past_only_arrays:
            result[field_name] = self.data[field_name]

        for field_name in self.schema.past_future_arrays:
            result[field_name] = self.data[field_name].split_time(
                -self.future_length
            )[0]

        return result

    @property
    def future(self):
        result = {}

        for field_name in self.schema.static_values:
            result[field_name] = self.data[field_name]

        for field_name in self.schema.past_future_arrays:
            result[field_name] = self.data[field_name].split_time(
                -self.future_length
            )[1]

        return result

    def split(self, idx, limit=None):
        assert self.future_length == 0

        if limit is None:
            stop = None
        else:
            stop = idx + limit

        result = {}
        for field_name in self.schema.static_values:
            result[field_name] = self.data[field_name]

        for field_name in chain(
            self.schema.past_only_arrays, self.schema.past_future_arrays
        ):
            past, future = self.data[field_name].split_time(idx, stop)

            result["past_" + field_name] = past
            result["future_" + field_name] = future

        return result
