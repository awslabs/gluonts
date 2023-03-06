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


from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

import numpy as np
from toolz import valfilter, curry

from ._timeframe import time_frame, split_frame


@dataclass
class Field:
    ndims: Optional[int] = None
    tdim: Optional[int] = None
    dtype: Type = np.float32
    past_only: bool = False
    required: bool = True
    internal: bool = False
    default: Any = None
    preprocess: Optional[Callable] = None


@dataclass
class Schema:
    fields: dict

    @property
    def static(self):
        return valfilter(lambda field: field.tdim is None, self.fields)

    @property
    def columns(self):
        return valfilter(lambda field: field.tdim is not None, self.fields)

    def _load(self, name: str, ty: Field, data: dict):
        if ty.internal:
            value = ty.default

        elif ty.required:
            try:
                value = data[name]
            except KeyError:
                raise
        else:
            value = data.get(name, ty.default)

        if ty.preprocess is not None:
            return ty.preprocess(value)

        return value

    def _load_static(self, data):
        return {
            name: self._load(name, ty, data)
            for name, ty in self.static.items()
        }

    @curry
    def load_timeframe(self, data: dict, start=None, freq=None):
        columns = {
            name: self._load(name, ty, data)
            for name, ty in self.fields.items()
            if ty.tdim is not None
        }

        return time_frame(
            columns, static=self._load_static(data), start=start, freq=freq
        )

    @curry
    def load_splitframe(
        self, data: dict, future_length: int, start=None, freq=None
    ):
        past = {
            name: self._load(name, ty, data)
            for name, ty in self.fields.items()
            if ty.past_only
        }
        full = {
            name: self._load(name, ty, data)
            for name, ty in self.fields.items()
            if ty.tdim is not None and not ty.past_only
        }
        return split_frame(
            full,
            past=past,
            static=self._load_static(data),
            future_length=future_length,
            start=start,
            freq=freq,
        )
