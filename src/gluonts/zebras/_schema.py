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
from typing import Any, Callable, Optional, Union, Type, Dict

import numpy as np
from toolz import valfilter

from gluonts.itertools import partition

from ._freq import Freq
from ._period import Period
from ._time_frame import time_frame, TimeFrame
from ._split_frame import split_frame, SplitFrame

"""
This module provides tooling to extract ``zebras.TimeFrame`` and
``zebras.SplitFrame`` instances from Python dictionaries::

    schema = zebras.Schema(...)
    tf = schema.load_timeframe(raw_dict)

    sf = schema.load_splitframe(raw_dict)

The idea is to define expected types and shapes for each field (column) using
``zebras.Field`` and then wrap them in a ``zebras.Schema``:

    schema = zebras.Schema({
        "target": Field(ndim=1, tdim=0, past_only=True),
        "time_feat": zebras.Field(ndim=2, tdim=-1),
        "static_feat": zebras.Field(ndim=1),

    })

See the documentation of ``Field`` on how to configure the columns of a schema.
"""


@dataclass
class Field:
    """
    Specification for user provided input data.

    Parameters
    ----------
    ndim, optional
        The expected number of dimensions of the input data. If provided, it is
        ensured that the input array has the expected number of dimensions.
    tdim, optional
        Mark an array as a time series and specifies which axis is the time
        dimension. When this value is ``None`` the array is classified as
        ``"static``".
    dtype
        The data type, passed to ``numpy.array``.
    past_only
        If the value is a time series, this marks if data is only expected for
        the past range when loading ``zebras.SplitFrame``. The value is ignored
        for static fields.
    required
        When set to true, the field has to be in the user data. Otherwise
        ``default`` is used as a fallback value.
    internal
        Allows to ignore user provided data when set, and instead ``default``
        is always used as the value.
    default
        The default value to use when either ``required`` or ``internal`` is
        set to true.
    preprocess, optional
        This function is called on the value before validating the value. For
        example, one can set ``preprocess = np.atleast_2d`` to also allow
        one dimensional arrays as input even when ``ndim = 2``.
    """

    ndim: Optional[int] = None
    tdim: Optional[int] = None
    dtype: Type = np.float32
    past_only: bool = False
    required: bool = True
    internal: bool = False
    default: Any = None
    preprocess: Optional[Callable] = None

    def validate(self, value: Any, name: str) -> np.ndarray:
        value = np.array(value, dtype=self.dtype)

        if self.ndim is not None:
            assert value.ndim == self.ndim, (
                f"Field {name} has incorrect number of dimensions. "
                f"Expected ndim = {self.ndim}, got: {value.ndim}"
            )

        return value

    def load_from(self, dct: dict, name: str) -> np.ndarray:
        """Load field ``name`` from ``dct`` and apply validation.

        Note: We do the lookup of the value in this function, since the field
        can be optional.
        """
        if self.internal:
            value = self.default

        elif self.required:
            try:
                value = dct[name]
            except KeyError:
                # TODO: Add error message
                raise
        else:
            value = dct.get(name, self.default)

        if self.preprocess is not None:
            return self.preprocess(value)

        value = self.validate(value, name)

        return value


Fields = Dict[str, Field]


@dataclass
class Schema:
    fields: Fields

    @property
    def static(self) -> Fields:
        """Return only fields that are static (where ``tdim`` is ``None``)."""
        return valfilter(lambda field: field.tdim is None, self.fields)

    @property
    def columns(self) -> Fields:
        """Return time series fields (where ``tdim`` is defined)."""

        return valfilter(lambda field: field.tdim is not None, self.fields)

    def _load_static(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Helper to load static data from ``data``.

        Used by ``load_timeframe`` and ``load_splitframe``.
        """
        return {
            name: field.load_from(data, name)
            for name, field in self.static.items()
        }

    def load_timeframe(
        self,
        data: Dict[str, Any],
        start: Optional[Union[Period, str]] = None,
        freq: Optional[Union[Freq, str]] = None,
    ) -> TimeFrame:
        columns = {
            name: field.load_from(data, name)
            for name, field in self.columns.items()
        }

        return time_frame(
            columns,
            static=self._load_static(data),
            start=start,
            freq=freq,
        )

    def load_splitframe(
        self,
        data: Dict[str, Any],
        future_length: Optional[int] = None,
        start: Optional[Union[Period, str]] = None,
        freq: Optional[Union[Freq, str]] = None,
    ) -> SplitFrame:
        past_fields, full_fields = partition(  # type: ignore
            self.columns.items(), lambda item: item[1].past_only
        )
        past = {
            name: field.load_from(data, name) for name, field in past_fields
        }
        full = {
            name: field.load_from(data, name) for name, field in full_fields
        }

        return split_frame(
            full,
            past=past,
            static=self._load_static(data),
            future_length=future_length,
            start=start,
            freq=freq,
        )
