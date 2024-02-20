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


from typing import Any, Callable, Optional, Union, Type, Dict

import numpy as np

from gluonts.itertools import partition
from gluonts.pydantic import parse_obj_as, BaseModel

from ._freq import Freq
from ._period import Period
from ._time_frame import time_frame, TimeFrame
from ._split_frame import split_frame, SplitFrame


"""
This module provides tooling to extract ``zebras.TimeFrame`` and
``zebras.SplitFrame`` instances from Python dictionaries::

    schema = zebras.schema.Schema(...)
    tf = schema.load_timeframe(raw_dict)

    sf = schema.load_splitframe(raw_dict)

The idea is to define expected types and shapes for each field (column)
in a ``zebras.schema.Schema``:

    schema = zebras.schema.Schema({
        "target": zebras.schema.TimeSeries(),
        "time_feat": zebras.schema.TimeSeries(ndim=2, tdim=-1, past_only=False),
        "static_feat": zebras.schema.Array(ndim=1),

    })

"""


class Field(BaseModel):
    """
    Specification for user provided input data.
    """

    required: bool = True
    internal: bool = False
    default: Any = None
    preprocess: Optional[Callable] = None

    def load_from(self, data, name, **kwargs):
        try:
            value = self._get(data, name)
            return self._load(value, **kwargs)
        except Exception as err:
            raise ValueError(f"Error when processing field {name!r}.") from err

    def _get(self, data, name):
        if self.internal:
            value = self.default
        elif self.required:
            try:
                value = data[name]
            except KeyError:
                # TODO: Add error message
                raise
        else:
            value = data.get(name, self.default)

        if self.preprocess is not None:
            value = self.preprocess(value)

        return value


class Metadata(Field):
    type: Any = Any
    required: bool = False

    def _load(self, value):
        return parse_obj_as(self.type, value)


class Scalar(Field):
    type: Type

    def _load(self, value):
        return parse_obj_as(self.type, value)


class Array(Field):
    ndim: Optional[int] = None
    shape: Optional[tuple] = None
    dtype: Type = np.float32

    def _load(self, value):
        value = np.asarray(value, dtype=self.dtype)

        if not self.required and self.shape is not None and value.ndim == 0:
            value = np.full(self.shape, value.item())

        if self.ndim is not None:
            assert self.ndim == value.ndim

        return value


class TimeSeries(Field):
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
    dtype: Type = np.float32
    tdim: int = -1
    past_only: bool = True

    def _load(self, value, length: Optional[int] = None) -> np.ndarray:
        """
        Load field ``name`` from ``data`` and apply validation.

        Note: We do the lookup of the value in this function, since the field
        can be optional.
        """

        value = np.array(value, dtype=self.dtype)

        if value.ndim == 0 and not self.required:
            assert self.ndim is None or self.ndim == 1
            assert length is not None

            value = np.full(length, self.default)

        if self.ndim is not None:
            assert value.ndim == self.ndim, (
                "Incorrect number of dimensions. "
                f"Expected ndim = {self.ndim}, got: {value.ndim}"
            )

        return value


Fields = Dict[str, Field]


class Schema:
    fields: Fields

    def __init__(self, fields=None, **kwargs):
        if fields is not None:
            self.fields = {**fields, **kwargs}
        else:
            self.fields = kwargs

        self.columns = {}
        self.time_series_ref = None
        self.static = {}
        self.metadata = {}

        for name, ty in self.fields.items():
            if isinstance(ty, TimeSeries):
                self.columns[name] = ty
            elif isinstance(ty, Metadata):
                self.metadata[name] = ty
            elif isinstance(ty, (Scalar, Array)):
                self.static[name] = ty
            else:
                raise ValueError(f"Unknown field type for {name}")

        if self.columns:
            try:
                self.time_series_ref = next(
                    name for name in self.columns if self.fields[name].required
                )
            except StopIteration:
                raise ValueError(
                    "At least one time series needs to be required."
                )

    def _load_static(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Helper to load static data from ``data``.

        Used by ``load_timeframe`` and ``load_splitframe``.
        """
        return {
            name: field.load_from(data, name)
            for name, field in self.static.items()
        }

    def _load_metadata(self, data: Dict[str, Any]) -> Optional[dict]:
        if self.metadata:
            return {
                name: field.load_from(data, name)
                for name, field in self.metadata.items()
            }

        return None

    def load_timeframe(
        self,
        data: Dict[str, Any],
        start: Optional[Union[Period, str]] = None,
        freq: Optional[Union[Freq, str]] = None,
    ) -> TimeFrame:
        if self.time_series_ref is not None:
            ty = self.columns[self.time_series_ref]
            ref = ty.load_from(data, self.time_series_ref)
            length = ref.shape[ty.tdim]

            columns = {self.time_series_ref: ref}

            columns.update(
                {
                    name: field.load_from(data, name, length=length)
                    for name, field in self.columns.items()
                    if name != self.time_series_ref
                }
            )

        else:
            columns = {}

        return time_frame(
            columns,
            static=self._load_static(data),
            metadata=self._load_metadata(data),
            start=start,
            freq=freq,
        )

    def load_splitframe(
        self,
        data: Dict[str, Any],
        future_length: int,
        start: Optional[Union[Period, str]] = None,
        freq: Optional[Union[Freq, str]] = None,
    ) -> SplitFrame:
        if self.time_series_ref is not None:
            ty = self.columns[self.time_series_ref]
            ref = ty.load_from(data, self.time_series_ref)
            length = ref.shape[ty.tdim]

            if ty.past_only:
                past_length = length
            else:
                past_length = length - future_length

            past_fields, full_fields = partition(  # type: ignore
                self.columns.items(), lambda item: item[1].past_only
            )
            past = {
                name: field.load_from(data, name)
                for name, field in past_fields
            }
            full = {
                name: field.load_from(data, name)
                for name, field in full_fields
            }
        else:
            past_length = 0
            full = {}
            past = {}

        return split_frame(
            full,
            past=past,
            static=self._load_static(data),
            past_length=past_length,
            future_length=future_length,
            start=start,
            freq=freq,
        )
