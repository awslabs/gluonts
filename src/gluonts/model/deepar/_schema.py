
from typing import (
    Any,
    Dict,
    Optional,
    Union,
    Generic,
    TypeVar,
    Type
)

import numpy as np
import pandas as pd
from enum import Enum
from functools import lru_cache
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick
from gluonts.exceptions import GluonTSDataError


T = TypeVar("T")


class FieldType(Generic[T]):
    def __call__(self, v: Any) -> T:
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def is_compatible(self, v: Any) -> bool:
        raise NotImplementedError()


class NumpyArrayField(FieldType[np.ndarray]):
    def __init__(
        self,
        dtype: Type = np.float32,
        ndim: Optional[int] = None,
    ) -> None:
        self.dtype = dtype
        self.ndim = ndim

    def __eq__(self, other):
        if not isinstance(other, NumpyArrayField):
            return False
        return (
            self.dtype == other.dtype
            and self.ndim == other.ndim
        )

    def __repr__(self):
        return f"NumpyArrayField(dtype={self.dtype!r}, ndim={self.ndim!r}"

    def __call__(self, value: Any) -> np.ndarray:
        value = np.asarray(value, dtype=object)
        value = value.astype(self.dtype)
        if self.ndim is not None and self.ndim != value.ndim:
            raise GluonTSDataError(
                f"expected array with dimension {self.ndim}, but got {value.ndim}."
            )
        return value

    def is_compatible(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return False

        try:
            x = np.asarray(value)
        except:
            return False

        # int types
        if self.dtype in [np.int32, np.int64]:
            if x.dtype.kind != "i":
                return False
            return self.ndim is None or x.ndim == self.ndim

        # float types
        try:
            x = np.asarray(value, dtype=self.dtype)
        except:
            return False
        return self.ndim is None or x.ndim == self.ndim


class TimeZoneStrategy(Enum):
    ignore = "ignore"
    utc = "utc"
    error = "error"


Freq = Union[str, pd.DateOffset]


class PandasTimestampField(FieldType[pd.Timestamp]):
    def __init__(
        self,
        freq: Freq,
        tz_strategy: TimeZoneStrategy = TimeZoneStrategy.error,
    ) -> None:
        self.freq = to_offset(freq) if isinstance(freq, str) else freq
        self.tz_strategy = tz_strategy

    def __eq__(self, other):
        if not isinstance(other, PandasTimestampField):
            return False
        return (
            self.freq == other.freq
            and self.tz_strategy == other.tz_strategy
        )

    def __repr__(self):
        return f"PandasTimestampField(freq={self.freq!r}, tz_strategy={self.tz_strategy!r})"

    def __call__(self, value: Any) -> pd.Timestamp:
        timestamp = PandasTimestampField._process(value, self.freq)

        if timestamp.tz is not None:
            if self.tz_strategy == TimeZoneStrategy.error:
                raise GluonTSDataError("Timezone information is not supported")
            elif self.tz_strategy == TimeZoneStrategy.utc:
                # align timestamp to utc timezone
                timestamp = timestamp.tz_convert("UTC")

            # removes timezone information
            timestamp = timestamp.tz_localize(None)
        return timestamp

    @staticmethod
    @lru_cache(maxsize=10000)
    def _process(string: str, freq: pd.DateOffset) -> pd.Timestamp:
        timestamp = pd.Timestamp(string, freq=freq)

        # operate on time information (days, hours, minute, second)
        if isinstance(timestamp.freq, Tick):
            return pd.Timestamp(
                timestamp.floor(timestamp.freq), timestamp.freq
            )

        # since we are only interested in the data piece, we normalize the
        # time information
        timestamp = timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0, nanosecond=0
        )
        return timestamp.freq.rollforward(timestamp)

    def is_compatible(self, v: Any) -> bool:
        if not isinstance(v, (str, pd.Timestamp)):
            return False
        try:
            self(v)
        except:
            return False
        return True

class Schema:
    def __init__(self, fields: Dict[str, FieldType]) -> None:
        self.fields = fields
        # select fields that should be handled in the loop, because
        # - they are non optional
        # - or they are not AnyFields
        self._fields_for_processing = {
            k: f
            for k, f in self.fields.items()
        }

    def __eq__(self, other):
        if self.fields.keys() != other.fields.keys():
            return False
        return all(self.fields[k] == other.fields[k] for k in self.fields)

    def __repr__(self):
        return (
            "Schema(fields={"
            + ", ".join(f"'{k}':{v}" for k, v in self.fields.items())
            + "})"
        )

    def __call__(self, d: Dict[str, Any]) -> None:
        """
        Applies the schema to a data dict. The dictionary is updated in place.
        """
        for field_name, field_type in self._fields_for_processing.items():
            try:
                value = d[field_name]
            except KeyError:
                raise GluonTSDataError(
                        f"field {field_name} is not optional but key does not occur in the data"
                )
            try:
                d[field_name] = field_type(value)
            except Exception as e:
                raise GluonTSDataError(
                    f"Error when processing field {field_name} using {field_type}"
                ) from e