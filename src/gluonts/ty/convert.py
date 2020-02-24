import pydantic
import pandas as pd
import numpy as np
from typing import Any, Type, TypeVar, Sequence, Optional, Union


def as_numpy_array(xs, dtype, ndim):
    array = np.asarray(xs, dtype=dtype)
    assert ndim == array.ndim
    return array


T = TypeVar("T")


class BaseArray(np.ndarray):
    pass


def Array(dtype, ndim: int) -> Type[BaseArray]:
    class Array_(BaseArray, Sequence):
        @classmethod
        def __get_validators__(cls):
            return [lambda xs: as_numpy_array(xs, dtype=dtype, ndim=ndim)]

    return Array_


FeatStaticCat = Array(dtype=np.int32, ndim=1)
FeatDynamicCat = Array(dtype=np.int32, ndim=2)
FeatStaticReal = Array(dtype=np.float32, ndim=1)

T_float = Array(dtype=np.float32, ndim=1)


def AtLeast2D(dtype):
    def as_numpy_array(xs):
        return np.atleast_2d(np.asarray(xs, dtype))

    class AtLeast2DArray(BaseArray, Sequence):
        @classmethod
        def __get_validators__(cls):
            return [as_numpy_array]

    return AtLeast2DArray


class Timestamp(pd.Timestamp):
    # we need to sublcass, since pydantic otherwise converts the value into
    # datetime.datetime instead of using pd.Timestamp
    @classmethod
    def __get_validators__(cls):
        def conv(val):
            if isinstance(val, pd.Timestamp):
                return val
            else:
                return pd.Timestamp(val)

        yield conv


Int1D = Array(dtype=np.int32, ndim=1)
Int2D = Array(dtype=np.int32, ndim=2)

Real1D = Array(dtype=np.float, ndim=1)
Real2D = Array(dtype=np.float, ndim=2)


class DataModel(pydantic.BaseModel):
    class Config:
        validate_all = True

    # start: Timestamp
    target: AtLeast2D(float)
    feat_static_cat: Int1D = [0]


# print(DataModel(target=[1, 2, 3]).dict())


# for _ in range(100_000):
# DataModel(target=[1, 2, 3])


from time import time

data = list(map(str, range(1_000)))

t = time()

for _ in range(100_000):
    DataModel(target=data)

end = time() - t
print(end)


# t = time()

# for _ in range(100_000):
#     as_numpy_array([data], np.float32, ndim=2)

# end = time() - t
# print(end)


# class ProcessTimeSeriesField:
#     """
#     Converts a time series field identified by `name` from a list of numbers
#     into a numpy array.

#     Constructor parameters modify the conversion logic in the following way:

#     If `is_required=True`, throws a `GluonTSDataError` if the field is not
#     present in the `Data` dictionary.

#     If `is_cat=True`, the array type is `np.int32`, otherwise it is
#     `np.float32`.

#     If `is_static=True`, asserts that the resulting array is 1D,
#     otherwise asserts that the resulting array is 2D. 2D dynamic arrays of
#     shape (T) are automatically expanded to shape (1,T).

#     Parameters
#     ----------
#     name
#         Name of the field to process.
#     is_required
#         Whether the field must be present.
#     is_cat
#         Whether the field refers to categorical (i.e. integer) values.
#     is_static
#         Whether the field is supposed to have a time dimension.
#     """

#     # TODO: find a fast way to assert absence of nans.

#     def __init__(
#         self, name, is_required: bool, is_static: bool, is_cat: bool
#     ) -> None:
#         self.name = name
#         self.is_required = is_required
#         self.req_ndim = 1 if is_static else 2
#         self.dtype = np.int32 if is_cat else np.float32

#     def __call__(self, data: DataEntry) -> DataEntry:
#         value = data.get(self.name, None)
#         if value is not None:
#             value = np.asarray(value, dtype=self.dtype)

#             if self.req_ndim != value.ndim:
#                 raise GluonTSDataError(
#                     f"Array '{self.name}' has bad shape - expected "
#                     f"{self.req_ndim} dimensions, got {value.ndim}."
#                 )

#             data[self.name] = value

#             return data
#         elif not self.is_required:
#             return data
#         else:
#             raise GluonTSDataError(
#                 f"Object is missing a required field `{self.name}`"
#             )
