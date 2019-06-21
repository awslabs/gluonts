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

# Standard library imports
import abc
from collections import Counter
from functools import lru_cache, reduce
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import DType, validated
from gluonts.core.exception import GluonTSDateBoundsError, assert_data_error
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.stat import ScaleHistogram
from gluonts.time_feature import TimeFeature
from gluonts.runtime_params import GLUONTS_MAX_IDLE_TRANSFORMS


def serialize_data_entry(data_entry: DataEntry) -> Dict:
    """
    Encode the numpy values in the a DataEntry dictionary into lists so the
    dictionary can be json serialized.
    """

    def serialize_field(field):
        if isinstance(field, np.ndarray):
            # circumvent https://github.com/micropython/micropython/issues/3511
            nan_ix = np.isnan(field)
            field = field.astype(np.object_)
            field[nan_ix] = "NaN"
            return field.tolist()
        return str(field)

    return {
        k: serialize_field(v)
        for k, v in data_entry.data.items()
        if v is not None
    }


class FieldName:
    """
    A bundle of default field names to be used by clients when instantiating
    transformer instances.
    """

    START = 'start'
    TARGET = 'target'

    FEAT_STATIC_CAT = 'feat_static_cat'
    FEAT_STATIC_REAL = 'feat_static_real'
    FEAT_DYNAMIC_CAT = 'feat_dynamic_cat'
    FEAT_DYNAMIC_REAL = 'feat_dynamic_real'

    FEAT_TIME = 'time_feat'
    FEAT_CONST = 'feat_dynamic_const'
    FEAT_AGE = 'feat_dynamic_age'

    OBSERVED_VALUES = 'observed_values'
    IS_PAD = 'is_pad'
    FORECAST_START = 'forecast_start'


def shift_timestamp(ts: pd.Timestamp, offset: int) -> pd.Timestamp:
    """
    Computes a shifted timestamp.

    Basic wrapping around pandas ``ts + offset`` with caching and exception
    handling.
    """
    return _shift_timestamp_helper(ts, ts.freq, offset)


@lru_cache(maxsize=10000)
def _shift_timestamp_helper(
    ts: pd.Timestamp, freq: str, offset: int
) -> pd.Timestamp:
    """
    We are using this helper function which explicitly uses the frequency as a
    parameter, because the frequency is not included in the hash of a time
    stamp.

    I.e.
      pd.Timestamp(x, freq='1D')  and pd.Timestamp(x, freq='1min')

    hash to the same value.
    """
    try:
        # this line looks innocent, but can create a date which is out of
        # bounds values over year 9999 raise a ValueError
        # values over 2262-04-11 raise a pandas OutOfBoundsDatetime
        result = ts + offset * ts.freq
        # For freq M and W pandas seems to lose the freq of the timestamp,
        # so we explicitly set it.
        return pd.Timestamp(result, freq=ts.freq)
    except (ValueError, pd._libs.OutOfBoundsDatetime) as ex:
        raise GluonTSDateBoundsError(ex)


def target_transformation_length(
    target: np.array, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)


class InstanceSampler:
    """
    An InstanceSampler is called with the time series and the valid
    index bounds a, b and should return a set of indices a <= i <= b
    at which training instances will be generated.

    The object should be called with:

    Parameters
    ----------
    ts
        target that should be sampled with shape (dim, seq_len)
    a
        first index of the target that can be sampled
    b
        last index of the target that can be sampled

    Returns
    -------
    np.ndarray
        Selected points to sample
    """

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        raise NotImplementedError()


class UniformSplitSampler(InstanceSampler):
    """
    Samples each point with the same fixed probability.

    Parameters
    ----------
    p
        Probability of selecting a time point
    """

    @validated()
    def __init__(self, p: float = 1.0 / 20.0) -> None:
        self.p = p
        self.lookup = np.arange(2 ** 13)

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        assert a <= b
        while ts.shape[-1] >= len(self.lookup):
            self.lookup = np.arange(2 * len(self.lookup))
        mask = np.random.uniform(low=0.0, high=1.0, size=b - a + 1) < self.p
        return self.lookup[a : a + len(mask)][mask]


class TestSplitSampler(InstanceSampler):
    """
    Sampler used for prediction. Always selects the last time point for
    splitting i.e. the forecast point for the time series.
    """

    @validated()
    def __init__(self) -> None:
        pass

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        return np.array([b])


class ExpectedNumInstanceSampler(InstanceSampler):
    """
    Keeps track of the average time series length and adjusts the probability
    per time point such that on average `num_instances` training examples are
    generated per time series.

    Parameters
    ----------

    num_instances
        number of training examples generated per time series on average
    """

    @validated()
    def __init__(self, num_instances: float) -> None:
        self.num_instances = num_instances
        self.avg_length = 0.0
        self.n = 0.0
        self.lookup = np.arange(2 ** 13)

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        while ts.shape[-1] >= len(self.lookup):
            self.lookup = np.arange(2 * len(self.lookup))

        self.n += 1.0
        self.avg_length += float(b - a + 1 - self.avg_length) / float(self.n)
        p = self.num_instances / self.avg_length

        mask = np.random.uniform(low=0.0, high=1.0, size=b - a + 1) < p
        indices = self.lookup[a : a + len(mask)][mask]
        return indices


class BucketInstanceSampler(InstanceSampler):
    """
    This sample can be used when working with a set of time series that have a
    skewed distributions. For instance, if the dataset contains many time series
    with small values and few with large values.

    The probability of sampling from bucket i is the inverse of its number of elements.

    Parameters
    ----------
    scale_histogram
        The histogram of scale for the time series. Here scale is the mean abs
        value of the time series.
    """

    @validated()
    def __init__(self, scale_histogram: ScaleHistogram) -> None:
        # probability of sampling a bucket i is the inverse of its number of
        # elements
        self.scale_histogram = scale_histogram
        self.lookup = np.arange(2 ** 13)

    def __call__(self, ts: np.ndarray, a: int, b: int) -> None:
        while ts.shape[-1] >= len(self.lookup):
            self.lookup = np.arange(2 * len(self.lookup))
        p = 1.0 / self.scale_histogram.count(ts)
        mask = np.random.uniform(low=0.0, high=1.0, size=b - a + 1) < p
        indices = self.lookup[a : a + len(mask)][mask]
        return indices


class Transformation(metaclass=abc.ABCMeta):
    """
    Base class for all Transformations.

    A Transformation processes works on a stream (iterator) of dictionaries.
    """

    @abc.abstractmethod
    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        pass

    def estimate(self, data_it: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return data_it  # default is to pass through without estimation


class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    @validated()
    def __init__(self, trans: List[Transformation]) -> None:
        self.trans = trans

    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        tmp = data_it
        for t in self.trans:
            tmp = t(tmp, is_train)
        return tmp

    def estimate(self, data_it: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return reduce(lambda x, y: y.estimate(x), self.trans, data_it)


class Identity(Transformation):
    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        return data_it


class MapTransformation(Transformation):
    """
    Base class for Transformations that returns exactly one result per input in the stream.
    """

    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator:
        for data_entry in data_it:
            try:
                yield self.map_transform(data_entry.copy(), is_train)
            except Exception as e:
                raise e

    @abc.abstractmethod
    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        pass


class SimpleTransformation(MapTransformation):
    """
    Element wise transformations that are the same in train and test mode
    """

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return self.transform(data)

    @abc.abstractmethod
    def transform(self, data: DataEntry) -> DataEntry:
        pass


class AdhocTransform(SimpleTransformation):
    """
    Applies a function as a transformation
    This is called ad-hoc, because it is not serializable.
    It is OK to use this for experiments and outside of a model pipeline that
    needs to be serialized.
    """

    def __init__(self, func: Callable[[DataEntry], DataEntry]) -> None:
        self.func = func

    def transform(self, data: DataEntry) -> DataEntry:
        return self.func(data.copy())


class FlatMapTransformation(Transformation):
    """
    Transformations that yield zero or more results per input, but do not combine
    elements from the input stream.
    """

    def __call__(
        self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator:
        num_idle_transforms = 0
        for data_entry in data_it:
            num_idle_transforms += 1
            try:
                for result in self.flatmap_transform(
                    data_entry.copy(), is_train
                ):
                    num_idle_transforms = 0
                    yield result
            except Exception as e:
                raise e
            if num_idle_transforms > GLUONTS_MAX_IDLE_TRANSFORMS:
                raise Exception(
                    f'Reached maximum number of idle transformation calls.\n'
                    f'This means the transformation looped over '
                    f'GLUONTS_MAX_IDLE_TRANSFORMS={GLUONTS_MAX_IDLE_TRANSFORMS} '
                    f'inputs without returning any output.\n'
                    f'This occurred in the following transformation:\n{self}'
                )

    @abc.abstractmethod
    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pass


class FilterTransformation(FlatMapTransformation):
    def __init__(self, condition: Callable[[DataEntry], bool]) -> None:
        self.condition = condition

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        if self.condition(data):
            yield data


class RemoveFields(SimpleTransformation):
    @validated()
    def __init__(self, field_names: List[str]) -> None:
        self.field_names = field_names

    def transform(self, data: DataEntry) -> DataEntry:
        for k in self.field_names:
            if k in data.keys():
                del data[k]
        return data


class SetField(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    @validated()
    def __init__(self, output_field: str, value: Any) -> None:
        self.output_field = output_field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = self.value
        return data


class SetFieldIfNotPresent(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value, in case it does not exist already

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    @validated()
    def __init__(self, field: str, value: Any) -> None:
        self.output_field = field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        if self.output_field not in data.keys():
            data[self.output_field] = self.value
        return data


class AsNumpyArray(SimpleTransformation):
    """
    Converts the value of a field into a numpy array.

    Parameters
    ----------
    expected_ndim
        Expected number of dimensions. Throws an exception if the number of
        dimensions does not match.
    dtype
        numpy dtype to use.
    """

    @validated()
    def __init__(
        self, field: str, expected_ndim: int, dtype: DType = np.float32
    ) -> None:
        self.field = field
        self.expected_ndim = expected_ndim
        self.dtype = dtype

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.field]
        if not isinstance(value, float):
            # this lines produces "ValueError: setting an array element with a
            # sequence" on our test
            # value = np.asarray(value, dtype=np.float32)
            # see https://stackoverflow.com/questions/43863748/
            value = np.asarray(list(value), dtype=self.dtype)
        else:
            # ugly: required as list conversion will fail in the case of a
            # float
            value = np.asarray(value, dtype=self.dtype)
        assert_data_error(
            value.ndim >= self.expected_ndim,
            'Input for field "{self.field}" does not have the required'
            'dimension (field: {self.field}, ndim observed: {value.ndim}, '
            'expected ndim: {self.expected_ndim})',
            value=value,
            self=self,
        )
        data[self.field] = value
        return data


class ExpandDimArray(SimpleTransformation):
    """
    Expand dims in the axis specified, if the axis is not present does nothing.
    (This essentially calls np.expand_dims)

    Parameters
    ----------
    field
        Field in dictionary to use
    axis
        Axis to expand (see np.expand_dims for details)
    """

    @validated()
    def __init__(self, field: str, axis: Optional[int] = None) -> None:
        self.field = field
        self.axis = axis

    def transform(self, data: DataEntry) -> DataEntry:
        if self.axis is not None:
            data[self.field] = np.expand_dims(data[self.field], axis=self.axis)
        return data


class VstackFeatures(SimpleTransformation):
    """
    Stack fields together using ``np.vstack``.

    Fields with value ``None`` are ignored.

    Parameters
    ----------
    output_field
        Field name to use for the output
    input_fields
        Fields to stack together
    drop_inputs
        If set to true the input fields will be dropped.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_fields: List[str],
        drop_inputs: bool = True,
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.cols_to_drop = (
            []
            if not drop_inputs
            else [
                fname for fname in self.input_fields if fname != output_field
            ]
        )

    def transform(self, data: DataEntry) -> DataEntry:
        r = [
            data[fname]
            for fname in self.input_fields
            if data[fname] is not None
        ]
        output = np.vstack(r)
        data[self.output_field] = output
        for fname in self.cols_to_drop:
            del data[fname]
        return data


class ConcatFeatures(SimpleTransformation):
    """
    Concatenate fields together using ``np.concatenate``.

    Fields with value ``None`` are ignored.

    Parameters
    ----------
    output_field
        Field name to use for the output
    input_fields
        Fields to stack together
    drop_inputs
        If set to true the input fields will be dropped.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_fields: List[str],
        drop_inputs: bool = True,
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.cols_to_drop = (
            []
            if not drop_inputs
            else [
                fname for fname in self.input_fields if fname != output_field
            ]
        )

    def transform(self, data: DataEntry) -> DataEntry:
        r = [
            data[fname]
            for fname in self.input_fields
            if data[fname] is not None
        ]
        output = np.concatenate(r)
        data[self.output_field] = output
        for fname in self.cols_to_drop:
            del data[fname]
        return data


class SwapAxes(SimpleTransformation):
    """
    Apply `np.swapaxes` to fields.

    Parameters
    ----------
    input_fields
        Field to apply to
    axes
        Axes to use
    """

    @validated()
    def __init__(self, input_fields: List[str], axes: Tuple[int, int]) -> None:
        self.input_fields = input_fields
        self.axis1, self.axis2 = axes

    def transform(self, data: DataEntry) -> DataEntry:
        for field in self.input_fields:
            data[field] = self.swap(data[field])
        return data

    def swap(self, v):
        if isinstance(v, np.ndarray):
            return np.swapaxes(v, self.axis1, self.axis2)
        if isinstance(v, list):
            return [self.swap(x) for x in v]
        else:
            raise ValueError(
                f'Unexpected field type {type(v).__name__}, expected '
                f'np.ndarray or list[np.ndarray]'
            )


class ListFeatures(SimpleTransformation):
    """
    Creates a new field which contains a list of features.

    Parameters
    ----------
    output_field
        Field name for output
    input_fields
        Fields to combine into list
    drop_inputs
        If true the input fields will be removed from the result.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_fields: List[str],
        drop_inputs: bool = True,
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.cols_to_drop = (
            []
            if not drop_inputs
            else [
                fname for fname in self.input_fields if fname != output_field
            ]
        )

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = [data[fname] for fname in self.input_fields]
        for fname in self.cols_to_drop:
            del data[fname]
        return data


class AddObservedValuesIndicator(SimpleTransformation):
    """
    Replaces missing values in a numpy array (NaNs) with a dummy value and adds an "observed"-indicator
    that is
      1 - when values are observed
      0 - when values are missing


    Parameters
    ----------
    target_field
        Field for which missing values will be replaced
    output_field
        Field name to use for the indicator
    dummy_value
        Value to use for replacing missing values.
    convert_nans
        If set to true (default) missing values will be replaced. Otherwise
        they will not be replaced. In any case the indicator is included in the
        result.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dummy_value: int = 0,
        convert_nans: bool = True,
    ) -> None:
        self.dummy_value = dummy_value
        self.target_field = target_field
        self.output_field = output_field
        self.convert_nans = convert_nans

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.target_field]
        nan_indices = np.where(np.isnan(value))
        nan_entries = np.isnan(value)

        if self.convert_nans:
            value[nan_indices] = self.dummy_value

        data[self.target_field] = value
        # Invert bool array so that missing values are zeros and store as float
        data[self.output_field] = np.invert(nan_entries).astype(np.float32)
        return data


class RenameFields(SimpleTransformation):
    """
    Rename fields using a mapping

    Parameters
    ----------
    mapping
        Name mapping `input_name -> output_name`
    """

    @validated()
    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping
        values_count = Counter(mapping.values())
        for new_key, count in values_count.items():
            assert count == 1, f'Mapped key {new_key} occurs multiple time'

    def transform(self, data: DataEntry):
        for key, new_key in self.mapping.items():
            if key not in data:
                continue
            assert new_key not in data
            data[new_key] = data[key]
            del data[key]
        return data


class AddConstFeature(MapTransformation):
    """
    Expands a `const` value along the time axis as a dynamic feature, where
    the T-dimension is defined as the sum of the `pred_length` parameter and
    the length of a time series specified by the `target_field`.

    If `is_train=True` the feature matrix has the same length as the `target` field.
    If `is_train=False` the feature matrix has length len(target) + pred_length

    Parameters
    ----------
    output_field
        Field name for output.
    target_field
        Field containing the target array. The length of this array will be used.
    pred_length
        Prediction length (this is necessary since
        features have to be available in the future)
    const
        Constant value to use.
    dtype
        Numpy dtype to use for resulting array.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        target_field: str,
        pred_length: int,
        const: float = 1.0,
        dtype: DType = np.float32,
    ) -> None:
        self.pred_length = pred_length
        self.const = const
        self.dtype = dtype
        self.output_field = output_field
        self.target_field = target_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        data[self.output_field] = self.const * np.ones(
            shape=(1, length), dtype=self.dtype
        )
        return data


class AddTimeFeatures(MapTransformation):
    """
    Adds a set of time features.

    If `is_train=True` the feature matrix has the same length as the `target` field.
    If `is_train=False` the feature matrix has length len(target) + pred_length

    Parameters
    ----------
    start_field
        Field with the start time stamp of the time series
    target_field
        Field with the array containing the time series values
    output_field
        Field name for result.
    time_features
        list of time features to use.
    pred_length
        Prediction length
    """

    @validated()
    def __init__(
        self,
        start_field: str,
        target_field: str,
        output_field: str,
        time_features: List[TimeFeature],
        pred_length: int,
    ) -> None:
        self.date_features = time_features
        self.pred_length = pred_length
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field
        self._min_time_point: pd.Timestamp = None
        self._max_time_point: pd.Timestamp = None
        self._full_range_date_features: np.ndarray = None
        self._date_index: pd.DatetimeIndex = None

    def _update_cache(self, start: pd.Timestamp, length: int) -> None:
        end = shift_timestamp(start, length)
        if self._min_time_point is not None:
            if self._min_time_point <= start and end <= self._max_time_point:
                return
        if self._min_time_point is None:
            self._min_time_point = start
            self._max_time_point = end
        self._min_time_point = min(
            shift_timestamp(start, -50), self._min_time_point
        )
        self._max_time_point = max(
            shift_timestamp(end, 50), self._max_time_point
        )
        self.full_date_range = pd.date_range(
            self._min_time_point, self._max_time_point, freq=start.freq
        )
        self._full_range_date_features = np.vstack(
            [feat(self.full_date_range) for feat in self.date_features]
        )
        self._date_index = pd.Series(
            index=self.full_date_range,
            data=np.arange(len(self.full_date_range)),
        )

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        start = data[self.start_field]
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        self._update_cache(start, length)
        i0 = self._date_index[start]
        features = self._full_range_date_features[..., i0 : i0 + length]
        data[self.output_field] = features
        return data


class AddAgeFeature(MapTransformation):
    """
    Adds an 'age' feature to the data_entry.

    The age feature starts with a small value at the start of the time series
    and grows over time.

    If `is_train=True` the age feature has the same length as the `target` field.
    If `is_train=False` the age feature has length len(target) + pred_length

    Parameters
    ----------
    target_field
        Field with target values (array) of time series
    output_field
        Field name to use for the output.
    pred_length
        Prediction length
    log_scale
        If set to true the age feature grows logarithmically otherwise linearly over time.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        pred_length: int,
        log_scale: bool = True,
    ) -> None:
        self.pred_length = pred_length
        self.target_field = target_field
        self.feature_name = output_field
        self.log_scale = log_scale
        self._age_feature = np.zeros(0)

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )

        if self.log_scale:
            age = np.log10(2.0 + np.arange(length, dtype=np.float32))
        else:
            age = np.arange(length, dtype=np.float32)

        data[self.feature_name] = age.reshape((1, length))

        return data


class InstanceSplitter(FlatMapTransformation):
    """
    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.

    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).

    target -> past_target and future_target

    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.

    Convention: time axis is always the last axis.

    Parameters
    ----------

    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    train_sampler
        instance sampler that provides sampling indices given a time-series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target
    pick_incomplete
        whether training examples can be sampled with only a part of
        past_length time-units
        present for the time series. This is useful to train models for
        cold-start. In such case, is_pad_out contains an indicator whether
        data is padded or not.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        train_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        output_NTC: bool = True,
        time_series_fields: Optional[List[str]] = None,
        pick_incomplete: bool = True,
    ) -> None:

        assert future_length > 0

        self.train_sampler = train_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.output_NTC = output_NTC
        self.ts_fields = (
            time_series_fields if time_series_fields is not None else []
        )
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.pick_incomplete = pick_incomplete

    def _past(self, col_name):
        return f'past_{col_name}'

    def _future(self, col_name):
        return f'future_{col_name}'

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pl = self.future_length
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]

        len_target = target.shape[-1]

        if is_train:
            if len_target < self.future_length:
                # We currently cannot handle time series that are shorter than
                # the prediction length during training, so we just skip these.
                # If we want to include them we would need to pad and to mask
                # the loss.
                sampling_indices: List[int] = []
            else:
                if self.pick_incomplete:
                    sampling_indices = self.train_sampler(
                        target, 0, len_target - self.future_length
                    )
                else:
                    sampling_indices = self.train_sampler(
                        target,
                        self.past_length,
                        len_target - self.future_length,
                    )
        else:
            sampling_indices = [len_target]
        for i in sampling_indices:
            pad_length = max(self.past_length - i, 0)
            if not self.pick_incomplete:
                assert pad_length == 0
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                elif i < self.past_length:
                    pad_block = np.zeros(
                        d[ts_field].shape[:-1] + (pad_length,),
                        dtype=d[ts_field].dtype,
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][..., i : i + pl]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    d[self._past(ts_field)] = d[
                        self._past(ts_field)
                    ].transpose()
                    d[self._future(ts_field)] = d[
                        self._future(ts_field)
                    ].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], i
            )
            yield d


class CanonicalInstanceSplitter(FlatMapTransformation):
    """
    Selects instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    In training mode, the returned instances contain past_`target_field`
    as well as past_`time_series_fields`.

    In prediction mode, one can set `use_prediction_features` to get
    future_`time_series_fields`.

    If the target array is one-dimensional, the `target_field` in the resulting instance has shape
    (`instance_length`). In the multi-dimensional case, the instance has shape (`dim`, `instance_length`),
    where `dim` can also take a value of 1.

    In the case of insufficient number of time series values, the
    transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not, and the value is padded with
    `default_pad_value` with a default value 0.
    This is done only if `allow_target_padding` is `True`,
    and the length of `target` is smaller than `instance_length`.

    Parameters
    ----------
    target_field
        fields that contains time-series
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        field containing the forecast start date
    instance_sampler
        instance sampler that provides sampling indices given a time-series
    instance_length
        length of the target seen before making prediction
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target
    allow_target_padding
        flag to allow padding
    pad_value
        value to be used for padding
    use_prediction_features
        flag to indicate if prediction range features should be returned
    prediction_length
        length of the prediction range, must be set if
        use_prediction_features is True
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        instance_length: int,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        allow_target_padding: bool = False,
        pad_value: float = 0.0,
        use_prediction_features: bool = False,
        prediction_length: Optional[int] = None,
    ) -> None:
        self.instance_sampler = instance_sampler
        self.instance_length = instance_length
        self.output_NTC = output_NTC
        self.dynamic_feature_fields = time_series_fields
        self.target_field = target_field
        self.allow_target_padding = allow_target_padding
        self.pad_value = pad_value
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field

        assert (
            not use_prediction_features or prediction_length is not None
        ), "You must specify `prediction_length` if `use_prediction_features`"

        self.use_prediction_features = use_prediction_features
        self.prediction_length = prediction_length

    def _past(self, col_name):
        return f'past_{col_name}'

    def _future(self, col_name):
        return f'future_{col_name}'

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        ts_fields = self.dynamic_feature_fields + [self.target_field]
        ts_target = data[self.target_field]

        len_target = ts_target.shape[-1]

        if is_train:
            if len_target < self.instance_length:
                sampling_indices = (
                    # Returning [] for all time series will cause this to be in loop forever!
                    [len_target]
                    if self.allow_target_padding
                    else []
                )
            else:
                sampling_indices = self.instance_sampler(
                    ts_target, self.instance_length, len_target
                )
        else:
            sampling_indices = [len_target]

        for i in sampling_indices:
            d = data.copy()

            pad_length = max(self.instance_length - i, 0)

            # update start field
            d[self.start_field] = shift_timestamp(
                data[self.start_field], i - self.instance_length
            )

            # set is_pad field
            is_pad = np.zeros(self.instance_length)
            if pad_length > 0:
                is_pad[:pad_length] = 1
            d[self.is_pad_field] = is_pad

            # update time series fields
            for ts_field in ts_fields:
                full_ts = data[ts_field]
                if pad_length > 0:
                    pad_pre = self.pad_value * np.ones(
                        shape=full_ts.shape[:-1] + (pad_length,)
                    )
                    past_ts = np.concatenate(
                        [pad_pre, full_ts[..., :i]], axis=-1
                    )
                else:
                    past_ts = full_ts[..., (i - self.instance_length) : i]

                past_ts = past_ts.transpose() if self.output_NTC else past_ts
                d[self._past(ts_field)] = past_ts

                if self.use_prediction_features and not is_train:
                    if not ts_field == self.target_field:
                        future_ts = full_ts[
                            ..., i : i + self.prediction_length
                        ]
                        future_ts = (
                            future_ts.transpose()
                            if self.output_NTC
                            else future_ts
                        )
                        d[self._future(ts_field)] = future_ts

                del d[ts_field]

            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], self.instance_length
            )

            yield d


class SelectFields(MapTransformation):
    """
    Only keep the listed fields

    Parameters
    ----------
    input_fields
        List of fields to keep.
    """

    @validated()
    def __init__(self, input_fields: List[str]) -> None:
        self.input_fields = input_fields

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return {f: data[f] for f in self.input_fields}


class TransformedDataset(Dataset):
    """
    A dataset that corresponds to applying a list of transformations to each
    element in the base_dataset.
    This only supports SimpleTransformations, which do the same thing at
    prediction and training time.


    Parameters
    ----------
    base_dataset
        Dataset to transform
    transformations
        List of transformations to apply
    """

    def __init__(
        self, base_dataset: Dataset, transformations: List[Transformation]
    ) -> None:
        self.base_dataset = base_dataset
        self.transformations = Chain(transformations)

    def __iter__(self) -> Iterator[DataEntry]:
        yield from self.transformations(self.base_dataset, is_train=True)

    def __len__(self):
        return sum(1 for _ in self)
