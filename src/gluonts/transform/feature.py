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

from typing import List, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated, DType
from gluonts.dataset.common import DataEntry
from gluonts.time_feature import TimeFeature

from ._base import SimpleTransformation, MapTransformation
from .split import shift_timestamp


def target_transformation_length(
    target: np.array, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)


class MissingValueImputation:
    """
    The parent class for all the missing value imputation classes.
    You can just implement your own inheriting this class.
    """

    @validated()
    def __init__(self) -> None:
        pass

    def __call__(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class LeavesMissingValues(MissingValueImputation):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values


class DummyValueImputation(MissingValueImputation):
    @validated()
    def __init__(self, dummy_value: float = 0.0) -> None:
        self.dummy_value = dummy_value

    def __call__(self, values: np.ndarray) -> np.ndarray:
        nan_indices = np.where(np.isnan(values))
        values[nan_indices] = self.dummy_value
        return values


class MeanValueImputation(MissingValueImputation):
    """
    Careful this is not a 'causal' method in the sense that it leaks information about the furture in the imputation.
    You may prefer to use CausalMeanValueImputation instead.
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        nan_indices = np.where(np.isnan(values))
        values[nan_indices] = np.nanmean(values)
        return values


class LastValueImputation(MissingValueImputation):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.expand_dims(values, axis=0)

        mask = np.isnan(values)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = values[np.arange(idx.shape[0])[:, None], idx]

        values = np.squeeze(out)

        # in case we need to replace nan at the start of the array
        mask = np.isnan(values)
        values[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), values[~mask]
        )

        return values


class CausalMeanValueImputation(MissingValueImputation):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        mask = np.isnan(values)

        # we cannot compute the mean with this method if there are nans
        # so we do a temporary fix of the nan just for the mean computation using this:
        last_value_imputation = LastValueImputation()
        value_no_nans = last_value_imputation(values)

        # We do the cumulative sum shifted by one indices:
        adjusted_values_to_causality = np.concatenate(
            (np.repeat(0.0, 1), value_no_nans[:-1])
        )
        cumsum = np.cumsum(adjusted_values_to_causality)

        # We get the indices of the elements shifted by one indices:
        indices = np.linspace(0, len(value_no_nans) - 1, len(value_no_nans))

        ar_res = cumsum / indices.astype(float)
        values[mask] = ar_res[mask]

        # make sure that we do not leave the potential nan in the first position:
        values[0] = value_no_nans[0]

        return values


class RollingMeanValueImputation(MissingValueImputation):
    @validated()
    def __init__(self, window_size: int = 10) -> None:
        self.window_size = 1 if window_size < 1 else window_size

    def __call__(self, values: np.ndarray) -> np.ndarray:
        mask = np.isnan(values)

        # we cannot compute the mean with this method if there are nans
        # so we do a temporary fix of the nan just for the mean computation using this:
        last_value_imputation = LastValueImputation()
        value_no_nans = last_value_imputation(values)

        adjusted_values_to_causality = np.concatenate(
            (
                np.repeat(value_no_nans[0], self.window_size + 1),
                value_no_nans[:-1],
            )
        )

        cumsum = np.cumsum(adjusted_values_to_causality)

        ar_res = (
            cumsum[self.window_size :] - cumsum[: -self.window_size]
        ) / float(self.window_size)

        values[mask] = ar_res[mask]

        # make sure that we do not leave the potential nan in the first position:
        values[0] = value_no_nans[0]

        return values


class AddObservedValuesIndicator(SimpleTransformation):
    """
    Replaces missing values in a numpy array (NaNs) with a dummy value and adds
    an "observed"-indicator that is ``1`` when values are observed and ``0``
    when values are missing.


    Parameters
    ----------
    target_field
        Field for which missing values will be replaced
    output_field
        Field name to use for the indicator
    dummy_value
        Value to use for replacing missing values.
    imputation_method
        One of the methods from ImputationStrategy.
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
        dummy_value: float = 0.0,
        imputation_method: Optional[MissingValueImputation] = None,
        convert_nans: bool = True,
        dtype: DType = np.float32,
    ) -> None:
        self.dummy_value = dummy_value
        self.target_field = target_field
        self.output_field = output_field
        self.convert_nans = convert_nans
        self.dtype = dtype

        self.imputation_method = (
            imputation_method
            if imputation_method is not None
            else DummyValueImputation(dummy_value)
        )

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.target_field]
        nan_entries = np.isnan(value)

        if self.convert_nans:
            data[self.target_field] = self.imputation_method(value)

        data[self.output_field] = np.invert(
            nan_entries, out=nan_entries
        ).astype(self.dtype, copy=False)
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
        self._full_range_date_features = (
            np.vstack(
                [feat(self.full_date_range) for feat in self.date_features]
            )
            if self.date_features
            else None
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
        features = (
            self._full_range_date_features[..., i0 : i0 + length]
            if self.date_features
            else None
        )
        data[self.output_field] = features
        return data


class AddAgeFeature(MapTransformation):
    """
    Adds an 'age' feature to the data_entry.

    The age feature starts with a small value at the start of the time series
    and grows over time.

    If `is_train=True` the age feature has the same length as the `target`
    field.
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
        If set to true the age feature grows logarithmically otherwise linearly
        over time.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        pred_length: int,
        log_scale: bool = True,
        dtype: DType = np.float32,
    ) -> None:
        self.pred_length = pred_length
        self.target_field = target_field
        self.feature_name = output_field
        self.log_scale = log_scale
        self._age_feature = np.zeros(0)
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )

        if self.log_scale:
            age = np.log10(2.0 + np.arange(length, dtype=self.dtype))
        else:
            age = np.arange(length, dtype=self.dtype)

        data[self.feature_name] = age.reshape((1, length))

        return data
