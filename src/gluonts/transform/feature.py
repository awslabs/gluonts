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

from gluonts.core.component import DType, validated
from gluonts.dataset.common import DataEntry
from gluonts.time_feature import TimeFeature

from ._base import MapTransformation, SimpleTransformation
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
        """

        Parameters
        ----------
        values : the array of values with or without nans

        Returns
        -------
        values : the array of values with the nans replaced according to the method used.

        """
        raise NotImplementedError()


class LeavesMissingValues(MissingValueImputation):
    """
    Just leaves the missing values untouched.
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values


class DummyValueImputation(MissingValueImputation):
    """
    This class replaces all the missing values with the same dummy value given in advance.
    """

    @validated()
    def __init__(self, dummy_value: float = 0.0) -> None:
        self.dummy_value = dummy_value

    def __call__(self, values: np.ndarray) -> np.ndarray:
        nan_indices = np.where(np.isnan(values))
        values[nan_indices] = self.dummy_value
        return values


class MeanValueImputation(MissingValueImputation):
    """
    This class replaces all the missing values with the mean of the non missing values.
    Careful this is not a 'causal' method in the sense that it leaks information about the furture in the imputation.
    You may prefer to use CausalMeanValueImputation instead.
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
        nan_indices = np.where(np.isnan(values))
        values[nan_indices] = np.nanmean(values)
        return values


class LastValueImputation(MissingValueImputation):
    """
    This class replaces each missing value with the last value that was not missing.
    (If the first values are missing, they are replaced by the closest non missing value.)
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
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
    """
    This class replaces each missing value with the average of all the values up to this point.
    (If the first values are missing, they are replaced by the closest non missing value.)
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
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
    """
    This class replaces each missing value with the average of all the last window_size (default=10) values.
    (If the first values are missing, they are replaced by the closest non missing value.)
    """

    @validated()
    def __init__(self, window_size: int = 10) -> None:
        self.window_size = 1 if window_size < 1 else window_size

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
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
    imputation_method
        One of the methods from ImputationStrategy. If set to None, no imputation is
        done and only the indicator is included.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        imputation_method: Optional[
            MissingValueImputation
        ] = DummyValueImputation(0.0),
        dtype: DType = np.float32,
    ) -> None:
        self.target_field = target_field
        self.output_field = output_field
        self.dtype = dtype
        self.imputation_method = imputation_method

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.target_field]
        nan_entries = np.isnan(value)

        if self.imputation_method is not None:
            if nan_entries.any():
                value = value.copy()
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
        dtype: DType = np.float32,
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
        self.dtype = dtype

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
            ).astype(self.dtype)
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


class AddAggregateLags(MapTransformation):
    """
    Adds aggregate lags as a feature to the data_entry.

    Aggregates the original time series to a new frequency and selects
    the aggregated lags of interest. It does not use aggregate lags that
    need the last `prediction_length` values to be computed. Therefore
    the transformation is applicable to both training and inference.

    If `is_train=True` the lags have the same length as the `target` field.
    If `is_train=False` the lags have length len(target) + pred_length

    Parameters
    ----------
    target_field
        Field with target values (array) of time series
    output_field
        Field name to use for the output.
    pred_length
        Prediction length.
    base_freq
        Base frequency, i.e., the frequency of the original time series.
    agg_freq
        Aggregate frequency, i.e., the frequency of the aggregate time series.
    agg_lags
        List of aggregate lags given in the aggregate frequncy. If some of them
        are invalid (need some of the last `prediction_length` values to be computed)
        they are ignored.
    agg_fun
        Aggregation function. Default is 'mean'.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        pred_length: int,
        base_freq: str,
        agg_freq: str,
        agg_lags: List[int],
        agg_fun: str = "mean",
        dtype: DType = np.float32,
    ) -> None:
        self.pred_length = pred_length
        self.target_field = target_field
        self.feature_name = output_field
        self.base_freq = base_freq
        self.agg_freq = agg_freq
        self.agg_lags = agg_lags
        self.agg_fun = agg_fun
        self.dtype = dtype

        self.ratio = pd.Timedelta(self.agg_freq) / pd.Timedelta(self.base_freq)
        assert (
            self.ratio.is_integer() and self.ratio >= 1
        ), "The aggregate frequency should be a multiple of the base frequency."
        self.ratio = int(self.ratio)

        self.half_window = (self.ratio - 1) // 2
        self.valid_lags = [
            x
            for x in self.agg_lags
            if x > (self.pred_length - 1 + self.half_window) / self.ratio
        ]

        if set(self.agg_lags) - set(self.valid_lags):
            print(
                f"The aggregate lags {set(self.agg_lags) - set(self.valid_lags)} "
                f"of frequency {self.agg_freq} are ignored."
            )

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        assert self.base_freq == data["start"].freq

        # convert to pandas Series for easier indexing and aggregation
        if is_train:
            t = data[self.target_field]
        else:
            t = np.concatenate(
                [data[self.target_field], np.zeros(shape=(self.pred_length,))],
                axis=0,
            )

        # convert to pandas Series for easier rolling window aggregation
        t_agg = (pd.Series(t).rolling(self.ratio).agg(self.agg_fun))[
            self.ratio - 1 :
        ]

        # compute the aggregate lags for each time point of the time series
        agg_vals = np.concatenate(
            [
                np.zeros(
                    (max(self.valid_lags) * self.ratio + self.half_window + 1,)
                ),
                t_agg.values,
            ],
            axis=0,
        )
        lags = np.vstack(
            [
                agg_vals[
                    -(l * self.ratio - self.half_window + len(t)) : -(
                        l * self.ratio - self.half_window
                    )
                    if -(l * self.ratio - self.half_window) is not 0
                    else None
                ]
                for l in self.valid_lags
            ]
        )

        # update the data entry
        data[self.feature_name] = np.nan_to_num(lags)

        assert data[self.feature_name].shape == (
            len(self.valid_lags),
            len(data[self.target_field]) + self.pred_length * (not is_train),
        )

        return data
