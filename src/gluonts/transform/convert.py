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

from typing import Iterator, List, Optional, Tuple, Type

import numpy as np

from gluonts.core.component import validated, tensor_to_numpy
from gluonts.dataset.common import DataEntry
from gluonts.exceptions import assert_data_error

from ._base import (
    FlatMapTransformation,
    MapTransformation,
    SimpleTransformation,
)


def erf(x: np.ndarray) -> np.ndarray:
    # Using numerical recipes approximation for erf function
    # accurate to 1E-7

    ones = np.ones_like(x)
    zeros = np.zeros_like(x)

    t = ones / (ones + 0.5 * np.abs(x))

    coefficients = [
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]

    inner = zeros
    for c in coefficients[::-1]:
        inner = t * (c + inner)

    res = ones - t * np.exp(inner - 1.26551223 - np.square(x))
    return np.where(x >= zeros, res, -1.0 * res)


def erfinv(x: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(x)

    w = -np.log((1.0 - x) * (1.0 + x))
    mask_lesser = w < (zeros + 5.0)

    w = np.where(mask_lesser, w - 2.5, np.sqrt(w) - 3.0)

    coefficients_lesser = [
        2.81022636e-08,
        3.43273939e-07,
        -3.5233877e-06,
        -4.39150654e-06,
        0.00021858087,
        -0.00125372503,
        -0.00417768164,
        0.246640727,
        1.50140941,
    ]

    coefficients_greater_equal = [
        -0.000200214257,
        0.000100950558,
        0.00134934322,
        -0.00367342844,
        0.00573950773,
        -0.0076224613,
        0.00943887047,
        1.00167406,
        2.83297682,
    ]

    p = np.where(
        mask_lesser,
        coefficients_lesser[0] + zeros,
        coefficients_greater_equal[0] + zeros,
    )

    for c_l, c_ge in zip(
        coefficients_lesser[1:], coefficients_greater_equal[1:]
    ):
        c = np.where(mask_lesser, c_l + zeros, c_ge + zeros)
        p = c + p * w

    return p * x


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
        self, field: str, expected_ndim: int, dtype: Type = np.float32
    ) -> None:
        self.field = field
        self.expected_ndim = expected_ndim
        self.dtype = dtype

    def transform(self, data: DataEntry) -> DataEntry:
        value = np.asarray(data[self.field], dtype=self.dtype)

        assert_data_error(
            value.ndim == self.expected_ndim,
            'Input for field "{self.field}" does not have the required'
            "dimension (field: {self.field}, ndim observed: {value.ndim}, "
            "expected ndim: {self.expected_ndim})",
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
    Stack fields together using ``np.vstack`` when h_stack = False.
    Otherwise stack fields together using ``np.hstack``.

    Fields with value ``None`` are ignored.

    Parameters
    ----------
    output_field
        Field name to use for the output
    input_fields
        Fields to stack together
    drop_inputs
        If set to true the input fields will be dropped.
    h_stack
        To stack horizontally instead of vertically
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_fields: List[str],
        drop_inputs: bool = True,
        h_stack: bool = False,
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
        self.h_stack = h_stack

    def transform(self, data: DataEntry) -> DataEntry:
        r = [
            data[fname]
            for fname in self.input_fields
            if data[fname] is not None
        ]
        output = np.vstack(r) if not self.h_stack else np.hstack(r)
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
                f"Unexpected field type {type(v).__name__}, expected "
                "np.ndarray or list[np.ndarray]"
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


class TargetDimIndicator(SimpleTransformation):
    """
    Label-encoding of the target dimensions.
    """

    @validated()
    def __init__(self, field_name: str, target_field: str) -> None:
        self.field_name = field_name
        self.target_field = target_field

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.field_name] = np.arange(0, data[self.target_field].shape[0])
        return data


class SampleTargetDim(FlatMapTransformation):
    """
    Samples random dimensions from the target at training time.
    """

    @validated()
    def __init__(
        self,
        field_name: str,
        target_field: str,
        observed_values_field: str,
        num_samples: int,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        self.field_name = field_name
        self.target_field = target_field
        self.observed_values_field = observed_values_field
        self.num_samples = num_samples
        self.shuffle = shuffle

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        if not is_train:
            yield data
        else:
            # (target_dim,)
            target_dimensions = data[self.field_name]

            if self.shuffle:
                np.random.shuffle(target_dimensions)

            target_dimensions = target_dimensions[: self.num_samples]

            data[self.field_name] = target_dimensions
            # (seq_len, target_dim) -> (seq_len, num_samples)

            for field in [
                f"past_{self.target_field}",
                f"future_{self.target_field}",
                f"past_{self.observed_values_field}",
                f"future_{self.observed_values_field}",
            ]:
                data[field] = data[field][:, target_dimensions]

            yield data


class CDFtoGaussianTransform(MapTransformation):
    """
    Marginal transformation that transforms the target via an empirical CDF to
    a standard gaussian as described here: https://arxiv.org/abs/1910.03002.

    To be used in conjunction with a multivariate gaussian to from a copula.
    Note that this transformation is currently intended for multivariate
    targets only.
    """

    @validated()
    def __init__(
        self,
        target_dim: int,
        target_field: str,
        observed_values_field: str,
        cdf_suffix="_cdf",
        max_context_length: Optional[int] = None,
        dtype: Type = np.float32,
    ) -> None:
        """
        Constructor for CDFtoGaussianTransform.

        Parameters
        ----------
        target_dim
            Dimensionality of the target.
        target_field
            Field that will be transformed.
        observed_values_field
            Field that indicates observed values.
        cdf_suffix
            Suffix to mark the field with the transformed target.
        max_context_length
            Sets the maximum context length for the empirical CDF.
        dtype
            numpy dtype of output.
        """
        self.target_field = target_field
        self.past_target_field = "past_" + self.target_field
        self.future_target_field = "future_" + self.target_field
        self.past_observed_field = f"past_{observed_values_field}"
        self.sort_target_field = f"past_{target_field}_sorted"
        self.slopes_field = "slopes"
        self.intercepts_field = "intercepts"
        self.cdf_suffix = cdf_suffix
        self.max_context_length = max_context_length
        self.target_dim = target_dim
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        self._preprocess_data(data, is_train=is_train)
        self._calc_pw_linear_params(data)

        for target_field in [self.past_target_field, self.future_target_field]:
            data[target_field + self.cdf_suffix] = self.standard_gaussian_ppf(
                self._empirical_cdf_forward_transform(
                    data[self.sort_target_field],
                    data[target_field],
                    data[self.slopes_field],
                    data[self.intercepts_field],
                )
            )
        return data

    def _preprocess_data(self, data: DataEntry, is_train: bool):
        """
        Performs several preprocess operations for computing the empirical CDF.
        1) Reshaping the data. 2) Normalizing the target length. 3) Adding
        noise to avoid zero slopes (training only) 4) Sorting the target to
        compute the empirical CDF.

        Parameters
        ----------
        data
            DataEntry with input data.
        is_train
            if is_train is True, this function adds noise to the target to
            avoid zero slopes in the piece-wise linear function.
        Returns
        -------
        """
        # (target_length, target_dim)
        past_target_vec = data[self.past_target_field].copy()

        # pick only observed values
        target_length, target_dim = past_target_vec.shape

        # (target_length, target_dim)
        past_observed = (data[self.past_observed_field] > 0) * (
            data["past_is_pad"].reshape((-1, 1)) == 0
        )
        assert past_observed.ndim == 2
        assert target_dim == self.target_dim

        past_target_vec = past_target_vec[past_observed.min(axis=1)]

        assert past_target_vec.ndim == 2
        assert past_target_vec.shape[1] == self.target_dim

        expected_length = (
            target_length
            if self.max_context_length is None
            else self.max_context_length
        )

        if target_length != expected_length:
            # Fills values in the case where past_target_vec.shape[-1] <
            # target_length
            # as dataset.loader.BatchBuffer does not support varying shapes
            past_target_vec = CDFtoGaussianTransform._fill(
                past_target_vec, expected_length
            )

        # sorts along the time dimension to compute empirical CDF of each
        # dimension
        if is_train:
            past_target_vec = self._add_noise(past_target_vec).astype(
                self.dtype
            )

        past_target_vec.sort(axis=0)

        assert past_target_vec.shape == (expected_length, self.target_dim)

        data[self.sort_target_field] = past_target_vec

    def _calc_pw_linear_params(self, data: DataEntry):
        """
        Calculates the piece-wise linear parameters to interpolate between the
        observed values in the empirical CDF.

        Once current limitation is that we use a zero slope line as the last
        piece. Thus, we cannot forecast anything higher than the highest
        observed value.

        Parameters
        ----------
        data
            Input data entry containing a sorted target field.

        Returns
        -------
        """
        sorted_target = data[self.sort_target_field]
        sorted_target_length, target_dim = sorted_target.shape

        quantiles = np.stack(
            [np.arange(sorted_target_length) for _ in range(target_dim)],
            axis=1,
        ) / float(sorted_target_length)

        x_diff = np.diff(sorted_target, axis=0)
        y_diff = np.diff(quantiles, axis=0)

        # Calculate slopes of the pw-linear pieces.
        slopes = np.where(
            x_diff == 0.0, np.zeros_like(x_diff), y_diff / x_diff
        )

        zeroes = np.zeros_like(np.expand_dims(slopes[0, :], axis=0))
        slopes = np.append(slopes, zeroes, axis=0)

        # Calculate intercepts of the pw-linear pieces.
        intercepts = quantiles - slopes * sorted_target

        # Populate new fields with the piece-wise linear parameters.
        data[self.slopes_field] = slopes.astype(self.dtype)
        data[self.intercepts_field] = intercepts.astype(self.dtype)

    def _empirical_cdf_forward_transform(
        self,
        sorted_values: np.ndarray,
        values: np.ndarray,
        slopes: np.ndarray,
        intercepts: np.ndarray,
    ) -> np.ndarray:
        """
        Applies the empirical CDF forward transformation.

        Parameters
        ----------
        sorted_values
            Sorted target vector.
        values
            Values (real valued) that will be transformed to empirical CDF
            values.
        slopes
            Slopes of the piece-wise linear function.
        intercepts
            Intercepts of the piece-wise linear function.

        Returns
        -------
        quantiles
            Empirical CDF quantiles in [0, 1] interval with winzorized cutoff.
        """
        m = sorted_values.shape[0]
        quantiles = self._forward_transform(
            sorted_values, values, slopes, intercepts
        )

        quantiles = np.clip(
            quantiles, self.winsorized_cutoff(m), 1 - self.winsorized_cutoff(m)
        )
        return quantiles

    @staticmethod
    def _add_noise(x: np.ndarray) -> np.ndarray:
        scale_noise = 0.2
        std = np.sqrt(
            (np.square(x - x.mean(axis=1, keepdims=True))).mean(
                axis=1, keepdims=True
            )
        )
        noise = np.random.normal(
            loc=np.zeros_like(x), scale=np.ones_like(x) * std * scale_noise
        )
        x = x + noise
        return x

    @staticmethod
    def _search_sorted(
        sorted_vec: np.ndarray, to_insert_vec: np.ndarray
    ) -> np.ndarray:
        """
        Finds the indices of the active piece-wise linear function.

        Parameters
        ----------
        sorted_vec
            Sorted target vector.
        to_insert_vec
            Vector for which the indicies of the active linear functions
            will be computed

        Returns
        -------
        indices
            Indices mapping to the active linear function.
        """
        indices_left = np.searchsorted(sorted_vec, to_insert_vec, side="left")
        indices_right = np.searchsorted(
            sorted_vec, to_insert_vec, side="right"
        )

        indices = indices_left + (indices_right - indices_left) // 2
        indices = indices - 1
        indices = np.minimum(indices, len(sorted_vec) - 1)
        indices[indices < 0] = 0
        return indices

    def _forward_transform(
        self,
        sorted_vec: np.ndarray,
        target: np.ndarray,
        slopes: np.ndarray,
        intercepts: np.ndarray,
    ) -> np.ndarray:
        """
        Applies the forward transformation to the marginals of the multivariate
        target. Target (real valued) -> empirical cdf [0, 1]

        Parameters
        ----------
        sorted_vec
            Sorted (past) target vector.
        target
            Target that will be transformed.
        slopes
            Slopes of the piece-wise linear function.
        intercepts
            Intercepts of the piece-wise linear function

        Returns
        -------
        transformed_target
            Transformed target vector.
        """
        transformed = list()
        for sorted_vector, t, slope, intercept in zip(
            sorted_vec.transpose(),
            target.transpose(),
            slopes.transpose(),
            intercepts.transpose(),
        ):
            indices = self._search_sorted(sorted_vector, t)
            transformed_value = slope[indices] * t + intercept[indices]
            transformed.append(transformed_value)
        return np.array(transformed).transpose()

    @staticmethod
    def standard_gaussian_cdf(x: np.ndarray) -> np.ndarray:
        u = x / (np.sqrt(2.0))
        return (erf(u) + 1.0) / 2.0

    @staticmethod
    def standard_gaussian_ppf(y: np.ndarray) -> np.ndarray:
        y_clipped = np.clip(y, a_min=1.0e-6, a_max=1.0 - 1.0e-6)
        return np.sqrt(2.0) * erfinv(2.0 * y_clipped - 1.0)

    @staticmethod
    def winsorized_cutoff(m: float) -> float:
        """
        Apply truncation to the empirical CDF estimator to reduce variance as
        described here: https://arxiv.org/abs/0903.0649.

        Parameters
        ----------
        m
            Input empirical CDF value.

        Returns
        -------
        res
            Truncated empirical CDf value.
        """
        res = 1 / (4 * m**0.25 * np.sqrt(3.14 * np.log(m)))
        assert 0 < res < 1
        return res

    @staticmethod
    def _fill(target: np.ndarray, expected_length: int) -> np.ndarray:
        """
        Makes sure target has at least expected_length time-units by repeating
        it or using zeros.

        Parameters
        ----------
        target : shape (seq_len, dim)
        expected_length

        Returns
        -------
            array of shape (target_length, dim)
        """

        current_length, target_dim = target.shape
        if current_length == 0:
            # todo handle the case with no observation better,
            # we could use dataset statistics but for now we use zeros
            filled_target = np.zeros((expected_length, target_dim))
        elif current_length < expected_length:
            filled_target = np.vstack(
                [target for _ in range(expected_length // current_length + 1)]
            )
            filled_target = filled_target[:expected_length]
        elif current_length > expected_length:
            filled_target = target[-expected_length:]
        else:
            filled_target = target

        assert filled_target.shape == (expected_length, target_dim)

        return filled_target


def cdf_to_gaussian_forward_transform(
    input_batch: DataEntry, outputs: np.ndarray
) -> np.ndarray:
    """
    Forward transformation of the CDFtoGaussianTransform.

    Parameters
    ----------
    input_batch
        Input data to the predictor.
    outputs
        Predictor outputs.
    Returns
    -------
    outputs
        Forward transformed outputs.
    """

    def _empirical_cdf_inverse_transform(
        batch_target_sorted: np.ndarray,
        batch_predictions: np.ndarray,
        slopes: np.ndarray,
        intercepts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply forward transformation of the empirical CDF.

        Parameters
        ----------
        batch_target_sorted
            Sorted targets of the input batch.
        batch_predictions
            Predictions of the underlying probability distribution
        slopes
            Slopes of the piece-wise linear function.
        intercepts
            Intercepts of the piece-wise linear function.

        Returns
        -------
        outputs
            Forward transformed outputs.
        """

        num_timesteps = batch_target_sorted.shape[1]
        indices = np.floor(batch_predictions * num_timesteps)
        # indices = indices - 1
        # for now project into [0, 1]
        indices = np.clip(indices, 0, num_timesteps - 1)
        indices = indices.astype(int)

        transformed = np.where(
            np.take_along_axis(slopes, indices, axis=1) != 0.0,
            (
                batch_predictions
                - np.take_along_axis(intercepts, indices, axis=1)
            )
            / np.take_along_axis(slopes, indices, axis=1),
            np.take_along_axis(batch_target_sorted, indices, axis=1),
        )
        return transformed

    # applies inverse cdf to all outputs
    samples = outputs.shape[1]
    for sample_index in range(0, samples):
        outputs[:, sample_index, :, :] = _empirical_cdf_inverse_transform(
            tensor_to_numpy(input_batch["past_target_sorted"]),
            CDFtoGaussianTransform.standard_gaussian_cdf(
                outputs[:, sample_index, :, :]
            ),
            tensor_to_numpy(input_batch["slopes"]),
            tensor_to_numpy(input_batch["intercepts"]),
        )
    return outputs


class ToIntervalSizeFormat(FlatMapTransformation):
    """
    Convert a sparse univariate time series to the `interval-size` format,
    i.e., a two dimensional time series where the first dimension corresponds
    to the time since last positive value (1-indexed), and the second dimension
    corresponds to the size of the demand. This format is used often in the
    intermittent demand literature, where predictions are performed on this
    "dense" time series, e.g., as in Croston's method.

    As an example, the time series `[0, 0, 1, 0, 3, 2, 0, 4]` is converted into
    the 2-dimensional time series `[[3, 2, 1, 2], [1, 3, 2, 4]]`, with a
    shape (2, M) where M denotes the number of non-zero items in the time
    series.

    Parameters
    ----------
    target_field
        The target field to be converted, containing a univariate and sparse
        time series
    drop_empty
        If True, all-zero time series will be dropped.
    discard_first
        If True, the first element in the converted dense series will be
        dropped, replacing the target with a (2, M-1) tet instead. This can be
        used when the first 'inter-demand' time is not well-defined. e.g.,
        when the true starting index of the time series is not known.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        drop_empty: bool = False,
        discard_first: bool = False,
    ) -> None:
        super().__init__()

        self.target_field = target_field
        self.drop_empty = drop_empty
        self.discard_first = discard_first

    def _process_sparse_time_sample(self, a: List) -> Tuple[List, List]:
        a: np.ndarray = np.array(a)
        (non_zero_index,) = np.nonzero(a)

        if len(non_zero_index) == 0:
            return [], []

        times = np.diff(non_zero_index, prepend=-1.0).tolist()
        sizes = a[non_zero_index].tolist()

        if self.discard_first:
            return times[1:], sizes[1:]
        return times, sizes

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        target = data[self.target_field]

        times, sizes = self._process_sparse_time_sample(target)

        if len(times) > 0 or not self.drop_empty:
            data[self.target_field] = [times, sizes]
            yield data
