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

from typing import Iterator, List, Tuple, Optional

import numpy as np

from gluonts.core.component import validated, DType
from gluonts.core.exception import assert_data_error
from gluonts.dataset.common import DataEntry
from gluonts.model.common import Tensor
from gluonts.support.util import erf, erfinv

from ._base import (
    SimpleTransformation,
    MapTransformation,
    FlatMapTransformation,
)


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
                f"Unexpected field type {type(v).__name__}, expected "
                f"np.ndarray or list[np.ndarray]"
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
        self.field_name = field_name
        self.target_field = target_field
        self.observed_values_field = observed_values_field
        self.num_samples = num_samples
        self.shuffle = shuffle

    def flatmap_transform(
        self, data: DataEntry, is_train: bool, slice_future_target: bool = True
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
    Marginal transformation that transforms the target via an empirical CDF
    to a standard gaussian as described here: https://arxiv.org/abs/1910.03002

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
        1) Reshaping the data.
        2) Normalizing the target length.
        3) Adding noise to avoid zero slopes (training only)
        4) Sorting the target to compute the empirical CDF

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
            past_target_vec = self._add_noise(past_target_vec)

        past_target_vec.sort(axis=0)

        assert past_target_vec.shape == (expected_length, self.target_dim)

        data[self.sort_target_field] = past_target_vec

    def _calc_pw_linear_params(self, data: DataEntry):
        """
        Calculates the piece-wise linear parameters to interpolate between
        the observed values in the empirical CDF.

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
        data[self.slopes_field] = slopes
        data[self.intercepts_field] = intercepts

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
    def _add_noise(x: np.array) -> np.array:
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
        sorted_vec: np.array, to_insert_vec: np.array
    ) -> np.array:
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
        sorted_vec: np.array,
        target: np.array,
        slopes: np.array,
        intercepts: np.array,
    ) -> np.array:
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
        for sorted, t, slope, intercept in zip(
            sorted_vec.transpose(),
            target.transpose(),
            slopes.transpose(),
            intercepts.transpose(),
        ):
            indices = self._search_sorted(sorted, t)
            transformed_value = slope[indices] * t + intercept[indices]
            transformed.append(transformed_value)
        return np.array(transformed).transpose()

    @staticmethod
    def standard_gaussian_cdf(x: np.array) -> np.array:
        u = x / (np.sqrt(2.0))
        return (erf(np, u) + 1.0) / 2.0

    @staticmethod
    def standard_gaussian_ppf(y: np.array) -> np.array:
        y_clipped = np.clip(y, a_min=1.0e-6, a_max=1.0 - 1.0e-6)
        return np.sqrt(2.0) * erfinv(np, 2.0 * y_clipped - 1.0)

    @staticmethod
    def winsorized_cutoff(m: np.array) -> np.array:
        """
        Apply truncation to the empirical CDF estimator to reduce variance as
        described here: https://arxiv.org/abs/0903.0649

        Parameters
        ----------
        m
            Input array with empirical CDF values.

        Returns
        -------
        res
            Truncated empirical CDf values.
        """
        res = 1 / (4 * m ** 0.25 * np.sqrt(3.14 * np.log(m)))
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
    input_batch: DataEntry, outputs: Tensor
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
        batch_target_sorted: Tensor,
        batch_predictions: Tensor,
        slopes: Tensor,
        intercepts: Tensor,
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
        slopes = slopes.asnumpy()
        intercepts = intercepts.asnumpy()

        batch_target_sorted = batch_target_sorted.asnumpy()
        batch_size, num_timesteps, target_dim = batch_target_sorted.shape
        indices = np.floor(batch_predictions * num_timesteps)
        # indices = indices - 1
        # for now project into [0, 1]
        indices = np.clip(indices, 0, num_timesteps - 1)
        indices = indices.astype(np.int)

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
    batch_size, samples, target_dim, time = outputs.shape
    for sample_index in range(0, samples):
        outputs[:, sample_index, :, :] = _empirical_cdf_inverse_transform(
            input_batch["past_target_sorted"],
            CDFtoGaussianTransform.standard_gaussian_cdf(
                outputs[:, sample_index, :, :]
            ),
            input_batch["slopes"],
            input_batch["intercepts"],
        )
    return outputs
