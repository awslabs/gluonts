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

from typing import Tuple, Optional

import mxnet.ndarray as nd
from mxnet.gluon import nn

from gluonts.core.component import validated
from gluonts.mx import Tensor


class Scaler(nn.HybridBlock):
    """
    Base class for blocks used to scale data.

    Parameters
    ----------
    keepdims
        toggle to keep the dimension of the input tensor.
    axis
        specify the axis over which to scale. Default is 1 for (N, T, C)
        shaped input tensor.
    """

    def __init__(self, keepdims: bool = False, axis: int = 1):

        super().__init__()
        self.keepdims = keepdims
        self.axis = axis

    def compute_scale(self, F, data: Tensor, observed_indicator: Tensor):
        """
        Computes the scale of the given input data.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        data
            tensor containing the data to be scaled.

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.
        """
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, data: Tensor, observed_indicator: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        data
            tensor containing the data to be scaled.

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            Tensor containing the "scaled" data.
        Tensor
            Tensor containing the scale: this has the same shape as the data, except for the axis ``axis``
            along which the scale is computed, which is removed if ``keepdims == False``, and kept with
            length 1 otherwise. For example, if ``data`` has shape ``(N, T, C)`` and ``axis ==1 ``, then
            ``scale`` has shape ``(N, C)`` if ``keepdims == False``, and ``(N, 1, C)`` otherwise.

        """
        scale = self.compute_scale(F, data, observed_indicator)

        if self.keepdims:
            scale = scale.expand_dims(axis=self.axis)
            return F.broadcast_div(data, scale), scale
        else:
            return (
                F.broadcast_div(data, scale.expand_dims(axis=self.axis)),
                scale,
            )


class MeanScaler(Scaler):
    """
    The ``MeanScaler`` computes a per-item scale according to the average
    absolute value over time of each item. The average is computed only among
    the observed values in the data tensor, as indicated by the second
    argument. Items with no observed data are assigned a scale based on the
    global average.

    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    @validated()
    def __init__(
        self,
        minimum_scale: float = 1e-10,
        default_scale: Optional[float] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    def compute_scale(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,  # shapes (N, T, C) or (N, C, T)
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        data
            tensor containing the data to be scaled.

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            shape (N, C), computed according to the
            average absolute value over time of the observed values.
        """

        # these will have shape (N, C)
        num_observed = F.sum(observed_indicator, axis=self.axis)
        sum_observed = (data.abs() * observed_indicator).sum(axis=self.axis)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(axis=0)
        denominator = F.maximum(total_observed, 1.0)
        if self.default_scale is not None:
            default_scale = self.default_scale * F.ones_like(num_observed)
        else:
            # shape (C, )
            default_scale = sum_observed.sum(axis=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = F.maximum(num_observed, 1.0)
        scale = sum_observed / denominator  # shape (N, C)

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        cond = F.broadcast_greater(sum_observed, F.zeros_like(sum_observed))
        scale = F.where(
            cond,
            scale,
            F.broadcast_mul(default_scale, F.ones_like(num_observed)),
        )

        return F.maximum(scale, self.minimum_scale)


class MinMax(Scaler):
    """
    The 'MinMax' scales the input data using a min-max approach along the specified axis.
    """

    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_scale(
        self, F, data: Tensor, observed_indicator: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        data
            tensor containing the data to be scaled.

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            shape (N, T, C) or (N, C, T) scaled along the specified axis.

        """

        axis_zero = nd.prod(
            data == data.zeros_like(), self.axis, keepdims=True
        )  # Along the specified axis, which array are always at zero
        axis_zero = nd.broadcast_to(
            axis_zero, shape=data.shape
        )  # Broadcast it to the shape of data

        min_val = nd.where(
            1 - observed_indicator,
            nd.broadcast_to(data.max(keepdims=True), shape=data.shape),
            data,
        ).min(
            axis=self.axis, keepdims=True
        )  # return the min value along specified axis while ignoring value according to observed_indicator
        max_val = nd.where(
            1 - observed_indicator,
            nd.broadcast_to(data.min(keepdims=True), shape=data.shape),
            data,
        ).max(
            axis=self.axis, keepdims=True
        )  # return the max value along specified axis while ignoring value according to observed_indicator

        scaled_data = (data - min_val) / (max_val - min_val)  # Rescale
        scaled_data = nd.where(
            axis_zero, scaled_data.zeros_like(), scaled_data
        )  # Clip Nan values to zero if the data was equal to zero along specified axis
        scaled_data = nd.where(
            scaled_data != scaled_data, scaled_data.ones_like(), scaled_data
        )  # Clip the Nan values to one. scaled_date!=scaled_data tells us where the Nan values are in scaled_data

        return nd.where(
            1 - observed_indicator, scaled_data.zeros_like(), scaled_data
        )  # Replace data with zero where observed_indicator tells us to.


class NOPScaler(Scaler):
    """
    The ``NOPScaler`` assigns a scale equals to 1 to each input item, i.e.,
    no scaling is applied upon calling the ``NOPScaler``.
    """

    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # noinspection PyMethodOverriding
    def compute_scale(
        self, F, data: Tensor, observed_indicator: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        data
            tensor containing the data to be scaled.

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            shape (N, C), identically equal to 1.
        """
        return F.ones_like(data).mean(axis=self.axis)
