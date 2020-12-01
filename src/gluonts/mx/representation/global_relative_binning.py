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

from typing import List, Optional, Tuple

import mxnet as mx
import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.mx import Tensor
from gluonts.mx.context import get_mxnet_context

from .binning_helpers import (
    bin_edges_from_bin_centers,
    ensure_binning_monotonicity,
)
from .representation import Representation


class GlobalRelativeBinning(Representation):
    """
    A class representing a global relative binning approach.
    This binning first rescales all input series by their respective mean (relative) and then performs one binning
    across all series (global).

    Parameters
    ----------
    num_bins
        The number of discrete bins/buckets that we want values to be mapped to.
        (default: 1024)
    is_quantile
        Whether the binning is quantile or linear. Quantile binning allocated bins based on the cumulative
        distribution function, while linear binning allocates evenly spaced bins.
        (default: True, i.e. quantile binning)
    linear_scaling_limit
        The linear scaling limit. Values which are larger than linear_scaling_limit times the mean will be capped at
        linear_scaling_limit.
        (default: 10)
    quantile_scaling_limit
        The quantile scaling limit. Values which are larger than the quantile evaluated at quantile_scaling_limit
        will be capped at the quantile evaluated at quantile_scaling_limit.
        (default: 0.99)
    """

    @validated()
    def __init__(
        self,
        num_bins: int = 1024,
        is_quantile: bool = True,
        linear_scaling_limit: int = 10,
        quantile_scaling_limit: float = 0.99,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.is_quantile = is_quantile

        self.linear_scaling_limit = linear_scaling_limit
        self.quantile_scaling_limit = quantile_scaling_limit

        self.bin_edges = self.params.get_constant(
            "bin_edges", mx.nd.zeros(self.num_bins + 1)
        )
        self.bin_centers = self.params.get_constant(
            "bin_centers", mx.nd.zeros(self.num_bins)
        )

    def initialize_from_dataset(
        self, input_dataset: Dataset, ctx: mx.Context = get_mxnet_context()
    ):
        # Rescale all time series in training set.
        train_target_sequence = np.array([])
        for train_entry in input_dataset:
            train_entry_target = train_entry["target"]
            train_tar_mean = np.mean(train_entry_target)
            train_entry_target /= train_tar_mean
            train_target_sequence = np.concatenate(
                [train_target_sequence, train_entry_target]
            )
        self.initialize_from_array(train_target_sequence, ctx)

    def initialize_from_array(
        self, input_array: np.ndarray, ctx: mx.Context = get_mxnet_context()
    ):
        # Calculate bin centers and bin edges using linear or quantile binning..
        if self.is_quantile:
            bin_centers = np.quantile(
                input_array,
                np.linspace(0, self.quantile_scaling_limit, self.num_bins),
            )
            bin_centers = ensure_binning_monotonicity(bin_centers)
        else:
            has_negative_data = np.any(input_array < 0)
            low = -self.linear_scaling_limit if has_negative_data else 0
            high = self.linear_scaling_limit
            bin_centers = np.linspace(low, high, self.num_bins)
        bin_edges = bin_edges_from_bin_centers(bin_centers)

        # Store bin centers and edges since their are globally applicable to all time series.
        with ctx:
            self.bin_edges.initialize()
            self.bin_centers.initialize()
        self.bin_edges.set_data(mx.nd.array(bin_edges))
        self.bin_centers.set_data(mx.nd.array(bin_centers))

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
        rep_params: List[Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        bin_edges = kwargs["bin_edges"]
        bin_centers = kwargs["bin_centers"]

        # Calculate local scale if scale is not already supplied.
        if scale is None:
            scale = F.expand_dims(
                F.sum(data * observed_indicator, axis=-1)
                / F.sum(observed_indicator, axis=-1),
                -1,
            )
            # Clip scale on the bottom to prevent division by zero.
            scale = F.clip(scale, 1e-20, np.inf)

        # Rescale the data.
        data_rescaled = F.broadcast_div(data, scale)

        # Discretize the data.
        data = F.Custom(data_rescaled, bin_edges, op_type="digitize")

        # Bin centers for later usage in post_transform.
        bin_centers_hyb = F.repeat(
            F.expand_dims(bin_centers, axis=0), len(data), axis=0
        )

        return data, scale, [bin_centers_hyb, bin_edges]

    def post_transform(
        self, F, samples: Tensor, scale: Tensor, rep_params: List[Tensor]
    ) -> Tensor:
        bin_centers_hyb = rep_params[0]

        transf_samples = F.one_hot(F.squeeze(samples), self.num_bins)

        # Pick corresponding bin centers for all samples
        transf_samples = F.sum(
            bin_centers_hyb * transf_samples, axis=1
        ).expand_dims(-1)

        # Transform bin centers back to the oiginal scale
        x = F.broadcast_mul(scale, transf_samples)

        return x
