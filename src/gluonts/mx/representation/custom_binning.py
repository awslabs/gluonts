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

from .binning_helpers import bin_edges_from_bin_centers
from .representation import Representation


class CustomBinning(Representation):
    """
    A class representing binned representations with custom centers.

    Parameters
    ----------
    bin_centers
        The bins to be used to discretize the data.
        (default: 1024)
    """

    @validated()
    def __init__(self, bin_centers: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bin_edges = self.params.get_constant(
            "bin_edges", mx.nd.array(bin_edges_from_bin_centers(bin_centers))
        )
        self.bin_centers = self.params.get_constant(
            "bin_centers", mx.nd.array(bin_centers)
        )

        self.num_bins = len(bin_centers)

    def initialize_from_dataset(
        self, input_dataset: Dataset, ctx: mx.Context = get_mxnet_context()
    ):
        self.initialize_from_array(np.array([]), ctx)

    def initialize_from_array(
        self, input_array: np.ndarray, ctx: mx.Context = get_mxnet_context()
    ):
        with ctx:
            self.bin_edges.initialize()
            self.bin_centers.initialize()

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

        # Discretize the data.
        # Note: Replace this once there is a clean way to do this in MXNet.
        data = F.Custom(data, bin_edges, op_type="digitize")

        # Store bin centers for later usage in post_transform.
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

        return transf_samples
