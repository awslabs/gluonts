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

from .representation import Representation
from .binning_helpers import bin_edges_from_bin_centers

# Standard library imports
from typing import Tuple, Optional

# Third-party imports
import numpy as np
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class CustomBinning(Representation):
    """
    A class representing binned representations with custom centers.

    Parameters
    ----------
    bin_centers
        The bins to be used to discritize the data.
        (default: 1024)
    """

    @validated()
    def __init__(self, bin_centers: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bin_centers = bin_centers
        self.bin_edges = bin_edges_from_bin_centers(bin_centers)
        self.num_bins = len(bin_centers)
        self.scale = np.array([])

        self.bin_centers_hyb = np.array([])

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Calculate local scale if scale is not already supplied.
        if scale is None:
            scale = F.expand_dims(
                F.sum(data, axis=-1) / F.sum(observed_indicator, axis=-1), -1
            )
            # Clip scale on the bottom to prevent division by zero.
            scale = F.clip(scale, 1e-20, np.inf)

        # Discretize the data.
        # Note: Replace this once there is a clean way to do this in MXNet.
        data_binned = np.digitize(
            data.asnumpy(), bins=self.bin_edges, right=False
        )

        data = F.array(data_binned)

        # Store bin centers for later usage in post_transform.
        self.bin_centers_hyb = np.repeat(
            np.swapaxes(np.expand_dims(self.bin_centers, axis=-1), 0, 1),
            len(data),
            axis=0,
        )

        return data, scale

    def post_transform(self, F, x: Tensor):
        bin_cent = F.array(self.bin_centers_hyb)
        x_oh = F.one_hot(F.squeeze(x), self.num_bins)

        # Pick corresponding bin centers for all samples
        x = F.sum(bin_cent * x_oh, axis=1).expand_dims(-1)

        return x
