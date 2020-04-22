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
from .binning_helpers import (
    ensure_binning_monotonicity,
    bin_edges_from_bin_centers,
)

# Standard library imports
from typing import Tuple, Optional

# Third-party imports
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class LocalAbsoluteBinning(Representation):
    """
    A class representing a local absolute binning approach.
    This binning estimates a binning for every single time series on a local level and therefore implicitly acts as
    a scaling mechanism.

    Parameters
    ----------
    num_bins
        The number of discrete bins/buckets that we want values to be mapped to.
        (default: 1024)
    is_quantile
        Whether the binning is quantile or linear. Quantile binning allocated bins based on the cumulative
        distribution function, while linear binning allocates evenly spaced bins.
        (default: True, i.e. quantile binning)
    embedding_size
        The size of the embedding layer.
        (default: round(num_bins**(1/4)))
    pit
        Whether the binning should be used to transform its inputs using a discrete probability integral transform.
        This requires is_quantile=True.
        (default: False)
    mlp_tranf
        Whether we want to post-process the pit-transformed valued using a MLP which can learn an appropriate
        binning, which would ensure that pit models have the same expressiveness as standard quantile binning with
        embedding. This requires pit=True.
        (default: False)
    """

    @validated()
    def __init__(
        self,
        num_bins: int = 1024,
        is_quantile: bool = True,
        embedding_size: int = -1,
        pit: bool = False,
        mlp_transf: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.is_quantile = is_quantile
        self.pit = pit
        self.mlp_transf = mlp_transf

        if embedding_size == -1:
            # Embedding size heuristic that seems to work well in practice. For reference see:
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            self.embedding_size = round(self.num_bins ** (1 / 4))
        else:
            self.embedding_size = embedding_size

        self.bin_edges_hyb = np.array([])
        self.bin_centers_hyb = np.array([])

        with self.name_scope():
            if self.mlp_transf:
                self.mlp = mx.gluon.nn.HybridSequential()
                self.mlp.add(
                    mx.gluon.nn.Dense(
                        units=self.num_bins, activation="relu", flatten=False
                    )
                )
                self.mlp.add(
                    mx.gluon.nn.Dense(units=self.embedding_size, flatten=False)
                )
            else:
                self.mlp = None

            if self.is_output or self.pit:
                self.embedding = lambda x: x
            else:
                self.embedding = nn.Embedding(
                    input_dim=self.num_bins, output_dim=self.embedding_size
                )

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        data_np = data.asnumpy()
        observed_indicator_np = observed_indicator.astype("int32").asnumpy()

        if scale is None:
            # Even though local binning implicitly scales the data, we still return the scale as an input to the model.
            scale = F.expand_dims(
                F.sum(data, axis=-1) / F.sum(observed_indicator, axis=-1), -1
            )

            self.bin_centers_hyb = np.ones((len(data), self.num_bins)) * (-1)
            self.bin_edges_hyb = np.ones((len(data), self.num_bins + 1)) * (-1)

            # Every time series needs to be binned individually
            for i in range(len(data_np)):
                # Identify observed data points.
                data_loc = data_np[i]
                observed_indicator_loc = observed_indicator_np[i]
                data_obs_loc = data_loc[observed_indicator_loc == 1]

                if data_obs_loc.size > 0:
                    # Calculate time series specific bin centers and edges.
                    if self.is_quantile:
                        bin_centers_loc = np.quantile(
                            data_obs_loc, np.linspace(0, 1, self.num_bins)
                        )
                    else:
                        bin_centers_loc = np.linspace(
                            np.min(data_obs_loc),
                            np.max(data_obs_loc),
                            self.num_bins,
                        )
                    self.bin_centers_hyb[i] = ensure_binning_monotonicity(
                        bin_centers_loc
                    )
                    self.bin_edges_hyb[i] = bin_edges_from_bin_centers(
                        self.bin_centers_hyb[i]
                    )

                    # Bin the time series.
                    data_obs_loc_binned = np.digitize(
                        data_obs_loc, bins=self.bin_edges_hyb[i], right=False
                    )
                else:
                    data_obs_loc_binned = []

                # Write the binned time series back into the data array.
                data_loc[observed_indicator_loc == 1] = data_obs_loc_binned
                data_np[i] = data_loc
        else:
            self.bin_edges_hyb = np.repeat(
                self.bin_edges_hyb,
                len(data_np) / len(self.bin_edges_hyb),
                axis=0,
            )
            self.bin_centers_hyb = np.repeat(
                self.bin_centers_hyb,
                len(data_np) / len(self.bin_centers_hyb),
                axis=0,
            )

            for i in range(len(data_np)):
                data_loc = data_np[i]
                observed_indicator_loc = observed_indicator_np[i]
                data_obs_loc = data_loc[observed_indicator_loc == 1]

                # Bin the time series based on previously computed bin edges.
                data_obs_loc_binned = np.digitize(
                    data_obs_loc, bins=self.bin_edges_hyb[i], right=False
                )

                data_loc[observed_indicator_loc == 1] = data_obs_loc_binned
                data_np[i] = data_loc

        data = mx.nd.array(data_np)

        # In PIT mode, we rescale the binned data to [0,1] and optionally pass the data through a MLP to achieve the
        # same level of expressiveness as binning with embedding.
        if self.pit:
            data = data / self.num_bins
            data_exp = data.expand_dims(-1)
            if self.mlp_transf:
                return self.mlp(data_exp).swapaxes(1, 2), scale
            else:
                return data_exp.swapaxes(1, 2), scale

        # In output mode, no embedding is used since the data is directly used to compute the loss.
        # In input mode, we embed the categorical data to ensure that the network can learn similarities between bins.
        if self.is_output:
            return data, scale
        else:
            emb = self.embedding(data)
            return emb.swapaxes(1, 2), scale

    def post_transform(self, F, x: Tensor):
        bin_cent = mx.nd.array(self.bin_centers_hyb)
        x_oh = F.one_hot(F.squeeze(x), self.num_bins)

        # Pick corresponding bin centers for all samples
        x = F.sum(bin_cent * x_oh, axis=1).expand_dims(-1)

        return x
