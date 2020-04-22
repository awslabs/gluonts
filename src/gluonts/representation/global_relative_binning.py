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
import math

# Third-party imports
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.dataset.common import Dataset


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
        embedding_size: int = -1,
        linear_scaling_limit: int = 10,
        quantile_scaling_limit: float = 0.99,
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

        self.bin_centers_hyb = np.array([])

        self.linear_scaling_limit = linear_scaling_limit
        self.quantile_scaling_limit = quantile_scaling_limit

        self.bin_centers = np.array([])
        self.bin_edges = np.array([])
        self.scale = np.array([])

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

    def initialize_from_dataset(self, input_dataset: Dataset):
        # Rescale all time series in training set.
        train_target_sequence = np.array([])
        for train_entry in input_dataset:
            train_entry_target = train_entry["target"]
            train_tar_mean = np.mean(train_entry_target)
            train_entry_target /= train_tar_mean
            train_target_sequence = np.concatenate(
                [train_target_sequence, train_entry_target]
            )
        self.initialize_from_array(train_target_sequence)

    def initialize_from_array(self, input_array: np.ndarray):
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
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges

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
        self.scale = scale.asnumpy()

        # Rescale the data.
        data_rescaled = data.asnumpy() / np.repeat(
            self.scale, data.shape[1], axis=1
        )

        # Discretize the data.
        # Note: Replace this once there is a clean way to do this in MXNet.
        data_binned = np.digitize(
            data_rescaled, bins=self.bin_edges, right=False
        )

        data = F.array(data_binned)

        # Store bin centers for later usage in post_transform.
        self.bin_centers_hyb = np.repeat(
            np.swapaxes(np.expand_dims(self.bin_centers, axis=-1), 0, 1),
            len(data),
            axis=0,
        )

        # In PIT mode, we rescale the binned data to [0,1] and optionally pass the data through a MLP to achieve the
        # same level of expressiveness as binning with embedding.
        if self.pit:
            data = data / self.num_bins
            if not self.is_output:
                data = data.expand_dims(-1)
            if self.mlp_transf:
                return self.mlp(data).swapaxes(1, 2), scale
            else:
                if self.is_output:
                    return data, scale
                else:
                    return data.swapaxes(1, 2), scale

        # In output mode, no embedding is used since the data is directly used to compute the loss.
        # In input mode, we embed the categorical data to ensure that the network can learn similarities between bins.
        if self.is_output:
            return data, scale
        else:
            emb = self.embedding(data)
            return emb.swapaxes(1, 2), scale

    def post_transform(self, F, x: Tensor):
        if self.pit:
            # Transform the data back from [0,1] to the set of bins
            x_upsc = x * F.array([self.num_bins])
            x_np = x_upsc.asnumpy()
            x_np = np.digitize(x_np, np.arange(self.num_bins))
            x = F.array(x_np)

        bin_cent = F.array(self.bin_centers_hyb)
        x_oh = F.one_hot(F.squeeze(x), self.num_bins)

        # Pick corresponding bin centers for all samples
        x = F.sum(bin_cent * x_oh, axis=1).expand_dims(-1)

        # Transform bin centers back to the oiginal scale
        x = F.broadcast_mul(F.array(self.scale), x)

        return x
