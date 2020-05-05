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
from .global_relative_binning import GlobalRelativeBinning
from .local_absolute_binning import LocalAbsoluteBinning

# Standard library imports
from typing import Tuple, Optional, Union, List
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated, get_mxnet_context
from gluonts.model.common import Tensor
from gluonts.dataset.common import Dataset


LearnedBinning = Union[GlobalRelativeBinning, LocalAbsoluteBinning]


class DiscretePIT(Representation):
    """
    A class representing a discrete probability integral transform of a given quantile-based learned binning.

    Parameters
    ----------
    learned_binning
        The underlying binning. This needs to be quantile-based, i.e. is_quantile needs to be True.    
    mlp_tranf
        Whether we want to post-process the pit-transformed valued using a MLP which can learn an appropriate
        binning, which would ensure that pit models have the same expressiveness as standard quantile binning with
        embedding.
        (default: False)
    embedding_size
        The desired layer output size if mlp_tranf=True.
        (default: round(num_bins**(1/4)))
    """

    @validated()
    def __init__(
        self,
        learned_binning: LearnedBinning,
        mlp_transf: bool = False,
        embedding_size: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert (
            learned_binning.is_quantile
        ), "PIT requires CDF-transformed values."

        self.learned_binning = learned_binning
        self.register_child(learned_binning)
        self.num_bins = learned_binning.num_bins
        self.mlp_transf = mlp_transf

        if embedding_size == -1:
            # Embedding size heuristic that seems to work well in practice. For reference see:
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            self.embedding_size = round(self.num_bins ** (1 / 4))
        else:
            self.embedding_size = embedding_size

        if mlp_transf:
            self.mlp = nn.HybridSequential()
            self.mlp.add(
                nn.Dense(units=self.num_bins, activation="relu", flatten=False)
            )
            self.mlp.add(nn.Dense(units=self.embedding_size, flatten=False))
        else:
            self.mlp = None

    def initialize_from_dataset(
        self, input_dataset: Dataset, ctx: mx.Context = get_mxnet_context()
    ):
        self.learned_binning.initialize_from_dataset(input_dataset, ctx)

    def initialize_from_array(
        self, input_array: np.ndarray, ctx: mx.Context = get_mxnet_context()
    ):
        self.learned_binning.initialize_from_array(input_array, ctx)

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
        repr_data, scale, rep_params = self.learned_binning(
            data, observed_indicator, scale, rep_params,
        )

        repr_data = repr_data / self.num_bins
        if self.mlp_transf:
            repr_data = F.expand_dims(repr_data, axis=-1)
            repr_data = self.mlp(repr_data)

        return repr_data, scale, rep_params
