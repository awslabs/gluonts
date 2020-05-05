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
from .custom_binning import CustomBinning
from .global_relative_binning import GlobalRelativeBinning
from .local_absolute_binning import LocalAbsoluteBinning

# Standard library imports
from typing import Tuple, Optional, Union, List
import mxnet as mx
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated, get_mxnet_context
from gluonts.model.common import Tensor
from gluonts.dataset.common import Dataset


Binning = Union[CustomBinning, GlobalRelativeBinning, LocalAbsoluteBinning]


class Embedding(Representation):
    """
    A class representing an embedding operation on top of a given binning.
    
    Parameters
    ----------
    binning
        The underlying binning.
    size
        The desired embedding size.
        (default: round(num_bins**(1/4)))
    """

    @validated()
    def __init__(
        self, binning: Binning, size: Optional[int] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.binning = binning
        self.register_child(binning)
        self.num_bins = binning.num_bins

        if size is None:
            # Embedding size heuristic that seems to work well in practice. For reference see:
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            self.size = round(self.num_bins ** (1 / 4))
        else:
            self.size = size

        self.embedding = nn.Embedding(
            input_dim=self.num_bins, output_dim=self.size
        )

    def initialize_from_dataset(
        self, input_dataset: Dataset, ctx: mx.Context = get_mxnet_context()
    ):
        self.binning.initialize_from_dataset(input_dataset, ctx)

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
        repr_data, scale, rep_params = self.binning(
            data, observed_indicator, scale, rep_params
        )

        emb_data = self.embedding(repr_data)

        return emb_data, scale, rep_params
