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

# Standard library imports
from typing import Tuple, Optional, Union, List
import numpy as np
import mxnet as mx
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated, get_mxnet_context
from gluonts.model.common import Tensor
from gluonts.dataset.common import Dataset


class Embedding(Representation):
    """
    A class representing an embedding operation on top of a given binning.
    Note that this representation is intended to applied on top of categorical/binned data.
    
    Parameters
    ----------
    num_bins
        The number of categories/bins of the data on which this representation is applied.
    size
        The desired embedding size.
        (default: round(num_bins**(1/4)))
    """

    @validated()
    def __init__(
        self, num_bins: int, size: Optional[int] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_bins = num_bins

        if size is None:
            # Embedding size heuristic that seems to work well in practice. For reference see:
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            self.size = round(self.num_bins ** (1 / 4))
        else:
            self.size = size

        self.embedding = nn.Embedding(
            input_dim=self.num_bins, output_dim=self.size
        )

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
        data = self.embedding(data)
        return data, scale, rep_params
