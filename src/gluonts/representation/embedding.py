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
from typing import Tuple, Optional, Union

from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
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
    def __init__(self, binning: Binning, size: int = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binning = binning
        self.register_child(binning)
        self.num_bins = binning.num_bins

        if size == -1:
            # Embedding size heuristic that seems to work well in practice. For reference see:
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            self.size = round(self.num_bins ** (1 / 4))
        else:
            self.size = size

        self.embedding = nn.Embedding(
            input_dim=self.num_bins, output_dim=self.size
        )

    def initialize_from_dataset(self, input_dataset: Dataset):
        self.binning.initialize_from_dataset(input_dataset)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        repr_data, scale = self.binning(data, observed_indicator, scale)

        emb_data = self.embedding(repr_data)
        emb_data = emb_data.swapaxes(1, 2)

        return emb_data, scale
