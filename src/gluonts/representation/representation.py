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

# Third-party imports
from mxnet.gluon import nn
import numpy as np

# Standard library imports
from typing import Tuple, Optional

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.dataset.common import Dataset


class Representation(nn.HybridBlock):
    """
        An abstract class representing input/output representations.

        Parameters
        ----------
        is_output
            Whether the representation is an input or output representation. It is necessary to make this distinction
            because of different shaping considerations and optional child-block creation.
            (default: False)
    """

    @validated()
    def __init__(self, is_output: bool = False):
        super().__init__()
        self.is_output = is_output

    def initialize_from_dataset(self, input_dataset: Dataset):
        r"""
        Perform some computation based on an entire dataset.

        Parameters
        ----------
        input_dataset
            GluonTS dataset.
        """
        pass

    def initialize_from_array(self, input_array: np.ndarray):
        r"""
        Perform some computation based on a numpy array.

        Parameters
        ----------
        input_array
            Numpy array.
        """
        pass

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Transform the data into the desired representation.

        Parameters
        ----------
        F
        data
            Target data.
        observed_indicator
            Target observed indicator.
        scale
            Pre-computed scale.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Tuple consisting of the transformed data and the computed scale.
        """
        raise NotImplementedError

    def post_transform(self, F, samples: Tensor):
        r"""
        Transform samples back to the original representation.

        Parameters
        ----------
        samples
            Samples from a distribution.
        """
        return samples
