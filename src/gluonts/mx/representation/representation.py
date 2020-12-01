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
from mxnet.gluon import nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.mx import Tensor
from gluonts.mx.context import get_mxnet_context


class Representation(nn.HybridBlock):
    """
    An abstract class representing input/output representations.
    """

    @validated()
    def __init__(self):
        super().__init__()

    def initialize_from_dataset(
        self, input_dataset: Dataset, ctx: mx.Context = get_mxnet_context()
    ):
        r"""
        Initialize the representation based on an entire dataset.

        Parameters
        ----------
        input_dataset
            GluonTS dataset.
        ctx
            MXNet context.
        """
        pass

    def initialize_from_array(
        self, input_array: np.ndarray, ctx: mx.Context = get_mxnet_context()
    ):
        r"""
        Initialize the representation based on a numpy array.

        Parameters
        ----------
        input_array
            Numpy array.
        ctx
            MXNet context.
        """
        pass

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
        rep_params
            Additional pre-computed representation parameters.
        **kwargs,
            Additional block-specfic parameters.

        Returns
        -------
        Tuple[Tensor, Tensor, List[Tensor]]
            Tuple consisting of the transformed data, the computed scale,
            and additional parameters to be passed to post_transform.
        """
        return data, F.ones_like(data), []

    def post_transform(
        self, F, samples: Tensor, scale: Tensor, rep_params: List[Tensor]
    ) -> Tensor:
        r"""
        Transform samples back to the original representation.

        Parameters
        ----------
        samples
            Samples from a distribution.
        scale
            The scale of the samples.
        rep_params
            Additional representation-specific parameters used during post
            transformation.

        Returns
        -------
        Tensor
            Post-transformed samples.
        """
        return samples
