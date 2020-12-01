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

from mxnet.gluon import nn

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .representation import Representation

# LearnedBinning = Union[GlobalRelativeBinning, LocalAbsoluteBinning]


class DiscretePIT(Representation):
    """
    A class representing a discrete probability integral transform of a given
    quantile-based learned binning.  Note that this representation is intended
    to be applied on top of a quantile-based binning representation.

    Parameters
    ----------
    num_bins
        Number of bins used by the data on which this representation is
        applied.
    mlp_tranf
        Whether we want to post-process the pit-transformed valued using a MLP
        which can learn an appropriate binning, which would ensure that pit
        models have the same expressiveness as standard quantile binning with
        embedding.
        (default: False)
    embedding_size
        The desired layer output size if mlp_tranf=True. By default, the
        following heuristic is used:
        https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        (default: round(num_bins**(1/4)))
    """

    @validated()
    def __init__(
        self,
        num_bins: int,
        mlp_transf: bool = False,
        embedding_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.mlp_transf = mlp_transf

        if embedding_size is None:
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
        data = data / self.num_bins
        if self.mlp_transf:
            data = F.expand_dims(data, axis=-1)
            data = self.mlp(data)
        return data, scale, rep_params

    def post_transform(
        self, F, samples: Tensor, scale: Tensor, rep_params: List[Tensor]
    ) -> Tensor:
        samples = samples * F.full(1, self.num_bins)
        samples = F.Custom(
            samples, F.arange(self.num_bins), op_type="digitize"
        )
        return samples
