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

from typing import List
import torch
from gluonts.torch.distributions.distribution_output import Output
from gluonts.core.component import validated


class QuantileOutput(Output):
    """
    Output layer using a quantile loss and projection layer to connect the
    quantile output to the network.

    Parameters
    ----------
    quantiles
        list of quantiles to compute loss over.

    quantile_weights
        weights of the quantiles.
    """

    @validated()
    def __init__(self, quantiles: List[float]) -> None:
        assert len(quantiles) > 0
        assert all(0.0 < q < 1.0 for q in quantiles)
        self._quantiles = quantiles
        self.num_quantiles = len(self._quantiles)
        self.args_dim = {"quantiles_pred": self.num_quantiles}

    @property
    def quantiles(self) -> List[float]:
        return self._quantiles

    def domain_map(self, quantiles_pred: torch.Tensor):
        return quantiles_pred

    def quantile_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean quantile loss.

        Parameters
        ----------
        y_true
            Ground truth values, shape [N_1, ..., N_k]
        y_pred
            Predicted quantiles, shape [N_1, ..., N_k num_quantiles]

        Returns
        -------
        loss
            Quantile loss, shape [N_1, ..., N_k]
        """
        y_true = y_true.unsqueeze(-1)
        quantiles = torch.tensor(
            self.quantiles, dtype=y_pred.dtype, device=y_pred.device
        )
        return 2 * (
            (y_true - y_pred) * ((y_true <= y_pred).float() - quantiles)
        ).abs().sum(dim=-1)
