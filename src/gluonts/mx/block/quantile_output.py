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

from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss

from gluonts.core.component import validated
from gluonts.mx import Tensor


class QuantileLoss(Loss):
    @validated()
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
        weight: Optional[float] = None,
        batch_axis: int = 0,
        **kwargs,
    ) -> None:
        """
        Represents the quantile loss used to fit decoders that learn quantiles.

        Parameters
        ----------
        quantiles
            list of quantiles to compute loss over.

        quantile_weights
            weights of the quantiles.

        weight
            weighting of the loss.

        batch_axis
            indicates axis that represents the batch.
        """
        super().__init__(weight, batch_axis, **kwargs)

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = (
            [1.0 / self.num_quantiles for i in range(self.num_quantiles)]
            if not quantile_weights
            else quantile_weights
        )

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, y_true: Tensor, y_pred: Tensor, sample_weight=None
    ):
        """
        Compute the weighted sum of quantile losses.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        y_true
            true target, shape (N1 x N2 x ... x Nk x dimension of time series
            (normally 1))
        y_pred
            predicted target, shape (N1 x N2 x ... x Nk x num_quantiles)
        sample_weight
            sample weights

        Returns
        -------
        Tensor
            weighted sum of the quantile losses, shape N1 x N1 x ... Nk
        """
        if self.num_quantiles > 1:
            y_pred_all = F.split(
                y_pred, axis=-1, num_outputs=self.num_quantiles, squeeze_axis=1
            )
        else:
            y_pred_all = [F.squeeze(y_pred, axis=-1)]

        qt_loss = []
        for i, y_pred_q in enumerate(y_pred_all):
            q = self.quantiles[i]
            weighted_qt = (
                self.compute_quantile_loss(F, y_true, y_pred_q, q)
                * self.quantile_weights[i]
            )
            qt_loss.append(weighted_qt)
        stacked_qt_losses = F.stack(*qt_loss, axis=-1)
        sum_qt_loss = F.mean(
            stacked_qt_losses, axis=-1
        )  # avg across quantiles
        if sample_weight is not None:
            return sample_weight * sum_qt_loss
        else:
            return sum_qt_loss

    @staticmethod
    def compute_quantile_loss(
        F, y_true: Tensor, y_pred_p: Tensor, p: float
    ) -> Tensor:
        """
        Compute the quantile loss of the given quantile

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        y_true
            true target, shape (N1 x N2 x ... x Nk x dimension of time series
            (normally 1)).

        y_pred_p
            predicted target quantile, shape (N1 x N2 x ... x Nk x 1).

        p
            quantile error to compute the loss.

        Returns
        -------
        Tensor
            quantile loss, shape: (N1 x N2 x ... x Nk x 1)
        """

        under_bias = p * F.maximum(y_true - y_pred_p, 0)
        over_bias = (1 - p) * F.maximum(y_pred_p - y_true, 0)

        qt_loss = 2 * (under_bias + over_bias)

        return qt_loss


class ProjectParams(nn.HybridBlock):
    """
    Defines a dense layer to compute the projection weights into the quantile
    space.

    Parameters
    ----------
    num_quantiles
        number of quantiles to compute the projection.
    """

    @validated()
    def __init__(self, num_quantiles, **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.projection = nn.Dense(units=num_quantiles, flatten=False)

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        x
            input tensor

        Returns
        -------
        Tensor
            output of the projection layer
        """
        return self.projection(x)


class QuantileOutput:
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
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
    ) -> None:
        self.quantiles = quantiles
        self.quantile_weights = quantile_weights

    def get_loss(self) -> nn.HybridBlock:
        """
        Returns
        -------
        nn.HybridBlock
            constructs quantile loss object.
        """
        return QuantileLoss(
            quantiles=self.quantiles, quantile_weights=self.quantile_weights
        )

    def get_quantile_proj(self, **kwargs) -> nn.HybridBlock:
        """
        Returns
        -------
        nn.HybridBlock
            constructs projection parameter object.

        """
        return ProjectParams(len(self.quantiles), **kwargs)
