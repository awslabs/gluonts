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

from typing import List, Optional

from mxnet.gluon import nn
from mxnet.gluon.loss import Loss

from gluonts.core.component import validated
from gluonts.mx import Tensor


def uniform_weights(objects: list) -> List[float]:
    """
    Return uniform weights for a list of objects.

    >>> uniform_weights(["a", "b", "c", "d"])
    [0.25, 0.25, 0.25, 0.25]

    Parameters
    ----------
    objects
        Objects that need to be weighted.

    Returns
    -------
    List[float]
        List of weights.
    """
    return [1.0 / len(objects)] * len(objects)


def crps_weights_pwl(quantile_levels: List[float]) -> List[float]:
    """
    Compute the quantile loss weights making mean quantile loss equal to CRPS
    under linear interpolation assumption.

    Quantile levels are assumed to be sorted in increasing order.

    Under the assumption of linear interpolation

    .. math:: CRPS = sum_{i=0}^{n-1} 0.5 * (q_{i+1}-q_{i}) * (z_{i+1}+z_{i})

    where :math:`z_i` is the i-th quantile prediction :math:`q_i`.
    The inner terms cancel due to the telescoping sum property and we obtain

    .. math:: CRPS = sum_{i=1}^n w_i z_i

    with the weights :math:`w_i = (q_{i+1}-q_{i-1})/2` for
    :math:`i = 1, ..., n-1`, :math:`w_0 = (q_1-q_0)/2` and
    :math:`w_n = (w_n - w_{n-1})/2`.
    """
    num_quantiles = len(quantile_levels)

    if num_quantiles < 2:
        return [1.0] * num_quantiles

    return (
        [0.5 * (quantile_levels[1] - quantile_levels[0])]
        + [
            0.5 * (quantile_levels[i + 1] - quantile_levels[i - 1])
            for i in range(1, num_quantiles - 1)
        ]
        + [0.5 * (quantile_levels[-1] - quantile_levels[-2])]
    )


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
        Represent the quantile loss used to fit decoders that learn quantiles.

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
        assert len(quantiles) > 0

        super().__init__(weight, batch_axis, **kwargs)

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = (
            quantile_weights
            if quantile_weights is not None
            else uniform_weights(quantiles)
        )

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
            ground truth values, shape (N1 x N2 x ... x Nk)
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
        for level, weight, y_pred_q in zip(
            self.quantiles, self.quantile_weights, y_pred_all
        ):
            qt_loss.append(
                weight * self.compute_quantile_loss(F, y_true, y_pred_q, level)
            )
        stacked_qt_losses = F.stack(*qt_loss, axis=-1)
        sum_qt_loss = F.mean(
            stacked_qt_losses, axis=-1
        )  # avg across quantiles
        if sample_weight is not None:
            return sample_weight * sum_qt_loss
        return sum_qt_loss

    @staticmethod
    def compute_quantile_loss(
        F, y_true: Tensor, y_pred_p: Tensor, p: float
    ) -> Tensor:
        """
        Compute the quantile loss of the given quantile.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        y_true
            ground truth values to compute the loss against.
        y_pred_p
            predicted target quantile, same shape as ``y_true``.
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
        self._quantiles = quantiles
        self.num_quantiles = len(self._quantiles)
        self.quantile_weights = quantile_weights

    @property
    def quantiles(self) -> List[float]:
        return self._quantiles

    def get_loss(self) -> nn.HybridBlock:
        """
        Returns
        -------
        nn.HybridBlock
            constructs quantile loss object.
        """
        return QuantileLoss(
            quantiles=self.quantiles,
            quantile_weights=(
                self.quantile_weights
                if self.quantile_weights is not None
                else uniform_weights(self.quantiles)
            ),
        )

    def get_quantile_proj(self, **kwargs) -> nn.HybridBlock:
        return nn.Dense(units=self.num_quantiles, flatten=False)


class IncrementalDenseLayerProjection(nn.HybridBlock):
    """
    A dense layer that outputs non-decreasing values.

    Parameters
    ----------
    num_outputs
        number of outputs of the layer.
    """

    @validated()
    def __init__(self, num_outputs: int, **kwargs):
        super().__init__(**kwargs)

        self.num_outputs = num_outputs
        with self.name_scope():
            self.proj_intrcpt = nn.Dense(1, flatten=False)
            if self.num_outputs > 1:
                self.proj_incrmnt = nn.Dense(
                    self.num_outputs - 1,
                    flatten=False,
                    activation="relu",
                )  # increments between quantile estimates

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        return (
            self.proj_intrcpt(x)
            if self.num_outputs == 1
            else (
                F.cumsum(
                    F.concat(
                        self.proj_intrcpt(x), self.proj_incrmnt(x), dim=-1
                    ),
                    axis=3,
                )
            )
        )


class IncrementalQuantileOutput(QuantileOutput):
    """
    Output layer using a quantile loss and projection layer to connect the
    quantile output to the network.

    Differently from ``QuantileOutput``, this class enforces the correct
    order relation between quantiles: this is done by parametrizing
    the increments between quantiles instead of the quantiles directly.

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
        super().__init__(sorted(quantiles), quantile_weights)

    def get_loss(self) -> nn.HybridBlock:
        """
        Returns
        -------
        nn.HybridBlock
            constructs quantile loss object.
        """
        return QuantileLoss(
            quantiles=self.quantiles,
            quantile_weights=(
                self.quantile_weights
                if self.quantile_weights is not None
                else crps_weights_pwl(self.quantiles)
            ),
        )

    def get_quantile_proj(self, **kwargs) -> nn.HybridBlock:
        """
        Returns
        -------
        nn.HybridBlock
            constructs projection parameter object.

        """
        return IncrementalDenseLayerProjection(self.num_quantiles, **kwargs)
