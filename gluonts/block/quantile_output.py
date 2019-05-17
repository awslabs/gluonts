# Standard library imports
from typing import List, Optional, Tuple

# Third-party imports
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class QuantileLoss(Loss):
    @validated()
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: List[float] = None,
        weight=None,
        batch_axis=0,
        **kwargs,
    ) -> None:
        super().__init__(weight, batch_axis, **kwargs)

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = (
            nd.ones(self.num_quantiles) / self.num_quantiles
            if not quantile_weights
            else quantile_weights
        )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, y_true, y_pred, sample_weight=None):
        '''
        Compute the weighted sum of quantile losses

        :param y_true: shape N1 x N2 x ... x Nk x dimension of time series (normally 1)
        :param y_pred: shape N1 x N2 x ... x Nk x num_quantiles
        :return: weighted sum of the quantile losses, shape N1 x N1 x ... Nk
        '''
        y_pred_all = F.split(
            y_pred, axis=-1, num_outputs=self.num_quantiles, squeeze_axis=1
        )

        qt_loss = []
        for i, y_pred_q in enumerate(y_pred_all):
            q = self.quantiles[i]
            weighted_qt = (
                self.compute_quantile_loss(F, y_true, y_pred_q, q)
                * self.quantile_weights[i].asscalar()
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
    def compute_quantile_loss(F, y_true, y_pred_p, p):
        """
        Compute the quantile loss

        :param y_true:
        :param y_pred_p:
        :param p:
        :return:
        """

        under_bias = p * F.maximum(y_true - y_pred_p, 0)
        over_bias = (1 - p) * F.maximum(y_pred_p - y_true, 0)

        qt_loss = 2 * (under_bias + over_bias)

        return qt_loss


class ProjectParams(nn.HybridBlock):
    @validated()
    def __init__(self, num_quantiles, **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.projection = nn.Dense(units=num_quantiles, flatten=False)

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.projection(x)


class QuantileOutput:
    @validated()
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
    ) -> None:
        self.quantiles = quantiles
        self.quantile_weights = quantile_weights

    def get_loss(self) -> nn.HybridBlock:
        return QuantileLoss(
            quantiles=self.quantiles, quantile_weights=self.quantile_weights
        )

    def get_quantile_proj(self, **kwargs) -> nn.HybridBlock:
        return ProjectParams(len(self.quantiles), **kwargs)
