from typing import List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from gluonts.core.component import validated


class QuantileLoss(nn.Module):
    @validated()
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = (
            [1.0 / self.num_quantiles for i in range(self.num_quantiles)]
            if not quantile_weights
            else quantile_weights
        )

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight=None):
        if self.num_quantiles > 1:
            y_pred_all = torch.chunk(y_pred, self.num_quantiles, dim=-1)
        else:
            y_pred_all = [y_pred]

        qt_loss = []
        for i, y_pred_q in enumerate(y_pred_all):
            q = self.quantiles[i]
            weighted_qt = (
                self.compute_quantile_loss(y_true, y_pred_q.squeeze(-1), q)
                * self.quantile_weights[i]
            )
            qt_loss.append(weighted_qt)
        stacked_qt_losses = torch.stack(qt_loss, dim=-1)
        sum_qt_loss = torch.mean(stacked_qt_losses, dim=-1)
        if sample_weight is not None:
            return sample_weight * sum
        else:
            return sum_qt_loss

    @staticmethod
    def compute_quantile_loss(
        y_true: torch.Tensor, y_pred_p: torch.Tensor, p: float
    ) -> torch.Tensor:
        under_bias = p * torch.clamp(y_true - y_pred_p, min=0)
        over_bias = (1 - p) * torch.clamp(y_pred_p - y_true, min=0)

        qt_loss = 2 * (under_bias + over_bias)
        return qt_loss


class ProjectParams(nn.Module):
    @validated()
    def __init__(self, in_features, num_quantiles):
        super().__init__()
        self.projection = nn.Linear(in_features=in_features, out_features=num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class QuantileOutput:
    @validated()
    def __init__(
        self,
        input_size,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
    ) -> None:
        self.input_size = input_size
        self.quantiles = quantiles
        self.quantile_weights = quantile_weights

    def get_loss(self) -> nn.Module:
        return QuantileLoss(
            quantiles=self.quantiles, quantile_weights=self.quantile_weights
        )

    def get_quantile_proj(self) -> nn.Module:
        return ProjectParams(
            in_features=self.input_size, num_quantiles=len(self.quantiles)
        )
