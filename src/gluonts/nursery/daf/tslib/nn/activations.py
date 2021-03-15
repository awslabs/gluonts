from typing import List
import math

import torch as pt
from torch import Tensor, nn


class GeLU(nn.Module):
    """
    Gaussian error Linear Unit
    y = 1/2 * x * (1 + tanh(\sqrt{2/pi} * (x + 0.044715*x^3)))
    """

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1
                + pt.tanh(
                    math.sqrt(2 / math.pi) * (x + 0.044715 * pt.pow(x, 3))
                )
            )
        )


class Swish(nn.Sigmoid):
    """
    Swish activation by google https://arxiv.org/pdf/1710.05941v1.pdf
    y = \sigma(x) * x
    """

    def forward(self, x: Tensor) -> Tensor:
        super(Swish, self).forward(x) * x


class PositiveSoftplus(nn.Softplus):
    """
    Softplus function that ensures a strictly positive activation

    Parameters
    ----------
    margin : float
        the minimum value of activation. when =0, same as vanilla softplus
    beta: float

    threshold: float

    """

    def __init__(
        self, margin: float = 0.0, beta: float = 1.0, threshold: float = 20.0
    ):
        super(PositiveSoftplus, self).__init__(beta, threshold)
        assert margin >= 0.0
        self.margin = margin

    def forward(self, x: Tensor) -> Tensor:
        return (
            super(PositiveSoftplus, self).forward(x - self.margin)
            + self.margin
        )


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit activation proposed by https://arxiv.org/pdf/1612.08083.pdf

    Parameters
    ----------
    dim : int
        the dimension to split gate and activation
    nonlinear: bool (default True)
        if True apply a tanh to activation before applying gate
    """

    def __init__(self, dim: int = -1, nonlinear: bool = True):
        super(GatedLinearUnit, self).__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: Tensor) -> Tensor:
        if x.size(self.dim) % 2 > 0:
            raise ValueError("The specified dimension must be even")
        val, gate = x.chunk(2, dim=self.dim)
        if self.nonlinear:
            val = pt.tanh(val)
        gate = pt.sigmoid(gate)
        return val * gate
