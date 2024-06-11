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

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import (
    Beta,
    Distribution,
    Gamma,
    Laplace,
    Normal,
    Poisson,
)

from gluonts.core.component import validated
from gluonts.model.forecast_generator import (
    DistributionForecastGenerator,
    ForecastGenerator,
)
from gluonts.torch.distributions import AffineTransformed

from .output import Output


class DistributionOutput(Output):
    r"""
    Class to construct a distribution given the output of a network.
    """

    distr_cls: type

    @validated()
    def __init__(self, beta: float = 0.0) -> None:
        self.beta = beta

    def _base_distribution(self, distr_args):
        return self.distr_cls(*distr_args)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        r"""
        Construct the associated distribution, given the collection of
        constructor arguments and, optionally, a scale tensor.

        Parameters
        ----------
        distr_args
            Constructor arguments for the underlying Distribution type.
        loc
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        scale
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        """
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformed(distr, loc=loc, scale=scale)

    def loss(
        self,
        target: torch.Tensor,
        distr_args: Tuple[torch.Tensor, ...],
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distribution = self.distribution(distr_args, loc=loc, scale=scale)
        nll = -distribution.log_prob(target)
        if self.beta > 0.0:
            variance = distribution.variance
            nll = nll * (variance.detach() ** self.beta)
        return nll

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of
        the distributions that this object constructs.
        """
        return len(self.event_shape)

    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain.

        The domain depends on the type of distribution, while the correct shape
        is obtained by reshaping the trailing axis in such a way that the
        returned tensors define a distribution of the right event_shape.
        """
        raise NotImplementedError()

    @property
    def forecast_generator(self) -> ForecastGenerator:
        return DistributionForecastGenerator(self)


class NormalOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = Normal

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):  # type: ignore
        scale = F.softplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()


class LaplaceOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = Laplace

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):  # type: ignore
        scale = F.softplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()


class BetaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"concentration1": 1, "concentration0": 1}
    distr_cls: type = Beta

    @classmethod
    def domain_map(  # type: ignore
        cls, concentration1: torch.Tensor, concentration0: torch.Tensor
    ):
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon
        concentration1 = F.softplus(concentration1) + epsilon
        concentration0 = F.softplus(concentration0) + epsilon
        return concentration1.squeeze(dim=-1), concentration0.squeeze(dim=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5


class GammaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"concentration": 1, "rate": 1}
    distr_cls: type = Gamma

    @classmethod
    def domain_map(cls, concentration: torch.Tensor, rate: torch.Tensor):  # type: ignore
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon
        concentration = F.softplus(concentration) + epsilon
        rate = F.softplus(rate) + epsilon
        return concentration.squeeze(dim=-1), rate.squeeze(dim=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5


class PoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1}
    distr_cls: type = Poisson

    @classmethod
    def domain_map(cls, rate: torch.Tensor):  # type: ignore
        rate_pos = F.softplus(rate).clone()
        return (rate_pos.squeeze(-1),)

    # Overwrites the parent class method. We cannot scale using the affine
    # transformation since Poisson should return integers. Instead we scale
    # the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        (rate,) = distr_args

        if scale is not None:
            rate *= scale

        return Poisson(rate=rate)

    @property
    def event_shape(self) -> Tuple:
        return ()
