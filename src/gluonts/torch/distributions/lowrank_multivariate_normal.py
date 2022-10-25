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

from typing import Optional, Tuple

import numpy as np
import torch

from gluonts.core.component import validated

from gluonts.torch.modules.distribution_output import (
    Distribution,
    DistributionOutput,
)

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
    LowRankMultivariateNormal,
)


def inv_softplus(y):
    if y < 20.0:
        # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
        return np.log(np.exp(y) - 1)
    else:
        return y


class LowRankMultivariateNormalOutput(DistributionOutput):
    distr_cls: type = LowRankMultivariateNormal

    @validated()
    def __init__(
        self,
        target_dim: int,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-4,
    ) -> None:
        super().__init__(self)

        assert (
            isinstance(rank, int) and rank >= 0
        ), "rank should be a nonnegative integer"

        assert (
            sigma_init >= 0
        ), "sigma_init should be greater than or equal to 0"

        assert sigma_minimum > 0, "sigma_minimum should be greater than 0"

        self.target_dim = target_dim
        self.rank = rank
        if rank == 0:
            self.args_dim = {"mu": target_dim, "D": target_dim}
        else:
            self.args_dim = {
                "mu": target_dim,
                "D": target_dim,
                "W": target_dim * rank,
            }
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum

    def domain_map(
        self,
        mu_vector: torch.Tensor,
        D_vector: torch.Tensor,
        W_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        r"""

        Parameters
        ----------
        F
        mu_vector
            Tensor of shape (*batch_shape, target_dim)
        D_vector
            Tensor of shape (*batch_shape, target_dim)
        W_vector
            Tensor of shape (*batch_shape, target_dim * rank)

        Returns
        -------
        Tuple
            A tuple containing tensors mu, D, and W, with shapes
            (*batch_shape, target_dim), (*batch_shape, target_dim),
            and (*batch_shape, target_dim, rank), respectively.

        """

        softplus = torch.nn.Softplus()

        # Compute softplus^{-1}(sigma_init)
        D_bias = inv_softplus(self.sigma_init) if self.sigma_init > 0 else 0

        D_diag = softplus(D_vector + D_bias) + self.sigma_minimum

        if self.rank == 0:
            # Torch's bulit-in LowRankMultivariateNormal
            # doesn't support rank=0. So we pass a zero vector.
            W_matrix = torch.zeros(
                (*mu_vector[:-1], self.target_dim, 1),
                dtype=mu_vector.dtype,
                device=mu_vector.device,
                layout=mu_vector.layout,
            )
        else:
            assert (
                W_vector is not None
            ), "W_vector cannot be None if rank is not zero!"
            # reshape from vector form
            # (*batch_shape, target_dim * rank) to
            # matrix form (*batch_shape, target_dim, rank)
            W_matrix = W_vector.reshape(
                *W_vector.shape[:-1], self.target_dim, self.rank
            )

        return mu_vector, W_matrix, D_diag

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:

        distr = self.distr_cls(*distr_args)
        if loc == 0 and scale is None:
            return distr
        else:
            return TransformedDistribution(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return (self.target_dim,)
