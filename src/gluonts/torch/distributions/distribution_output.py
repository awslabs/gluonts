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

from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Beta,
    Distribution,
    Gamma,
    Independent,
    LowRankMultivariateNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)

from gluonts.core.component import validated
from gluonts.torch.distributions import AffineTransformed
from gluonts.torch.modules.lambda_layer import LambdaLayer


class PtArgProj(nn.Module):
    r"""
    A PyTorch module that can be used to project from a dense layer
    to PyTorch distribution arguments.

    Parameters
    ----------
    in_features
        Size of the incoming features.
    dim_args
        Dictionary with string key and int value
        dimension of each arguments that will be passed to the domain
        map, the names are not used.
    domain_map
        Function returning a tuple containing one tensor
        a function or a nn.Module. This will be called with num_args
        arguments and should return a tuple of outputs that will be
        used when calling the distribution constructor.
    """

    def __init__(
        self,
        in_features: int,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[torch.Tensor]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class Output:
    """
    Class to connect a network to some output.
    """

    in_features: int
    args_dim: Optional[Dict[str, int]] = None
    _dtype: Type = np.float32

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Type):
        self._dtype = dtype

    def get_args_proj(self, in_features: int) -> nn.Module:
        return PtArgProj(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    def domain_map(self, *args: torch.Tensor):
        raise NotImplementedError()


class DistributionOutput(Output):
    r"""
    Class to construct a distribution given the output of a network.
    """

    distr_cls: type

    @validated()
    def __init__(self) -> None:
        pass

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
            return AffineTransformed(
                distr, loc=loc, scale=scale, event_dim=self.event_dim
            )

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions
        that this object constructs.
        """
        raise NotImplementedError()

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple,
        of the distributions that this object constructs.
        """
        return len(self.event_shape)

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the
        log-loss of the corresponding distribution. By default 0.0.
        This value will be used when padding data series.
        """
        return 0.0

    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends
        on the type of distribution, while the correct shape is obtained by
        reshaping the trailing axis in such a way that the returned tensors
        define a distribution of the right event_shape.
        """
        raise NotImplementedError()


class NormalOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = Normal

    @validated()
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distr_cls(*distr_args)
        else:
            return Independent(self.distr_cls(*distr_args), 1)

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        scale = F.softplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim == 1 else (self.dim,)


class StudentTOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distr_cls: type = StudentT

    @validated()
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distr_cls(*distr_args)
        else:
            return Independent(self.distr_cls(*distr_args), 1)

    @classmethod
    def domain_map(
        cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ):
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim == 1 else (self.dim,)


class BetaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"concentration1": 1, "concentration0": 1}
    distr_cls: type = Beta

    @validated()
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distr_cls(*distr_args)
        else:
            return Independent(self.distr_cls(*distr_args), 1)

    @classmethod
    def domain_map(
        cls, concentration1: torch.Tensor, concentration0: torch.Tensor
    ):
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon
        concentration1 = F.softplus(concentration1) + epsilon
        concentration0 = F.softplus(concentration0) + epsilon
        return concentration1.squeeze(dim=-1), concentration0.squeeze(dim=-1)

    @property
    def value_in_support(self) -> float:
        return 0.5

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim == 1 else (self.dim,)


class GammaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"concentration": 1, "rate": 1}
    distr_cls: type = Gamma

    @validated()
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distr_cls(*distr_args)
        else:
            return Independent(self.distr_cls(*distr_args), 1)

    @classmethod
    def domain_map(cls, concentration: torch.Tensor, rate: torch.Tensor):
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon
        concentration = F.softplus(concentration) + epsilon
        rate = F.softplus(rate) + epsilon
        return concentration.squeeze(dim=-1), rate.squeeze(dim=-1)

    @property
    def value_in_support(self) -> float:
        return 0.5

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim == 1 else (self.dim,)


class PoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1}
    distr_cls: type = Poisson

    @validated()
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distr_cls(*distr_args)
        else:
            return Independent(self.distr_cls(*distr_args), 1)

    @classmethod
    def domain_map(cls, rate: torch.Tensor):
        rate_pos = F.softplus(rate).clone()
        return (rate_pos.squeeze(-1),)

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim == 1 else (self.dim,)


class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    distr_cls: type = NegativeBinomial

    @validated()
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        total_count = F.softplus(total_count)
        return total_count.squeeze(-1), logits.squeeze(-1)

    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        if self.dim == 1:
            return self.distr_cls(total_count=total_count, logits=logits)
        else:
            return Independent(
                self.distr_cls(total_count=total_count, logits=logits), 1
            )

    # Overwrites the parent class method. We cannot scale using the affine
    # transformation since negative binomial should return integers. Instead
    # we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        return self._base_distribution((total_count, logits))

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim == 1 else (self.dim,)


class LowRankMultivariateNormalOutput(DistributionOutput):
    distr_cls: type = LowRankMultivariateNormal

    @validated()
    def __init__(
        self,
        dim: int,
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

        self.dim = dim
        self.rank = rank
        if rank == 0:
            self.args_dim = {"mu": dim, "D": dim}
        else:
            self.args_dim = {
                "mu": dim,
                "D": dim,
                "W": dim * rank,
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
        mu_vector
            Tensor of shape (*batch_shape, dim)
        D_vector
            Tensor of shape (*batch_shape, dim)
        W_vector
            Tensor of shape (*batch_shape, dim * rank)
        Returns
        -------
        Tuple
            A tuple containing tensors mu, D, and W, with shapes
            (*batch_shape, dim), (*batch_shape, dim),
            and (*batch_shape, dim, rank), respectively.
        """

        # Compute softplus^{-1}(sigma_init)
        D_bias = (
            self._inv_softplus(self.sigma_init) if self.sigma_init > 0 else 0
        )

        D_diag = F.softplus(D_vector + D_bias) + self.sigma_minimum

        if self.rank == 0:
            # Torch's built-in LowRankMultivariateNormal
            # doesn't support rank=0. So we pass a zero vector.
            W_matrix = torch.zeros(
                (*mu_vector[:-1], self.dim, 1),
                dtype=mu_vector.dtype,
                device=mu_vector.device,
                layout=mu_vector.layout,
            )
        else:
            assert (
                W_vector is not None
            ), "W_vector cannot be None if rank is not zero!"
            # reshape from vector form
            # (*batch_shape, dim * rank) to
            # matrix form (*batch_shape, dim, rank)
            W_matrix = W_vector.reshape(
                *W_vector.shape[:-1], self.dim, self.rank
            )

        return mu_vector, W_matrix, D_diag

    def _inv_softplus(self, y):
        if y < 20.0:
            # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
            return np.log(np.exp(y) - 1)
        else:
            return y

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
