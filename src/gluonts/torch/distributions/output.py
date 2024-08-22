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

from gluonts.model.forecast_generator import ForecastGenerator
from gluonts.torch.modules.lambda_layer import LambdaLayer


class PtArgProj(nn.Module):
    r"""
    A PyTorch module that can be used to project from a dense layer to PyTorch
    distribution arguments.

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
    Converts raw neural network output into a forecast and computes loss.
    """

    in_features: int
    args_dim: Dict[str, int]
    _dtype: Type = np.float32

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Type):
        self._dtype = dtype

    def loss(
        self,
        target: torch.Tensor,
        distr_args: Tuple[torch.Tensor, ...],
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for target data given network output.

        Parameters
        ----------
        target
            Values of the target time series for which loss is to be computed.
        distr_args
            Arguments that can be used to construct the output distribution.
        loc
            Location parameter of the distribution, optional.
        scale
            Scale parameter of the distribution, optional.

        Returns
        -------
        loss_values
            Values of the loss, has same shape as target.
        """
        raise NotImplementedError()

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event compatible with the output object.
        """
        raise NotImplementedError()

    @property
    def forecast_generator(self) -> ForecastGenerator:
        raise NotImplementedError()

    def get_args_proj(self, in_features: int) -> nn.Module:
        return PtArgProj(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    def domain_map(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        r"""
        Converts arguments to the right shape and domain.

        The domain depends on the type of distribution, while the correct shape
        is obtained by reshaping the trailing axis in such a way that the
        returned tensors define a distribution of the right event_shape.
        """
        raise NotImplementedError()

    @property
    def value_in_support(self) -> float:
        r"""
        A float value that is valid for computing the loss of the corresponding
        output.

        By default 0.0.
        """
        return 0.0
