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

# Standard library imports
import math
from typing import Dict, Tuple

# First-party imports
from gluonts.distribution.distribution import getF, softplus
from gluonts.model.common import Tensor

# Relative imports
from . import Kernel, KernelOutputDict


class RBFKernel(Kernel):
    r"""
    Computes a covariance matrix based on the RBF (squared exponential) kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    :math:`k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \theta_0 \exp \left
    ( -\frac{\|\mathbf{x_1} - \mathbf{x_2}\|^2}
    {2\theta_1^2} \right)`,
    where :math:`\theta_0` is the amplitude parameter and
    :math:`\theta_1` is the length scale parameter.
    """

    # noinspection PyMethodOverriding,PyPep8Naming
    def __init__(
        self, amplitude: Tensor, length_scale: Tensor, F=None
    ) -> None:
        """
        Parameters
        ----------
        amplitude : Tensor
            RBF kernel amplitude hyper-parameter of shape (batch_size, 1, 1).
        length_scale : Tensor
            RBF kernel length scale hyper-parameter of of shape (batch_size, 1, 1).
        F : ModuleType
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        """
        self.F = F if F else getF(amplitude)
        self.amplitude = amplitude
        self.length_scale = length_scale

    # noinspection PyMethodOverriding,PyPep8Naming
    def kernel_matrix(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Parameters
        --------------------
        x1 : Tensor
            Feature data of shape (batch_size, history_length, num_features).
        x2 : Tensor
            Feature data of shape (batch_size, history_length, num_features).

        Returns
        --------------------
        Tensor
            RBF kernel matrix of shape (batch_size, history_length, history_length).
        """
        self._compute_square_dist(self.F, x1, x2)

        return self.F.broadcast_mul(
            self.amplitude,
            self.F.exp(
                self.F.broadcast_div(
                    -self.square_dist, 2 * self.length_scale ** 2
                )
            ),
        )


class RBFKernelOutput(KernelOutputDict):
    args_dim: Dict[str, int] = {"amplitude": 1, "length_scale": 1}
    kernel_cls: type = RBFKernel

    # noinspection PyMethodOverriding,PyPep8Naming
    def gp_params_scaling(
        self, F, past_target: Tensor, past_time_feat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        This function returns the scales for the GP RBF Kernel hyper-parameters by using the standard deviations
        of the past_target and past_time_features.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        past_target
            Training time series values of shape (batch_size, context_length).
        past_time_feat
            Training features of shape (batch_size, context_length, num_features).

        Returns
        -------
        Tuple
            Two scaled GP hyper-parameters for the RBF Kernel and scaled model noise hyper-parameter.
            Each is a Tensor of shape (batch_size, 1, 1).
        """
        axis = 1
        sigma_scaling = (
            self.compute_std(F, past_target, axis=axis) / math.sqrt(2)
        ).expand_dims(axis=axis)
        amplitude_scaling = sigma_scaling ** 2
        length_scale_scaling = F.broadcast_mul(
            F.mean(self.compute_std(F, past_time_feat, axis=axis)),
            F.ones_like(amplitude_scaling),
        )
        return amplitude_scaling, length_scale_scaling, sigma_scaling

    # noinspection PyMethodOverriding,PyPep8Naming
    @classmethod
    def domain_map(
        cls, F, amplitude: Tensor, length_scale: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        This function applies the softmax to the RBF Kernel hyper-parameters.

        Parameters
        ----------
        F: mx.symbol or mx.nd
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        amplitude
            RBF kernel amplitude hyper-parameter of shape (batch_size, 1, 1).
        length_scale
            RBF kernel length scale hyper-parameter of of shape (batch_size, 1, 1).

        Returns
        -------
        Tuple
            Two GP RBF kernel hyper-parameters.
            Each is a Tensor of shape: (batch_size, 1, 1).
        """
        amplitude = softplus(F, amplitude)
        length_scale = softplus(F, length_scale)
        return amplitude, length_scale
