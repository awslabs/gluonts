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

import math
from typing import Dict, Tuple

from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution import getF, softplus

from . import Kernel, KernelOutputDict


class PeriodicKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Periodic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    :math:`k_{\text{Per}}(\mathbf{x_1}, \mathbf{x_2}) = \theta_0 \exp \left
    (\frac{-2\sin^2(\theta_2 \pi \|\mathbf{x_1} - \mathbf{x_2}\|)}
    {\theta_1^2} \right)`,
    where :math:`\theta_0` is the amplitude parameter,
    :math:`\theta_1` is the length scale parameter and
    :math:`\theta_2` is the frequency parameter.
    """

    # noinspection PyMethodOverriding,PyPep8Naming
    def __init__(
        self,
        amplitude: Tensor,
        length_scale: Tensor,
        frequency: Tensor,
        F=None,
    ) -> None:
        """
        Parameters
        ----------
        amplitude : Tensor
            Periodic kernel amplitude hyper-parameter of shape (batch_size, 1, 1).
        length_scale : Tensor
            Periodic kernel length scale hyper-parameter of of shape (batch_size, 1, 1).
        frequency : Tensor
            Periodic kernel hyper-parameter of shape (batch_size, 1, 1).
        F : ModuleType
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        """
        self.F = F if F else getF(amplitude)
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.frequency = frequency

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
            Periodic kernel matrix of shape (batch_size, history_length, history_length).
        """
        self._compute_square_dist(self.F, x1, x2)

        return self.F.broadcast_mul(
            self.amplitude,
            self.F.exp(
                self.F.broadcast_div(
                    -2
                    * self.F.sin(
                        self.F.broadcast_mul(
                            self.frequency,
                            math.pi
                            * self.F.sqrt(self.F.abs(self.square_dist)),
                        )
                    )
                    ** 2,
                    self.length_scale ** 2,
                )
            ),
        )


class PeriodicKernelOutput(KernelOutputDict):
    args_dim: Dict[str, int] = {
        "amplitude": 1,
        "length_scale": 1,
        "frequency": 1,
    }
    kernel_cls: type = PeriodicKernel

    # noinspection PyMethodOverriding,PyPep8Naming
    def gp_params_scaling(
        self, F, past_target: Tensor, past_time_feat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This function returns the scales for the GP Periodic Kernel hyper-parameters by using the standard deviations
        of the past_target and past_time_features.

        Parameters
        ----------
        F : ModuleType
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        past_target : Tensor
            Training time series values of shape (batch_size, context_length).
        past_time_feat : Tensor
            Training features of shape (batch_size, context_length, num_features).

        Returns
        -------
        Tuple
            Three scaled GP hyper-parameters for the Periodic Kernel and scaled model noise hyper-parameter.
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
        # TODO: Define scaling for the frequency
        frequency_scaling = F.ones_like(amplitude_scaling)
        return (
            amplitude_scaling,
            length_scale_scaling,
            frequency_scaling,
            sigma_scaling,
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    @classmethod
    def domain_map(cls, F, amplitude, length_scale, frequency):
        r"""
        This function applies the softmax to the Periodic Kernel hyper-parameters.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        amplitude
            Periodic kernel amplitude hyper-parameter of shape (batch_size, 1, 1).
        length_scale
            Periodic kernel length scale hyper-parameter of of shape (batch_size, 1, 1).
        frequency
            Periodic kernel hyper-parameter of shape (batch_size, 1, 1).

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            Three GP Periodic kernel hyper-parameters.
            Each is a Tensor of shape: (batch_size, 1, 1).
        """
        amplitude = softplus(F, amplitude)
        length_scale = softplus(F, length_scale)
        frequency = softplus(F, frequency)
        return amplitude, length_scale, frequency
