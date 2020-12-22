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

from typing import Dict, Tuple

import numpy as np
from mxnet import gluon

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution_output import ArgProj

from . import Kernel


class KernelOutput:
    """
    Class to connect a network to a kernel.
    """

    def get_args_proj(self, float_type: DType) -> gluon.HybridBlock:
        raise NotImplementedError()

    def kernel(self, args) -> Kernel:
        raise NotImplementedError()

    # noinspection PyMethodOverriding,PyPep8Naming
    @staticmethod
    def compute_std(F, data: Tensor, axis: int) -> Tensor:
        """
        This function computes the standard deviation of the data along a given
        axis.

        Parameters
        ----------
        F : ModuleType
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        data : Tensor
            Data to be used to compute the standard deviation.
        axis : int
            Axis along which to compute the standard deviation.

        Returns
        -------
        Tensor
            The standard deviation of the given data.
        """
        return F.sqrt(
            F.mean(
                F.broadcast_minus(
                    data, F.mean(data, axis=axis).expand_dims(axis=axis)
                )
                ** 2,
                axis=axis,
            )
        )


class KernelOutputDict(KernelOutput):
    args_dim: Dict[str, int]
    kernel_cls: type

    @validated()
    def __init__(self) -> None:
        pass

    def get_num_args(self) -> int:
        return len(self.args_dim)

    def get_args_proj(self, float_type: DType = np.float32) -> ArgProj:
        """
        This method calls the ArgProj block in distribution_output to project
        from a dense layer to kernel arguments.

        Parameters
        ----------
        float_type : DType
            Determines whether to use single or double precision.
        Returns
        -------
        ArgProj
        """
        return ArgProj(
            args_dim=self.args_dim,
            domain_map=gluon.nn.HybridLambda(self.domain_map),
            dtype=float_type,
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def gp_params_scaling(
        self, F, past_target: Tensor, past_time_feat: Tensor
    ) -> tuple:
        raise NotImplementedError()

    # noinspection PyMethodOverriding,PyPep8Naming
    def domain_map(self, F, *args: Tensor):
        raise NotImplementedError()

    def kernel(self, kernel_args) -> Kernel:
        """
        Parameters
        ----------
        kernel_args
            Variable length argument list.

        Returns
        -------
        gluonts.mx.kernels.Kernel
            Instantiated specified Kernel subclass object.
        """
        return self.kernel_cls(*kernel_args)
