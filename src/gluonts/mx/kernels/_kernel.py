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

from gluonts.mx import Tensor


class Kernel:
    # noinspection PyMethodOverriding,PyPep8Naming
    def kernel_matrix(self, x1: Tensor, x2: Tensor):
        # raise error in the base Kernel class, implement in the concrete subclasses
        raise NotImplementedError()

    # noinspection PyMethodOverriding,PyPep8Naming
    def _compute_square_dist(self, F, x1: Tensor, x2: Tensor) -> None:
        r"""
        Parameters
        --------------------
        F : ModuleType
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        x1 : Tensor
            Feature data of shape (batch_size, history_length, num_features).
        x2 : Tensor
            Feature data of shape (batch_size, history_length, num_features).

        Returns
        --------------------
        Tensor
            square distance matrix of shape (batch_size, history_length, history_length)
            :math: `\|\mathbf{x_1}-\mathbf{x_2}\|_2^2 = (\mathbf{x_1}-\mathbf{x_2})^T(\mathbf{x_1}-\mathbf{x_2})
                                                        = \|\mathbf{x_1}\|_2^2 - 2\mathbf{x_1}^T\mathbf{x_2}
                                                        + \|\mathbf{x_2}\|_2^2`.
        """
        feature_axis = 2
        # Column vector: Add to math:`x_i^Tx_i` to every column in row i
        x1_norm_square = (
            F.norm(x1, ord=2, axis=feature_axis) ** 2
        ).expand_dims(2)
        # Row vector: Add to math:`x_i^Tx_i` to every row in column i
        x2_norm_square = (
            F.norm(x2, ord=2, axis=feature_axis) ** 2
        ).expand_dims(1)
        x1x2_trans = F.linalg.gemm2(x1, x2, transpose_b=True)
        self.square_dist = F.broadcast_add(
            F.broadcast_sub(x1_norm_square, 2 * x1x2_trans), x2_norm_square
        )
