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
from typing import Optional

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.model.common import Tensor


def batch_diagonal(
    F,
    matrix: Tensor,
    num_data_points: Optional[int] = None,
    ctx=mx.cpu(),
    float_type=np.float32,
) -> Tensor:
    """
    This function extracts the diagonal of a batch matrix.

    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet.
    matrix
        matrix of shape (batch_size, num_data_points, num_data_points).
    num_data_points
        Number of rows in the kernel_matrix.

    Returns
    -------
    Tensor
        Diagonals of kernel_matrix of shape (batch_size, num_data_points, 1).

    """
    return F.linalg.gemm2(
        F.broadcast_mul(
            F.eye(num_data_points, ctx=ctx, dtype=float_type), matrix
        ),
        F.ones_like(F.slice_axis(matrix, axis=2, begin=0, end=1)),
    )


# noinspection PyMethodOverriding,PyPep8Naming
def jitter_cholesky_eig(
    F,
    matrix: Tensor,
    num_data_points: Optional[int] = None,
    ctx: mx.Context = mx.Context('cpu'),
    float_type: DType = np.float64,
    diag_weight: float = 1e-6,
) -> Tensor:
    """
    This function applies the jitter method using the eigenvalue decomposition.
    The eigenvalues are bound below by the jitter, which is proportional to the mean of the
    diagonal elements

    Parameters
    ----------
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet.
    matrix
        Matrix of shape (batch_size, num_data_points, num_data_points).
    num_data_points
        Number of rows in the kernel_matrix.
    ctx
        Determines whether to compute on the cpu or gpu.
    float_type
        Determines whether to use single or double precision.

    Returns
    -------
    Tensor
        Returns the approximate lower triangular Cholesky factor `L`
        of shape (batch_size, num_data_points, num_data_points)
    """
    diag = batch_diagonal(
        F, matrix, num_data_points, ctx, float_type
    )  # shape (batch_size, num_data_points, 1)
    diag_mean = diag.mean(axis=1).expand_dims(
        axis=2
    )  # shape (batch_size, 1, 1)
    U, Lambda = F.linalg.syevd(matrix)
    jitter = F.broadcast_mul(diag_mean, F.ones_like(diag)) * diag_weight
    # `K = U^TLambdaU`, where the rows of `U` are the eigenvectors of `K`.
    # The eigendecomposition :math:`U^TLambdaU` is used instead of :math: ULambdaU^T, sine
    # to utilize row-based computation (see Section 4, Seeger et al., 2018)
    return F.linalg.potrf(
        F.linalg.gemm2(
            U,
            F.linalg.gemm2(
                F.broadcast_mul(
                    F.eye(num_data_points, ctx=ctx, dtype=float_type),
                    F.maximum(jitter, Lambda.expand_dims(axis=2)),
                ),
                U,
            ),
            transpose_a=True,
        )
    )


# noinspection PyMethodOverriding,PyPep8Naming
def jitter_cholesky(
    F,
    matrix: Tensor,
    num_data_points: Optional[int] = None,
    ctx: mx.Context = mx.Context('cpu'),
    float_type: DType = np.float64,
    max_iter_jitter: int = 10,
    neg_tol: float = -1e-8,
    diag_weight: float = 1e-6,
    increase_jitter: int = 10,
) -> Optional[Tensor]:
    """
    This function applies the jitter method.  It iteratively tries to compute the Cholesky decomposition and
    adds a positive tolerance to the diagonal that increases at each iteration until the matrix is positive definite
    or the maximum number of iterations has been reached.

    Parameters
    ----------
    matrix
        Kernel matrix of shape (batch_size, num_data_points, num_data_points).
    num_data_points
        Number of rows in the kernel_matrix.
    ctx
        Determines whether to compute on the cpu or gpu.
    float_type
        Determines whether to use single or double precision.
    max_iter_jitter
        Maximum number of iterations for jitter to iteratively make the matrix positive definite.
    neg_tol
        Parameter in the jitter methods to eliminate eliminate matrices with diagonal elements smaller than this
        when checking if a matrix is positive definite.
    diag_weight
            Multiple of mean of diagonal entries to initialize the jitter.
    increase_jitter
        Each iteration multiply by jitter by this amount
    Returns
    -------
    Optional[Tensor]
        The method either fails to make the matrix positive definite within the maximum number of iterations
        and outputs an error or succeeds and returns the lower triangular Cholesky factor `L`
        of shape (batch_size, num_data_points, num_data_points)
    """
    num_iter = 0
    diag = batch_diagonal(
        F, matrix, num_data_points, ctx, float_type
    )  # shape (batch_size, num_data_points, 1)
    diag_mean = diag.mean(axis=1).expand_dims(
        axis=2
    )  # shape (batch_size, 1, 1)
    jitter = F.zeros_like(diag)  # shape (batch_size, num_data_points, 1)
    # Ensure that diagonal entries are numerically non-negative, as defined by neg_tol
    # TODO: Add support for symbolic case: Cannot use < operator with symbolic variables
    if F.sum(diag <= neg_tol) > 0:
        raise mx.base.MXNetError(
            ' Matrix is not positive definite: negative diagonal elements'
        )
    while num_iter <= max_iter_jitter:
        try:
            L = F.linalg.potrf(
                F.broadcast_add(
                    matrix,
                    F.broadcast_mul(
                        F.eye(num_data_points, ctx=ctx, dtype=float_type),
                        jitter,
                    ),
                )
            )
            # gpu will not throw error but will store nans. If nan, L.sum() = nan
            # so the error tolerance can be large.
            # TODO: Add support for symbolic case: Cannot use <= operator with symbolic variables
            assert F.max(F.abs(L.nansum() - L.sum()) <= 1e-1)
            return L
        except:
            if num_iter == 0:
                # Initialize the jitter: constant jitter per each batch
                jitter = (
                    F.broadcast_mul(diag_mean, F.ones_like(jitter))
                    * diag_weight
                )
            else:
                jitter = jitter * increase_jitter
        finally:
            num_iter += 1
    raise mx.base.MXNetError(
        f' Matrix is not positive definite after the maximum number of iterations = {max_iter_jitter} '
        f'with a maximum jitter = {F.max(jitter)}'
    )
