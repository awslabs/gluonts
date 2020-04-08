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
import sys

# First-party imports
from gluonts.core.component import check_gpu_support
from gluonts.kernels import RBFKernel
from gluonts.model.gp_forecaster.gaussian_process import GaussianProcess
from gluonts.support.linalg_util import jitter_cholesky, jitter_cholesky_eig

# Third-party imports
import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import pytest


# This test verifies that both eigenvalue decomposition and iterative jitter method
# make a non-positive definite matrix positive definite to be able to compute the cholesky.
# Both gpu and cpu as well as single and double precision are tested.
@pytest.mark.skipif(
    sys.platform == "linux",
    reason=f"skipping since potrf crashes on mxnet 1.6.0 on linux when matrix is not spd",
)
@pytest.mark.parametrize("ctx", [mx.Context("gpu"), mx.Context("cpu")])
@pytest.mark.parametrize("jitter_method", ["iter", "eig"])
@pytest.mark.parametrize("float_type", [np.float32, np.float64])
def test_jitter_unit(jitter_method, float_type, ctx) -> None:
    # TODO: Enable GPU tests on Jenkins
    if ctx == mx.Context("gpu") and not check_gpu_support():
        return
    matrix = nd.array(
        [[[1, 2], [3, 4]], [[10, 100], [-21.5, 41]]], ctx=ctx, dtype=float_type
    )
    F = mx.nd
    num_data_points = matrix.shape[1]
    if jitter_method == "eig":
        L = jitter_cholesky_eig(F, matrix, num_data_points, ctx, float_type)
    elif jitter_method == "iter":
        L = jitter_cholesky(F, matrix, num_data_points, ctx, float_type)
    assert np.sum(np.isnan(L.asnumpy())) == 0, "NaNs in Cholesky factor!"


# This test tests that the noiseless sample generated from the synthetic example does not have NaNs in it.
# Without the jitter method, NaNs occurs on the gpu for single and double precision and on the cpu for only single
# precision.  This test verifies that applying the default jitter method fixes these numerical issues on both cpu
# and gpu and for single and double precision.
@pytest.mark.skipif(
    sys.platform == "linux",
    reason=f"skipping since potrf crashes on mxnet 1.6.0 on linux when matrix is not spd",
)
@pytest.mark.parametrize("ctx", [mx.Context("gpu"), mx.Context("cpu")])
@pytest.mark.parametrize("jitter_method", ["iter", "eig"])
@pytest.mark.parametrize("float_type", [np.float32, np.float64])
def test_jitter_synthetic_gp(jitter_method, float_type, ctx) -> None:
    # TODO: Enable GPU tests on Jenkins
    if ctx == mx.Context("gpu") and not check_gpu_support():
        return
    # Initialize problem parameters
    batch_size = 1
    prediction_length = 50
    context_length = 5
    num_samples = 3

    # Initialize test data to generate Gaussian Process from
    lb = -5
    ub = 5
    dx = (ub - lb) / (prediction_length - 1)
    x_test = nd.arange(lb, ub + dx, dx, ctx=ctx, dtype=float_type).reshape(
        -1, 1
    )
    x_test = nd.tile(x_test, reps=(batch_size, 1, 1))

    # Define the GP hyper parameters
    amplitude = nd.ones((batch_size, 1, 1), ctx=ctx, dtype=float_type)
    length_scale = math.sqrt(0.4) * nd.ones_like(amplitude)
    sigma = math.sqrt(1e-5) * nd.ones_like(amplitude)

    # Instantiate desired kernel object and compute kernel matrix
    rbf_kernel = RBFKernel(amplitude, length_scale)

    # Generate samples from 0 mean Gaussian process with RBF Kernel and plot it
    gp = GaussianProcess(
        sigma=sigma,
        kernel=rbf_kernel,
        prediction_length=prediction_length,
        context_length=context_length,
        num_samples=num_samples,
        ctx=ctx,
        float_type=float_type,
        jitter_method=jitter_method,
        sample_noise=False,  # Returns sample without noise
    )

    # Generate training set on subset of interval using the sine function
    x_train = nd.array([-4, -3, -2, -1, 1], ctx=ctx, dtype=float_type).reshape(
        context_length, 1
    )
    x_train = nd.tile(x_train, reps=(batch_size, 1, 1))
    y_train = nd.sin(x_train.squeeze(axis=2))

    # Predict exact GP using the GP predictive mean and covariance using the same fixed hyper-parameters
    samples, predictive_mean, predictive_std = gp.exact_inference(
        x_train, y_train, x_test
    )

    assert (
        np.sum(np.isnan(samples.asnumpy())) == 0
    ), "NaNs in predictive samples!"
