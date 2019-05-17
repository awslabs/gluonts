# Standard library imports
import math

# First-party imports
from gluonts.kernels import RBFKernel
from gluonts.gp import GaussianProcess
from gluonts.support.linalg_util import jitter_cholesky, jitter_cholesky_eig

# Third-party imports
import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import pytest


# This test verifies that both eigenvalue decomposition and iterative jitter method
# make a non-positive definite matrix positive definite to be able to compute the cholesky.
# Both gpu and cpu as well as single and double precision are tested.
# FIXME: Add gpu support for the tests in braxil gets error Check failed: e == cudaSuccess || e ==
# FIXME: cudaErrorCudartUnloading CUDA: CUDA driver version is insufficient for CUDA runtime version
# FIXME: @pytest.mark.parametrize('ctx', [mx.Context('gpu'), mx.Context('cpu')])
@pytest.mark.parametrize('jitter_method', ['iter', 'eig'])
@pytest.mark.parametrize('float_type', [np.float32, np.float64])
def test_jitter_unit(jitter_method, float_type, ctx=mx.Context('cpu')):
    matrix = nd.array(
        [[[1, 2], [3, 4]], [[10, 100], [-21.5, 41]]], ctx=ctx, dtype=float_type
    )
    F = mx.nd
    num_data_points = matrix.shape[1]
    if jitter_method == 'eig':
        L = jitter_cholesky_eig(F, matrix, num_data_points, ctx, float_type)
    elif jitter_method == 'iter':
        L = jitter_cholesky(F, matrix, num_data_points, ctx, float_type)
    assert np.sum(np.isnan(L.asnumpy())) == 0, 'NaNs in Cholesky factor!'


# This test tests that the noiseless sample generated from the synthetic example does not have NaNs in it.
# Without the jitter method, NaNs occurs on the gpu for single and double precision and on the cpu for only single
# precision.  This test verifies that applying the default jitter method fixes these numerical issues on both cpu
# and gpu and for single and double precision.
# FIXME: Add gpu support for the tests in braxil gets error Check failed: e == cudaSuccess || e ==
# FIXME: cudaErrorCudartUnloading CUDA: CUDA driver version is insufficient for CUDA runtime version
# FIXME: @pytest.mark.parametrize('ctx', [mx.Context('gpu'), mx.Context('cpu')])
@pytest.mark.parametrize('jitter_method', ['iter', 'eig'])
@pytest.mark.parametrize('float_type', [np.float32, np.float64])
def test_jitter_synthetic(
    jitter_method, float_type, ctx=mx.Context('cpu')
) -> None:
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
    ), 'NaNs in predictive samples!'
