# Standard library imports
import math

# First-party imports
from gluonts.kernels import RBFKernel
from gluonts.model.gp_forecaster.gaussian_process import GaussianProcess

# Third-party imports
import mxnet.ndarray as nd
import mxnet as mx
import numpy as np


# In this file, we generate a synthetic dataset where the time series is drawn from a zero-mean Gaussian process with
# RBF covariance function.  We then generate an approximate training set and given fixed hyper-parameters, compute
# the Gaussian Process predictive mean and covariance matrices and sample from this distribution.


def main():
    # Initialize problem parameters
    batch_size = 1
    prediction_length = 50
    context_length = 5
    axis = [-5, 5, -3, 3]
    float_type = np.float64
    ctx = mx.Context("gpu")

    num_samples = 3
    ts_idx = 0

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
        sample_noise=False,  # Returns sample without noise
    )
    mean = nd.zeros((batch_size, prediction_length), ctx=ctx, dtype=float_type)
    covariance = rbf_kernel.kernel_matrix(x_test, x_test)
    gp.plot(x_test=x_test, samples=gp.sample(mean, covariance), ts_idx=ts_idx)

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

    gp.plot(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        ts_idx=ts_idx,
        mean=predictive_mean,
        std=predictive_std,
        samples=samples,
        axis=axis,
    )


if __name__ == "__main__":
    main()
