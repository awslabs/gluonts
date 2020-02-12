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

# Third-party imports
import numpy as np

# First-party imports
from gluonts.kernels import RBFKernel
from gluonts.model.gp_forecaster.gaussian_process import GaussianProcess

# Relative imports
from .data import (
    load_gp_params,
    load_exact_mean,
    load_exact_std,
    load_xfull,
    load_ytrain,
)

# Third-party imports
from mxnet import nd
import pytest


def relative_error(y_hat, y_exact):
    return nd.max(
        nd.max(nd.abs(y_exact - y_hat), axis=1)
        / nd.max(nd.abs(y_exact), axis=1)
    )


@pytest.mark.parametrize(
    "gp_params, mean_exact, std_exact, x_full, y_train",
    [
        # Test inference when training and test length are equal
        (
            load_gp_params(),
            load_exact_mean(),
            load_exact_std(),
            load_xfull(),
            load_ytrain(),
        ),
        # Test inference when training and test length are not equal
        (
            load_gp_params(),
            load_exact_mean()[:, :72],
            load_exact_std()[:, :72],
            load_xfull(),
            load_ytrain(),
        ),
    ],
)
def test_inference(gp_params, mean_exact, std_exact, x_full, y_train) -> None:
    # Initialize problem parameters
    tol = 1e-2
    num_samples = 100
    context_length = y_train.shape[1]
    prediction_length = mean_exact.shape[1]

    # Extract training and test set
    x_train = x_full[:, :context_length, :]
    x_test = x_full[:, context_length : context_length + prediction_length, :]

    amplitude = gp_params[:, 0, :].expand_dims(axis=2)
    length_scale = gp_params[:, 1, :].expand_dims(axis=2)
    sigma = gp_params[:, 2, :].expand_dims(axis=2)

    # Instantiate RBFKernel with its hyper-parameters
    kernel = RBFKernel(amplitude, length_scale)

    # Instantiate gp_inference object and hybridize
    gp = GaussianProcess(
        sigma=sigma,
        kernel=kernel,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        float_type=np.float32,
    )

    # Compute predictive mean and covariance
    _, mean, std = gp.exact_inference(x_train, y_train, x_test)

    # This test compares to the predictive mean and std generated from MATLAB's fitrgp, which
    # outputs the sample with the noise, i.e. adds :math:`sigma^2` to the diagonal of
    # the predictive covariance matrix.
    assert relative_error(mean, mean_exact) <= tol
    assert relative_error(std, std_exact) <= tol
