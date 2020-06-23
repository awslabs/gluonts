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

"""
Test that maximizing likelihood allows to correctly recover distribution parameters for all
distributions exposed to the user.
"""
# Standard library imports
from typing import List

# Third-party imports
import mxnet as mx
import numpy as np
import pytest
from pydantic import PositiveFloat, PositiveInt

# First-party imports
from gluonts.model.common import NPArrayLike
from gluonts.model.tpp.distribution import (
    Loglogistic,
    LoglogisticOutput,
    Weibull,
    WeibullOutput,
    TPPDistributionOutput,
)

NUM_SAMPLES = 2000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1

np.random.seed(1)
mx.random.seed(1)


def inv_softplus(y: NPArrayLike) -> np.ndarray:
    # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
    return np.log(np.exp(y) - 1)


def maximum_likelihood_estimate_sgd(
    distr_output: TPPDistributionOutput,
    samples: mx.ndarray,
    init_biases: List[mx.ndarray.NDArray] = None,
    num_epochs: PositiveInt = PositiveInt(10),
    learning_rate: PositiveFloat = PositiveFloat(1e-2),
    hybridize: bool = False,
) -> List[np.ndarray]:
    model_ctx = mx.cpu()

    arg_proj = distr_output.get_args_proj()
    arg_proj.initialize()

    if hybridize:
        arg_proj.hybridize()

    if init_biases is not None:
        for param, bias in zip(arg_proj.proj, init_biases):
            param.params[param.prefix + "bias"].initialize(
                mx.initializer.Constant(bias), force_reinit=True
            )

    trainer = mx.gluon.Trainer(
        arg_proj.collect_params(),
        "sgd",
        {"learning_rate": learning_rate, "clip_gradient": 10.0},
    )

    # The input data to our model is one-dimensional
    dummy_data = mx.nd.array(np.ones((len(samples), 1)))

    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(dummy_data, samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for e in range(num_epochs):
        cumulative_loss = 0
        num_batches = 0
        # inner loop
        for i, (data, sample_label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            sample_label = sample_label.as_in_context(model_ctx)
            with mx.autograd.record():
                distr_args = arg_proj(data)
                distr = distr_output.distribution(distr_args)
                loss = distr.loss(sample_label)
                if not hybridize:
                    assert loss.shape == distr.batch_shape
            loss.backward()
            trainer.step(BATCH_SIZE)
            num_batches += 1

            cumulative_loss += mx.nd.mean(loss).asscalar()

            assert not np.isnan(cumulative_loss)
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_batches))

    if len(distr_args[0].shape) == 1:
        return [
            param.asnumpy() for param in arg_proj(mx.nd.array(np.ones((1, 1))))
        ]

    return [
        param[0].asnumpy() for param in arg_proj(mx.nd.array(np.ones((1, 1))))
    ]


@pytest.mark.parametrize("rate, shape", [(2.0, 1.5)])
def test_weibull_likelihood(rate: float, shape: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    rates = mx.nd.zeros((NUM_SAMPLES,)) + rate
    shapes = mx.nd.zeros((NUM_SAMPLES,)) + shape

    distr = Weibull(rates, shapes)
    samples = distr.sample()

    init_biases = [
        inv_softplus(rate - START_TOL_MULTIPLE * TOL * rate),
        inv_softplus(shape - START_TOL_MULTIPLE * TOL * shape),
    ]

    rate_hat, shape_hat = maximum_likelihood_estimate_sgd(
        WeibullOutput(),
        samples,
        init_biases=init_biases,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    print("rate:", rate_hat, "shape:", shape_hat)
    assert (
        np.abs(rate_hat - rate) < TOL * rate
    ), f"rate did not match: rate = {rate}, rate_hat = {rate_hat}"
    assert (
        np.abs(shape_hat - shape) < TOL * shape
    ), f"shape did not match: shape = {shape}, shape_hat = {shape_hat}"


@pytest.mark.parametrize("mu, sigma", [(1.25, 0.5)])
def test_loglogistic_likelihood(mu: float, sigma: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    mus = mx.nd.zeros((NUM_SAMPLES,)) + mu
    sigmas = mx.nd.zeros((NUM_SAMPLES,)) + sigma

    distr = Loglogistic(mus, sigmas)
    samples = distr.sample()

    init_biases = [
        mu - START_TOL_MULTIPLE * TOL * mu,
        inv_softplus(sigma - START_TOL_MULTIPLE * TOL * sigma),
    ]

    mu_hat, sigma_hat = maximum_likelihood_estimate_sgd(
        LoglogisticOutput(),
        samples,
        init_biases=init_biases,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    print("mu:", mu_hat, "sigma:", sigma_hat)
    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(sigma_hat - sigma) < TOL * sigma
    ), f"sigma did not match: sigma = {sigma}, sigma_hat = {sigma_hat}"
