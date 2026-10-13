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
Test that maximizing likelihood allows to correctly recover Tweedie distribution
parameters.
"""
from typing import List

import numpy as np
import pytest
from lightning import seed_everything

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from gluonts.pydantic import PositiveFloat, PositiveInt
from gluonts.torch.distributions import (
    DistributionOutput,
    TweedieOutput,
    Tweedie,
)

NUM_SAMPLES = 5_000
BATCH_SIZE = 64
TOL = 0.30
START_TOL_MULTIPLE = 1

np.random.seed(42)
torch.manual_seed(42)


def inv_softplus(y: np.ndarray) -> np.ndarray:
    """Inverse of softplus: x = log(exp(y) - 1)."""
    return np.log(np.exp(y) - 1)


def inv_sigmoid(y: float) -> float:
    """Inverse of sigmoid: x = log(y / (1 - y))."""
    return np.log(y / (1 - y))


def maximum_likelihood_estimate_sgd(
    distr_output: DistributionOutput,
    samples: torch.Tensor,
    init_biases: List[np.ndarray] = None,
    num_epochs: PositiveInt = PositiveInt(5),
    learning_rate: PositiveFloat = PositiveFloat(1e-2),
):
    """
    Estimate distribution parameters using maximum likelihood via SGD.

    Parameters
    ----------
    distr_output
        The distribution output class to use.
    samples
        Tensor of samples from the true distribution.
    init_biases
        Initial biases for the neural network layers.
    num_epochs
        Number of training epochs.
    learning_rate
        Learning rate for SGD optimizer.

    Returns
    -------
    List of estimated parameters.
    """
    arg_proj = distr_output.get_args_proj(in_features=1)
    if init_biases is not None:
        for param, bias in zip(arg_proj.proj, init_biases):
            nn.init.constant_(param.bias, bias)
    dummy_data = torch.ones((len(samples), 1))
    dataset = TensorDataset(dummy_data, samples)
    train_data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = SGD(arg_proj.parameters(), lr=learning_rate)
    for e in range(num_epochs):
        cumulative_loss = 0
        num_batches = 0
        for i, (data, sample_label) in enumerate(train_data):
            optimizer.zero_grad()
            distr_args = arg_proj(data)
            distr = distr_output.distribution(distr_args)
            loss = -distr.log_prob(sample_label).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            clip_grad_norm_(arg_proj.parameters(), 10.0)
            optimizer.step()
            num_batches += 1
            cumulative_loss += loss.item()
    if len(distr_args[0].shape) == 1:
        return [
            param.detach().numpy() for param in arg_proj(torch.ones((1, 1)))
        ]
    return [
        param[0].detach().numpy() for param in arg_proj(torch.ones((1, 1)))
    ]


@pytest.mark.flaky(retries=3)
@pytest.mark.parametrize(
    "mu, dispersion, power",
    [
        (2.0, 1.0, 1.5),  # Standard case in middle of power range
        (5.0, 0.5, 1.3),  # Higher mean, lower dispersion, power near Poisson
        (1.0, 2.0, 1.7),  # Lower mean, higher dispersion, power near Gamma
    ],
)
def test_tweedie_likelihood(mu: float, dispersion: float, power: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters.
    """
    seed_everything(42)

    # Generate samples using the Tweedie distribution
    mus = torch.zeros((NUM_SAMPLES,)) + mu
    dispersions = torch.zeros((NUM_SAMPLES,)) + dispersion
    powers = torch.zeros((NUM_SAMPLES,)) + power

    distr = Tweedie(mu=mus, dispersion=dispersions, power=powers)
    samples = distr.sample()

    # Compute initial biases (inverse of domain_map transformations)
    # mu: softplus(x) + eps = mu => x = inv_softplus(mu - eps)
    # dispersion: softplus(x) + eps = dispersion => x = inv_softplus(dispersion - eps)
    # power: 1 + 0.01 + 0.98 * sigmoid(x) = power => sigmoid(x) = (power - 1.01) / 0.98
    eps = np.finfo(np.float32).eps

    mu_init = inv_softplus(mu * (1 - START_TOL_MULTIPLE * TOL) - eps)
    disp_init = inv_softplus(dispersion * (1 - START_TOL_MULTIPLE * TOL) - eps)
    power_normalized = (power * (1 - START_TOL_MULTIPLE * TOL * 0.1) - 1.01) / 0.98
    power_init = inv_sigmoid(np.clip(power_normalized, 0.01, 0.99))

    init_biases = [mu_init, disp_init, power_init]

    mu_hat, dispersion_hat, power_hat = maximum_likelihood_estimate_sgd(
        TweedieOutput(),
        samples,
        init_biases=init_biases,
        learning_rate=PositiveFloat(0.01),
        num_epochs=PositiveInt(20),
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(dispersion_hat - dispersion) < TOL * dispersion
    ), f"dispersion did not match: dispersion = {dispersion}, dispersion_hat = {dispersion_hat}"
    assert (
        np.abs(power_hat - power) < TOL * power
    ), f"power did not match: power = {power}, power_hat = {power_hat}"


@pytest.mark.flaky(retries=3)
def test_tweedie_sampling_properties() -> None:
    """
    Test that Tweedie sampling produces expected statistical properties.
    """
    seed_everything(42)

    mu = 3.0
    dispersion = 1.5
    power = 1.5

    mus = torch.zeros((NUM_SAMPLES,)) + mu
    dispersions = torch.zeros((NUM_SAMPLES,)) + dispersion
    powers = torch.zeros((NUM_SAMPLES,)) + power

    distr = Tweedie(mu=mus, dispersion=dispersions, power=powers)
    samples = distr.sample()

    # Check basic properties
    assert (samples >= 0).all(), "Tweedie samples should be non-negative"

    # Check that there are some zeros (characteristic of Tweedie for power in (1,2))
    zero_proportion = (samples == 0).float().mean().item()
    assert zero_proportion > 0.01, "Should have some zero samples"
    assert zero_proportion < 0.99, "Should have some non-zero samples"

    # Check mean is approximately correct (with tolerance)
    sample_mean = samples.mean().item()
    assert (
        np.abs(sample_mean - mu) < 0.3 * mu
    ), f"Sample mean {sample_mean} should be close to {mu}"

    # Check variance approximately follows V = phi * mu^p
    expected_variance = dispersion * (mu ** power)
    sample_variance = samples.var().item()
    assert (
        np.abs(sample_variance - expected_variance) < 0.5 * expected_variance
    ), f"Sample variance {sample_variance} should be close to {expected_variance}"


@pytest.mark.flaky(retries=3)
def test_tweedie_mean_variance_properties() -> None:
    """
    Test that Tweedie distribution correctly computes mean and variance.
    """
    mu = torch.tensor([1.0, 2.0, 3.0])
    dispersion = torch.tensor([0.5, 1.0, 1.5])
    power = torch.tensor([1.3, 1.5, 1.7])

    distr = Tweedie(mu=mu, dispersion=dispersion, power=power)

    # Check mean equals mu
    assert torch.allclose(
        distr.mean, mu
    ), "Mean should equal mu"

    # Check variance equals dispersion * mu^power
    expected_variance = dispersion * torch.pow(mu, power)
    assert torch.allclose(
        distr.variance, expected_variance
    ), "Variance should equal dispersion * mu^power"


@pytest.mark.flaky(retries=3)
def test_tweedie_log_prob_shape() -> None:
    """
    Test that log_prob returns correct shapes.
    """
    batch_size = 10
    mu = torch.ones(batch_size) * 2.0
    dispersion = torch.ones(batch_size) * 1.0
    power = torch.ones(batch_size) * 1.5

    distr = Tweedie(mu=mu, dispersion=dispersion, power=power)

    # Test with matching shape
    values = torch.rand(batch_size) * 5
    log_probs = distr.log_prob(values)
    assert log_probs.shape == torch.Size(
        [batch_size]
    ), f"Expected shape {[batch_size]}, got {log_probs.shape}"

    # Test with zeros (should handle point mass at zero)
    values_with_zeros = torch.zeros(batch_size)
    log_probs_zeros = distr.log_prob(values_with_zeros)
    assert log_probs_zeros.shape == torch.Size(
        [batch_size]
    ), f"Expected shape {[batch_size]}, got {log_probs_zeros.shape}"
    assert torch.isfinite(
        log_probs_zeros
    ).all(), "Log prob for zeros should be finite"


@pytest.mark.flaky(retries=3)
def test_tweedie_output_domain_map() -> None:
    """
    Test that TweedieOutput.domain_map produces valid parameters.
    """
    # Random unbounded inputs
    torch.manual_seed(42)
    mu_raw = torch.randn(10, 1)
    dispersion_raw = torch.randn(10, 1)
    power_raw = torch.randn(10, 1)

    mu, dispersion, power = TweedieOutput.domain_map(
        mu_raw, dispersion_raw, power_raw
    )

    # Check constraints
    assert (mu > 0).all(), "mu should be positive"
    assert (dispersion > 0).all(), "dispersion should be positive"
    assert (power > 1).all(), "power should be > 1"
    assert (power < 2).all(), "power should be < 2"

    # Check shapes (should squeeze last dimension)
    assert mu.shape == torch.Size([10]), f"Expected shape [10], got {mu.shape}"
    assert dispersion.shape == torch.Size(
        [10]
    ), f"Expected shape [10], got {dispersion.shape}"
    assert power.shape == torch.Size(
        [10]
    ), f"Expected shape [10], got {power.shape}"
