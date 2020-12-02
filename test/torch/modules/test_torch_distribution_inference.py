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
from typing import List, Tuple

import numpy as np

import pytest
import torch
import torch.nn as nn
from pydantic import PositiveFloat, PositiveInt
from torch.distributions import Beta
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from gluonts.model.common import NPArrayLike
from gluonts.torch.modules.distribution_output import (
    BetaOutput,
    DistributionOutput,
)

NUM_SAMPLES = 2000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1

np.random.seed(1)
torch.manual_seed(1)


def inv_softplus(y: NPArrayLike) -> np.ndarray:
    # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
    return np.log(np.exp(y) - 1)


def maximum_likelihood_estimate_sgd(
    distr_output: DistributionOutput,
    samples: torch.Tensor,
    init_biases: List[np.ndarray] = None,
    num_epochs: PositiveInt = PositiveInt(5),
    learning_rate: PositiveFloat = PositiveFloat(1e-2),
):
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


@pytest.mark.parametrize("concentration1, concentration0", [(3.75, 1.25)])
def test_beta_likelihood(concentration1: float, concentration0: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    concentration1s = torch.zeros((NUM_SAMPLES,)) + concentration1
    concentration0s = torch.zeros((NUM_SAMPLES,)) + concentration0

    distr = Beta(concentration1s, concentration0s)
    samples = distr.sample()

    init_biases = [
        inv_softplus(
            concentration1 - START_TOL_MULTIPLE * TOL * concentration1
        ),
        inv_softplus(
            concentration0 - START_TOL_MULTIPLE * TOL * concentration0
        ),
    ]

    concentration1_hat, concentration0_hat = maximum_likelihood_estimate_sgd(
        BetaOutput(),
        samples,
        init_biases=init_biases,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    assert (
        np.abs(concentration1_hat - concentration1) < TOL * concentration1
    ), f"concentration1 did not match: concentration1 = {concentration1}, concentration1_hat = {concentration1_hat}"
    assert (
        np.abs(concentration0_hat - concentration0) < TOL * concentration0
    ), f"concentration0 did not match: concentration0 = {concentration0}, concentration0_hat = {concentration0_hat}"
