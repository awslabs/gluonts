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
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
from pydantic import PositiveFloat, PositiveInt
from scipy.special import softmax
from torch.distributions import (
    Beta,
    Gamma,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from gluonts.torch.distributions import (
    BetaOutput,
    DistributionOutput,
    GammaOutput,
    NegativeBinomialOutput,
    NormalOutput,
    PoissonOutput,
    SplicedBinnedPareto,
    SplicedBinnedParetoOutput,
    StudentTOutput,
)
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood

NUM_SAMPLES = 3_000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1

np.random.seed(1)
torch.manual_seed(1)


def inv_softplus(y: np.ndarray) -> np.ndarray:
    # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
    return np.log(np.exp(y) - 1)


def inv_softmax(y: np.ndarray) -> np.ndarray:
    """
    Inverse of the scipy.special.softmax
    """
    return np.log(y)


def maximum_likelihood_estimate_sgd(
    distr_output: DistributionOutput,
    samples: torch.Tensor,
    init_biases: List[np.ndarray] = None,
    num_epochs: PositiveInt = PositiveInt(5),
    learning_rate: PositiveFloat = PositiveFloat(1e-2),
    loss: DistributionLoss = NegativeLogLikelihood(),
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


def compare_logits(
    logits_true: np.array, logits_hat: np.array, TOL: int = 0.3
):
    """
    Since logits {x_i} and logits {x_i + K} will result in the same probabilities {exp(x_i)/(sum_j exp(x_j))},
    one needs to apply softmax and inv_softmax before comparing logits within a certain tolerance
    """
    param_true = inv_softmax(softmax(logits_true, axis=-1))
    param_hat = inv_softmax(softmax(logits_hat, axis=-1))
    assert (
        np.abs(param_hat - param_true) < TOL * np.abs(param_true)
    ).all(), f"logits did not match: logits_true = {param_true}, logits_hat = {param_hat}"


@pytest.mark.flaky(max_runs=3, min_passes=1)
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


@pytest.mark.flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("concentration, rate", [(3.75, 1.25)])
def test_gamma_likelihood(concentration: float, rate: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    concentrations = torch.zeros((NUM_SAMPLES,)) + concentration
    rates = torch.zeros((NUM_SAMPLES,)) + rate

    distr = Gamma(concentrations, rates)
    samples = distr.sample()

    init_biases = [
        inv_softplus(concentration - START_TOL_MULTIPLE * TOL * concentration),
        inv_softplus(rate - START_TOL_MULTIPLE * TOL * rate),
    ]

    concentration_hat, rate_hat = maximum_likelihood_estimate_sgd(
        GammaOutput(),
        samples,
        init_biases=init_biases,
        learning_rate=PositiveFloat(0.05),
        num_epochs=PositiveInt(10),
    )

    assert (
        np.abs(concentration_hat - concentration) < TOL * concentration
    ), f"concentration did not match: concentration = {concentration}, concentration_hat = {concentration_hat}"
    assert (
        np.abs(rate_hat - rate) < TOL * rate
    ), f"rate did not match: rate = {rate}, rate_hat = {rate_hat}"


@pytest.mark.flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("loc, scale,", [(1.0, 0.1)])
def test_normal_likelihood(loc: float, scale: float):
    locs = torch.zeros((NUM_SAMPLES,)) + loc
    scales = torch.zeros((NUM_SAMPLES,)) + scale

    distr = Normal(loc=locs, scale=scales)
    samples = distr.sample()

    init_bias = [
        loc - START_TOL_MULTIPLE * TOL * loc,
        inv_softplus(scale - START_TOL_MULTIPLE * TOL * scale),
    ]

    loc_hat, scale_hat = maximum_likelihood_estimate_sgd(
        NormalOutput(),
        samples,
        init_biases=init_bias,
        num_epochs=5,
        learning_rate=1e-3,
    )

    assert (
        np.abs(loc_hat - loc) < TOL * loc
    ), f"loc did not match: loc = {loc}, loc_hat = {loc_hat}"
    assert (
        np.abs(scale_hat - scale) < TOL * scale
    ), f"scale did not match: scale = {scale}, scale_hat = {scale_hat}"


@pytest.mark.flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("df, loc, scale,", [(6.0, 2.3, 0.7)])
def test_studentT_likelihood(df: float, loc: float, scale: float):
    dfs = torch.zeros((NUM_SAMPLES,)) + df
    locs = torch.zeros((NUM_SAMPLES,)) + loc
    scales = torch.zeros((NUM_SAMPLES,)) + scale

    distr = StudentT(df=dfs, loc=locs, scale=scales)
    samples = distr.sample()

    init_bias = [
        inv_softplus(df - 2),
        loc - START_TOL_MULTIPLE * TOL * loc,
        inv_softplus(scale - START_TOL_MULTIPLE * TOL * scale),
    ]

    df_hat, loc_hat, scale_hat = maximum_likelihood_estimate_sgd(
        StudentTOutput(),
        samples,
        init_biases=init_bias,
        num_epochs=15,
        learning_rate=1e-3,
    )

    assert (
        np.abs(df_hat - df) < TOL * df
    ), f"df did not match: df = {df}, df_hat = {df_hat}"
    assert (
        np.abs(loc_hat - loc) < TOL * loc
    ), f"loc did not match: loc = {loc}, loc_hat = {loc_hat}"
    assert (
        np.abs(scale_hat - scale) < TOL * scale
    ), f"scale did not match: scale = {scale}, scale_hat = {scale_hat}"


@pytest.mark.flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("rate", [1.0])
def test_poisson(rate: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """
    # generate samples
    rates = torch.zeros((NUM_SAMPLES,)) + rate

    poisson_distr = Poisson(rate=rates)
    samples = poisson_distr.sample()

    init_biases = [inv_softplus(rate - START_TOL_MULTIPLE * TOL * rate)]

    (rate_hat,) = maximum_likelihood_estimate_sgd(
        PoissonOutput(),
        samples,
        init_biases=init_biases,
        num_epochs=20,
        learning_rate=0.05,
    )

    assert (
        np.abs(rate_hat - rate) < TOL * rate
    ), f"rate did not match: rate = {rate}, rate_hat = {rate_hat}"


@pytest.mark.parametrize(
    "total_count, logit",
    [
        (1.4, 0.56),
        (1.4, 2.0),
    ],
)
@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_neg_binomial(total_count: float, logit: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """
    # generate samples
    total_counts = torch.zeros((NUM_SAMPLES,)) + total_count
    logits = torch.zeros((NUM_SAMPLES,)) + logit

    neg_bin_distr = NegativeBinomial(total_count=total_counts, logits=logits)
    samples = neg_bin_distr.sample()

    init_biases = [
        inv_softplus(total_count - START_TOL_MULTIPLE * TOL * total_count),
        logit - START_TOL_MULTIPLE * TOL * logit,
    ]
    print("type(init_biases)", type(init_biases))
    print("len(init_biases)", len(init_biases))
    print("type(init_biases[0])", type(init_biases[0]))

    total_count_hat, logit_hat = maximum_likelihood_estimate_sgd(
        NegativeBinomialOutput(),
        samples,
        init_biases=init_biases,
        num_epochs=15,
    )

    assert (
        np.abs(total_count_hat - total_count) < TOL * total_count
    ), f"total_count did not match: total_count = {total_count}, total_count_hat = {total_count_hat}"
    assert (
        np.abs(logit_hat - logit) < TOL * logit
    ), f"logit did not match: logit = {logit}, logit_hat = {logit_hat}"


percentile_tail = 0.05


@pytest.mark.parametrize("percentile_tail", [percentile_tail])
@pytest.mark.parametrize(
    "np_logits", [[percentile_tail, 1 - 2 * percentile_tail, percentile_tail]]
)
@pytest.mark.parametrize(
    "lower_gp_xi, lower_gp_beta", [(0.4, PositiveFloat(1.5))]
)
@pytest.mark.parametrize(
    "upper_gp_xi, upper_gp_beta", [(0.3, PositiveFloat(1.0))]
)
@pytest.mark.timeout(300)
@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_splicedbinnedpareto_likelihood(
    percentile_tail: PositiveFloat,
    np_logits: np.ndarray,
    lower_gp_xi: float,
    lower_gp_beta: PositiveFloat,
    upper_gp_xi: float,
    upper_gp_beta: PositiveFloat,
) -> None:
    # percentile_tail = 0.05
    bins_lower_bound, bins_upper_bound = -1.0, 2.0
    # np_logits = [percentile_tail, 1 - 2 * percentile_tail, percentile_tail]
    # upper_gp_xi, upper_gp_beta = 0.3, 1
    # lower_gp_xi, lower_gp_beta = 0.4, 1

    assert percentile_tail < 1, "percentile_tail should be between 0.0 and 1.0"

    # Specify the distribution with parameter value from which to obtain samples
    NUM_SAMPLES = 10_000

    logits = torch.tensor(np_logits).unsqueeze(0)
    logits = logits.repeat(NUM_SAMPLES, 1)
    nbins = logits.shape[-1]
    distr_true = SplicedBinnedPareto(
        bins_lower_bound=bins_lower_bound,
        bins_upper_bound=bins_upper_bound,
        tail_percentile_gen_pareto=percentile_tail,
        numb_bins=nbins,
        logits=logits,
        lower_gp_xi=torch.tensor(lower_gp_xi, dtype=torch.float).repeat(
            NUM_SAMPLES, 1
        ),
        lower_gp_beta=torch.tensor(lower_gp_beta, dtype=torch.float).repeat(
            NUM_SAMPLES, 1
        ),
        upper_gp_xi=torch.tensor(upper_gp_xi, dtype=torch.float).repeat(
            NUM_SAMPLES, 1
        ),
        upper_gp_beta=torch.tensor(upper_gp_beta, dtype=torch.float).repeat(
            NUM_SAMPLES, 1
        ),
    )
    samples = distr_true.sample()

    # DistributionOutput
    distr_output = SplicedBinnedParetoOutput(
        bins_lower_bound=bins_lower_bound,
        bins_upper_bound=bins_upper_bound,
        num_bins=nbins,
        tail_percentile_gen_pareto=percentile_tail,
    )

    params_hat_values = maximum_likelihood_estimate_sgd(
        distr_output,
        samples,
        num_epochs=50,
    )

    params_hat = dict(
        zip(list(distr_output.args_dim.keys()), params_hat_values)
    )

    for param_name in params_hat.keys():
        param_hat = params_hat[param_name]
        param_true = (
            torch.unique(distr_true.__getattribute__(param_name), dim=0)
            .detach()
            .numpy()
        )

        if param_name == "logits":
            compare_logits(param_true, param_hat, TOL=TOL)
        else:
            assert (
                np.abs(param_hat - param_true) < TOL * param_true
            ).all(), f"{param_name} did not match: {param_name} = {param_true}, {param_name}_hat = {param_hat}"
