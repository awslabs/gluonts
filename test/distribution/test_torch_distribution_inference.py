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
from typing import Iterable, List, Tuple

import numpy as np
import pytest
from pydantic import PositiveFloat, PositiveInt
from scipy import stats
from scipy.special import softmax
from tqdm import tqdm


import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentT,
    StudentTOutput,
)

from gluonts.torch.distribution.spliced_binned_pareto import (
    SplicedBinnedPareto,
    SplicedBinnedParetoOutput,
)


NUM_SAMPLES = 5_000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1

np.random.seed(1)
torch.manual_seed(1)

# Device
cuda_id = "0"
if torch.cuda.is_available():
    dev = f"cuda:{cuda_id}"
else:
    dev = "cpu"
device = torch.device(dev)
print(device)


def inv_softmax(y):
    """
    Inverse of the scipy.special.softmax
    """
    return np.log(y)


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
    ).all(), f"{param_name} did not match: {param_name} = {param_true}, {param_name}_hat = {param_hat}"


# Super simple model for Marginal Distribution:
def append_layer(
    l_layers: List,
    input_dimension: int,
    output_dimension: int,
    dropout_probability: PositiveFloat = PositiveFloat(0.25),
):
    linear = torch.nn.Linear(input_dimension, output_dimension)
    l_layers.append(linear)

    l_layers.append(torch.nn.Dropout(dropout_probability))
    l_layers.append(torch.nn.LeakyReLU())
    return l_layers


class DistributionOutputNN(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int = 1,  # Dimension of input data
        number_hidden_layers: int = 10,
        number_hidden_dimensions: int = 30,
        distr_output: DistributionOutput = StudentTOutput(),
        verbose=False,
    ):
        super(DistributionOutputNN, self).__init__()

        # The number of parameters to fit
        self.distr_output = distr_output
        self.output_dimension = len(self.distr_output.args_dim)

        ###  Creating the main network:
        net_layers = []

        dropout_probability = 0.5
        dropout_probability_per_layer = np.linspace(
            start=dropout_probability, stop=0.0, num=number_hidden_layers
        )
        if number_hidden_layers > 1:
            dropout_probability_per_layer[-2] = 0.0

        # We add the first layer:
        net_layers = append_layer(
            net_layers,
            input_dimension,
            number_hidden_dimensions,
            dropout_probability,
        )

        # We add each of the hidden layers:
        for i in range(number_hidden_layers - 1):
            net_layers = append_layer(
                net_layers,
                number_hidden_dimensions,
                number_hidden_dimensions,
                dropout_probability_per_layer[i],
            )

        # We add the final layer:
        net_layers.append(
            torch.nn.Linear(number_hidden_dimensions, self.output_dimension)
        )

        # The network
        self.network = torch.nn.Sequential(*net_layers)

        # Reserve for specifying the distribution fit
        self.args_proj = distr_output.get_args_proj(self.output_dimension)
        self.distr_args = None
        self.distr = None

    def forward(self, x):
        net_out = self.network(x)
        net_out_final = (
            net_out.squeeze()
        )  # has shape: *batch_size,output_dimension

        self.distr_args = self.args_proj(net_out_final)
        self.distr = self.distr_output.distribution(self.distr_args)

        return self.distr


class PointDataset(Dataset):
    """
    Creates inputs of value 1 with
    * the same tensor shape as the output
    * on the same device as the output
    """

    def __init__(self, y, device=device):
        self.y = y.double()
        self.x = torch.unsqueeze(torch.ones_like(y), -1).double()
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index].to(self.device), self.y[index].to(self.device)


def maximum_likelihood_estimate_nn(
    model: DistributionOutputNN,
    training_x: np.ndarray,
    learning_rate: float = 0.0005,
    weight_decay: PositiveFloat = PositiveFloat(1e-3),
    batch_size: int = 100,
    epochs: int = 50,
):
    # Get the model parameters:
    model.to(device)
    model = model.double()
    params = list(model.parameters())

    train_losses = []

    optimizer = optim.Adam(
        params=params, lr=learning_rate, weight_decay=weight_decay
    )

    # Training loop:
    for epoch in tqdm(range(epochs)):

        loader_train = iter(
            DataLoader(
                PointDataset(torch.tensor(training_x)),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
        )

        #######################################
        # Train over all the batches:
        #######################################
        model.train()

        batch_losses = []
        for i in range(0, len(training_x) // batch_size):

            # Get the training examples:
            input_features, output_y = loader_train.next()

            # Compute Neg-Loglikelihood
            distr_hat = model(input_features)
            losses = -1 * distr_hat.log_prob(output_y)

            # Back-propagate the loss
            loss = torch.mean(losses)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.item())

        epoch_train_loss = np.mean(batch_losses)
        train_losses.append(epoch_train_loss)

    return distr_hat


@pytest.mark.parametrize("degree_freedom", [PositiveFloat(2.0)])
def test_studentt_likelihood(degree_freedom: PositiveFloat) -> None:

    training_x = stats.t.rvs(degree_freedom, size=NUM_SAMPLES)

    model = DistributionOutputNN(distr_output=StudentTOutput())
    distr_hat = maximum_likelihood_estimate_nn(
        model, training_x, epochs=50, batch_size=BATCH_SIZE
    )

    distr_parameter_names = list(model.distr_output.args_dim.keys())
    distr_parameter_true = dict(
        zip(distr_parameter_names, [degree_freedom, 0, 1])
    )

    for i in range(len(distr_parameter_names)):
        param_name = distr_parameter_names[i]
        if param_name not in ["loc", "scale"]:
            param_hat = (
                distr_hat.__getattribute__(distr_parameter_names[i])
                .detach()
                .numpy()
            )
            param_true = distr_parameter_true[distr_parameter_names[i]]
            assert (
                np.abs(param_hat - param_true) < TOL * param_true
            ).all(), f"{param_name} did not match: {param_name} = {param_true}, {param_name}_hat = {param_hat}"


@pytest.mark.parametrize(
    "np_logits", [[0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0]]
)
@pytest.mark.parametrize(
    "lower_gp_xi, lower_gp_beta", [(0.5, PositiveFloat(1.5))]
)
@pytest.mark.parametrize(
    "upper_gp_xi, upper_gp_beta", [(0.5, PositiveFloat(1.0))]
)
@pytest.mark.timeout(300)
def test_splicedbinnedpareto_likelihood(
    np_logits: np.ndarray,
    lower_gp_xi: float,
    lower_gp_beta: PositiveFloat,
    upper_gp_xi: float,
    upper_gp_beta: PositiveFloat,
) -> None:

    # Specify the distribution with parameter value from which to obtain samples
    variate_dim = 1
    percentile_tail = 0.05

    bins_lower_bound, bins_upper_bound = -3.0, 3.0
    distr = stats.t(3)
    xx = np.linspace(bins_lower_bound, bins_upper_bound, 11)
    pdf = distr.pdf(xx)
    bin_width = 1 / np.sum(pdf)
    np_logits = np.log(pdf / bin_width)

    variates = distr.rvs(20_000)
    percentile_tail = 0.05
    upper_variates = variates[variates > distr.ppf(1 - percentile_tail)]
    lower_variates = variates[variates < distr.ppf(percentile_tail)]
    upper_gp_xi, _, upper_gp_beta = stats.genpareto.fit(
        upper_variates, loc=distr.ppf(1 - percentile_tail)
    )
    lower_gp_xi, _, lower_gp_beta = stats.genpareto.fit(
        -lower_variates, loc=-distr.ppf(percentile_tail)
    )

    logits = torch.tensor(np_logits).unsqueeze(0)
    logits = logits.repeat(variate_dim, BATCH_SIZE, 1)
    nbins = logits.shape[-1]

    sbp_distr = SplicedBinnedPareto(
        bins_lower_bound=bins_lower_bound,
        bins_upper_bound=bins_upper_bound,
        tail_percentile_gen_pareto=percentile_tail,
        numb_bins=nbins,
        logits=logits,
        lower_gp_xi=torch.tensor(upper_gp_xi, dtype=torch.float).repeat(
            variate_dim, BATCH_SIZE, 1
        ),
        lower_gp_beta=torch.tensor(upper_gp_beta, dtype=torch.float).repeat(
            variate_dim, BATCH_SIZE, 1
        ),
        upper_gp_xi=torch.tensor(upper_gp_xi, dtype=torch.float).repeat(
            variate_dim, BATCH_SIZE, 1
        ),
        upper_gp_beta=torch.tensor(upper_gp_beta, dtype=torch.float).repeat(
            variate_dim, BATCH_SIZE, 1
        ),
    )

    samples = sbp_distr.sample(torch.tensor(range(0, NUM_SAMPLES // 16)).shape)
    training_x = samples.numpy().flatten()

    # Maximum Likelihood Estimation on the DistributionOutput
    model = DistributionOutputNN(
        distr_output=SplicedBinnedParetoOutput(
            bins_lower_bound=bins_lower_bound,
            bins_upper_bound=bins_upper_bound,
            num_bins=nbins,
            tail_percentile_gen_pareto=percentile_tail,
        )
    )
    fitted_distr = maximum_likelihood_estimate_nn(
        model, training_x, epochs=50, batch_size=BATCH_SIZE
    )

    # Assert each parameter estimates converges within tolerance to the true value
    distr_parameter_names = list(model.distr_output.args_dim.keys())
    distr_parameter_true = dict(
        zip(
            distr_parameter_names,
            [
                logits.squeeze().numpy(),
                upper_gp_xi,
                upper_gp_beta,
                lower_gp_xi,
                lower_gp_beta,
            ],
        )
    )

    for i in range(len(distr_parameter_names)):
        param_name = distr_parameter_names[i]
        param_hat = (
            fitted_distr.__getattribute__(distr_parameter_names[i])
            .detach()
            .numpy()
        )
        param_true = distr_parameter_true[distr_parameter_names[i]]

        if param_name == "logits":
            compare_logits(param_true, param_hat, TOL=TOL)
        else:
            assert (
                np.abs(param_hat - param_true) < TOL * param_true
            ).all(), f"{param_name} did not match: {param_name} = {param_true}, {param_name}_hat = {param_hat}"
