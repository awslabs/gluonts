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

from typing import List

import numpy as np
from scipy import stats
import torch


def add_spikes(ts: torch.Tensor, only_upper_spikes: bool = False):
    """
    Adds spikes to 15% of the time series in the form of heavy-tailed
    (Generalized Pareto) realizations.

    Arguments:
        ts: time series
        only_upper_spikes: boolean to indicate upper-tailed or two-tailed spikes
    """
    num_spikes = int(0.15 * ts.shape[0])
    indices_for_gp_spikes = np.random.choice(
        np.arange(len(ts)), replace=False, size=num_spikes
    )
    spike_direction = np.random.choice([-1, 1], replace=True, size=num_spikes)
    if only_upper_spikes:
        spike_direction = np.ones_like(spike_direction)
    spikes = stats.genpareto(1 / 50).rvs(num_spikes) * spike_direction
    ts[indices_for_gp_spikes] += spikes
    return ts


def create_ds(
    num_points: int,
    t_dof: int = 10,
    noise_mult: float = 0.25,
    points_per_sinusoid: int = 100,
    magnitude_sin: float = 1,
):
    """
    Creates noisy sinusoid.
    (Noise distributed as student t with degrees of freedom = t_dof.)
    Returns tensor of shape (1, 1, num_points).

    Arguments:
      num_points: int, number of points in the dataset.
      t_dof: int, degrees of freedom for student t distribution.
      noise_mult: float, standard deviation.
      points_per_sinusoid: int, datapoints per sine period
      magnitude_sin: float, magnitude of sine amplitude
    """
    end = num_points / points_per_sinusoid * 2 * np.pi
    sinusoid = np.sin(np.linspace(0, end, num_points)) * magnitude_sin
    noise = np.random.standard_t(t_dof, num_points) * noise_mult
    values = sinusoid + noise + 5.0
    values = add_spikes(values)
    return torch.tensor(values.reshape(1, 1, -1))


def create_ds_iid(num_points: int, noise_mult: float = 0.25):
    """
    Creates heavy-tailed gaussian iid. Returns tensor of shape (1, 1,
    num_points).

    Arguments:
      num_points: int, number of points in the dataset.
      noise_mult: float, standard deviation
    """
    values = np.random.normal(0, noise_mult, size=num_points)
    values = add_spikes(values)
    return torch.tensor(values.reshape(1, 1, -1))


def add_spikes_asymmetric(
    ts: torch.Tensor, xi: List[float] = [1 / 50.0, 1 / 25.0]
):
    """
    Adds spikes to 15% of the time series in the form of heavy-tailed
    (Generalized Pareto) realizations.

    Arguments:
        ts: time series
        xi: [float, float], GenPareto heaviness parameter for [lower, upper] noise respectively
    """
    num_spikes = int(0.15 * ts.shape[0])
    half_num_spikes = [int(num_spikes / 2)]
    half_num_spikes.append(num_spikes - half_num_spikes[0])
    spike_direction = [-1, 1]

    indices_for_gp_spikes = np.random.choice(
        np.arange(len(ts)), replace=False, size=num_spikes
    )

    idx = 0
    spikes = (
        stats.genpareto(xi[idx]).rvs(half_num_spikes[idx])
        * spike_direction[idx]
    )
    ts[indices_for_gp_spikes[: half_num_spikes[0]]] += spikes

    idx = 1
    spikes = (
        stats.genpareto(xi[idx]).rvs(half_num_spikes[idx])
        * spike_direction[idx]
    )
    ts[indices_for_gp_spikes[half_num_spikes[0] :]] += spikes

    return ts


def create_ds_asymmetric(
    num_points: int,
    t_dof: List[float] = [10, 10],
    noise_mult: List[float] = [0.25, 0.25],
    xi: List[float] = [1 / 50.0, 1 / 25.0],
    points_per_sinusoid: int = 100,
    magnitude_sin: float = 1,
):
    """
    Creates noisy sinusoid.
    (Noise distributed as student t with degrees of freedom = t_dof.)
    Returns tensor of shape (1, 1, num_points).

    Arguments:
      num_points: int, number of points in the dataset.
      t_dof: [int, int], degrees of freedom for Students'-t distribution for [lower, upper] noise respectively.
      noise_mult: [float, float], standard deviation for [lower, upper] noise respectively.
      points_per_sinusoid: int, datapoints per sine period
      magnitude_sin: float, magnitude of sine amplitude
    """
    end = num_points / points_per_sinusoid * 2 * np.pi
    sinusoid = np.sin(np.linspace(0, end, num_points)) * magnitude_sin

    idx = 0
    noise_lower = (
        np.random.standard_t(t_dof[idx], num_points) * noise_mult[idx]
    )
    noise_lower = np.minimum(0, noise_lower)

    idx = 1
    noise_upper = (
        np.random.standard_t(t_dof[idx], num_points) * noise_mult[idx]
    )
    noise_upper = np.maximum(0, noise_upper)

    values = sinusoid + noise_lower + noise_upper + 5.0
    values = add_spikes_asymmetric(values, xi=xi)
    return torch.tensor(values.reshape(1, 1, -1))
