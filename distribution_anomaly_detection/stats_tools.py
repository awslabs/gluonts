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

import numpy as np


def ecdf(samples, x=None):
    """
    Compute empirical cdf from samples

    Parameters
    ----------
    samples
        Array of samples from the distribution of interest
    x
        Array of points where to evaluate the cdf

    Returns
    -------
    x
        Array of points where to evaluate the cdf
    cdf_eval
        Evaluations of the empirical cdf
    """
    sorted_samples = np.sort(samples)
    if x is None:
        x = sorted_samples

    cdf_eval = np.searchsorted(sorted_samples, x, side="left")
    cdf_eval = cdf_eval / len(sorted_samples)
    cdf_eval[-1] += np.sum(sorted_samples == x[-1]) / len(sorted_samples)
    return x, cdf_eval


def compute_pwlinear_cdf(cdf_grid, cdf_values, x_vector, cdf_sorted=False):
    """
    Computes the piece-wise linear cdf on the elements of x_vector.
    The cdf is defined through cdf_grid and cdf_values.

    Parameters
    ----------
    cdf_grid
        The grid that defines the piece-wise linear cdf, of dimension d. The first element
        of the grid is assumed to be the 0th percentile. The last element is assumed to be the
        100th percentile

    cdf_values
        The values of the cdf on the grid. Must be of dimension d-1 or of dimension d but start
        with 0.

    x_vector
        Points where to compute the cdf

    cdf_sorted
        Optional, whether cdf_grid and cdf_values are already sorted
    """
    if not cdf_sorted:
        cdf_grid = np.sort(cdf_grid)
        cdf_values = np.sort(cdf_values)

    if len(cdf_grid) == len(cdf_values):
        assert (
            cdf_values[0] == 0
        ), "First element of the grid must have corresponding value equal to zero."
    else:
        assert (
            len(cdf_grid) == len(cdf_values) + 1
        ), "cdf_grid's length can be at most larger than cdf_values's by one."
        cdf_values = np.append(0.0, cdf_values)

    # Assert if the cdf is approximately equal to 1
    assert np.abs(np.max(cdf_values) - 1) < 1e-6

    return np.interp(x_vector, cdf_grid, cdf_values)


def compute_pwlinear_quantiles(
    cdf_grid, cdf_values, q_levels, cdf_sorted=False
):
    """
    Computes the quantiles of a piece-wise linear cdf.
    The cdf is defined through cdf_grid and cdf_values.

    Parameters
    ----------
    cdf_grid
        The grid that defines the piece-wise linear cdf, of dimension d. The first element
        of the grid is assumed to be the 0th percentile. The last element is assumed to be the
        100th percentile

    cdf_values
        The values of the cdf on the grid. Must be of dimension d-1 or of dimension d but start
        with 0.

    q_levels
        Quantiles levels

    cdf_sorted
        Optional, whether cdf_grid and cdf_values are already sorted
    """
    if not cdf_sorted:
        cdf_grid = np.sort(cdf_grid)
        cdf_values = np.sort(cdf_values)

    if len(cdf_grid) == len(cdf_values):
        assert (
            cdf_values[0] == 0
        ), "First element of the grid must have corresponding value equal to zero."
    else:
        assert (
            len(cdf_grid) == len(cdf_values) + 1
        ), "cdf_grid's length can be at most larger than cdf_values's by one."
        cdf_values = np.append(0.0, cdf_values)

    # Assert if the cdf is approximately equal to 1
    assert np.abs(np.max(cdf_values) - 1) < 1e-6

    return np.interp(q_levels, cdf_values, cdf_grid)


def ks_pwlinear_cdf(cdf_grid, cdf_values, samples):
    """
    Computes the Kolmogorov-Smirnov distance (https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
    between a piece-wise linear cdf and the empirical distribution of a set of samples.
    The piece-wise linear cdf is defined through cdf_grid and cdf_values.

    Parameters
    ----------
    cdf_grid
        The grid that defines the piece-wise linear cdf, of dimension d. The first element
        of the grid is assumed to be the 0th percentile. The last element is assumed to be the
        100th percentile

    cdf_values
        The values of the cdf on the grid. Must be of dimension d-1 or of dimension d but start
        with 0.

    samples
        Samples of observations from the true distribution.
    """
    cdf_grid = np.sort(cdf_grid)
    cdf_values = np.sort(cdf_values)

    # Evaluation of the cdf on the samples values
    cdf_on_samples = compute_pwlinear_cdf(
        cdf_grid, cdf_values, samples, cdf_sorted=True
    )

    n_samples = len(samples)
    result = np.max(
        np.abs(np.arange(1, n_samples + 1) / n_samples - cdf_on_samples)
    )
    return result


def ks_from_samples(samples1, samples2):
    """
    Computes the Kolmogorov-Smirnov distance (https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
    between a two empirical distributions defined by sets of samples.
    The piece-wise linear cdf is defined through cdf_grid and cdf_values.

    Parameters
    ----------
    samples1, samples2
        Samples from the two distributions
    """
    x_all = np.concatenate([samples1, samples2])
    x_all = np.sort(x_all)
    _, cdf1 = ecdf(samples1, x_all)
    _, cdf2 = ecdf(samples2, x_all)
    D = np.absolute(cdf1 - cdf2)
    return np.max(D)


def sample_dir_mult(alpha, n_trials, n_samples):
    """
    Generate samples from the Dirichlet-Multinomial distribution
    """
    if n_trials == 1:
        p = np.random.dirichlet(alpha)
        return np.random.multinomial(n_trials, p)

    p_array = np.random.dirichlet(alpha, n_samples)
    samples = np.array([np.random.multinomial(n_trials, p) for p in p_array])
    return samples
