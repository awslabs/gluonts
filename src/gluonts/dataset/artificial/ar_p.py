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

import warnings
from typing import Optional

import numpy as np

try:
    import numba
    from numba import njit
except:
    warnings.warn(
        "Could not import numba. ar_p will be slower for long series (len > 1000)."
    )
    # TODO: support parameters
    njit = lambda f: f


@njit
def ar_p(
    phi: np.ndarray,
    sigma: float,
    length: int,
    xhist: np.ndarray,
    c: float = 0.0,
    noise: Optional[np.ndarray] = None,
):
    """
    Generate samples from an AR(p) process.
    Parametrized as in

    https://en.wikipedia.org/wiki/Autoregressive_model#Graphs_of_AR(p)_processes

    Parameters
    ----------
    phi
        model parameters. This should be a vector of length p
    sigma
        noise amplitude
    c
        constant
    length
        number of steps to sample
    xhist
        initial condition. This should be a vector of length p
    noise
        An optional vector of noise samples to use. If provided it should have len `length`.
        If it is not provided, samples from a standard normal are used.
    """
    phi_ = np.asarray(phi, dtype=np.float64)
    xhist_ = np.asarray(xhist, dtype=np.float64)
    assert len(phi_) > 0
    assert len(xhist_) == len(phi_)
    if noise is not None:
        noise_ = np.asarray(noise, dtype=np.float64)
        assert len(noise_) == length
    else:
        noise_ = np.random.randn(length).astype(np.float64)
    p = len(xhist)
    x = np.zeros(length + p, dtype=np.float64)
    x[: len(xhist_)] = xhist_[:]
    phi_ = np.asarray(phi, dtype=np.float64)
    for t in range(p, length + p):
        u = 0.0
        for i in range(0, p):
            u += phi_[i] * x[t - i - 1]
        x[t] = c + u + sigma * noise_[t - p]
    return x[p:]
