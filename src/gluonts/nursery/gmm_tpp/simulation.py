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
import scipy as sp


def thinning_sampler(rng, lamb, xmin=0, lamb_min=1e-10):
    """Ogata's Thinning algorithm for time-varying exponential distribution
    with monotone-decreasing intensity function
    """
    while lamb(xmin) > lamb_min:
        dx = -np.log(rng.rand()) / lamb(xmin)
        x = xmin + dx
        accept_rate = lamb(x) / lamb(xmin)

        if rng.rand() < accept_rate:
            return x
        xmin = x
    raise ValueError(
        f"require lamb({xmin})>{lamb_min} to guarantee cdf(infty)=1"
    )


def Hawkes(rng, background, kernel, xmin, xmax, N_max=1e6):
    """ requires int x kernel(x)<infty """
    X = []
    while len(X) < N_max:
        lamb = lambda x: background + np.sum(
            [kernel(x - xi) if x >= xi else 0 for xi in X]
        )
        x = thinning_sampler(rng, lamb, max(X + [xmin]))
        if x > xmax:  # out of range
            return X
        else:
            X.append(x)
    raise ValueError(f"N>{N_max}; check if int_0^infty kernel < inf")
