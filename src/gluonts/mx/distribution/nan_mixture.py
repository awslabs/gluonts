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

# Standard library imports
from typing import List, Optional, Tuple

import numpy as np

# Third-party imports
from mxnet import gluon

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor

# Relative imports
from .distribution import (
    Distribution,
    _expand_param,
    _index_tensor,
    getF,
    nans_like,
)
from .distribution_output import DistributionOutput
from .mixture import MixtureDistribution, MixtureDistributionOutput
from .deterministic import Deterministic, DeterministicOutput


class NanMixture(MixtureDistribution):
    r"""
    A mixture distribution of a NaN-valued Deterministic distribution and distribution

    Parameters
    ----------
    nan_prob
        A tensor of the probabilities of missing values. The entries should all be positive
        and smaller than 1. All axis should either coincide with the ones from the component distributions,
        or be 1 (in which case, the NaN probability is shared across the axis).
    distribution
        A Distribution object representing the Distribution of non-NaN values.
        Distributions can be of different types. Each component's support
        should be made of tensors of shape (..., d).
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, nan_prob: Tensor, distribution: Distribution) -> None:
        F = getF(nan_prob)
        mixture_probs = F.stack(1 - nan_prob, nan_prob, axis=-1)
        super().__init__(
            mixture_probs=mixture_probs,
            components=[
                distribution,
                Deterministic(value=nans_like(nan_prob)),
            ],
            F=F,
        )

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        # masking zero-valued mixture_probs to prevent NaN edge case gradients
        nonzero_mixture_probs = F.where(
            self.mixture_probs == 0,
            self.mixture_probs.ones_like(),
            self.mixture_probs,
        )

        log_mix_weights = F.where(
            self.mixture_probs == 0,
            -1.0 / self.mixture_probs.zeros_like(),
            F.log(nonzero_mixture_probs),
        )

        # TODO does the replacement value has to be in the support of the distribution? maybe sample from it
        # masking data NaN's with ones to prevent NaN gradients
        x_non_nan = F.where(x != x, F.ones_like(x), x)
        non_nan_comp_log_likelihood = F.where(
            x != x,
            -x.ones_like() / 0.0,
            self.components[0].log_prob(x_non_nan),
        )

        # compute log probabilities of components
        component_log_likelihood = F.stack(
            *[non_nan_comp_log_likelihood, self.components[1].log_prob(x)],
            axis=-1
        )

        # compute mixture log probability by log-sum-exp
        summands = log_mix_weights + component_log_likelihood
        max_val = F.max_axis(summands, axis=-1, keepdims=True)
        # catch edge case max_val = -inf
        max_val = F.where(max_val == -np.inf, max_val.zeros_like(), max_val)
        sum_exp = F.sum(
            F.exp(F.broadcast_minus(summands, max_val)), axis=-1, keepdims=True
        )

        log_sum_exp = F.log(sum_exp) + max_val
        return log_sum_exp.squeeze(axis=-1)


class NanMixtureOutput(MixtureDistributionOutput):
    @validated()
    def __init__(self, distr_output: DistributionOutput) -> None:
        nan_output = DeterministicOutput()
        super().__init__([distr_output, nan_output])
