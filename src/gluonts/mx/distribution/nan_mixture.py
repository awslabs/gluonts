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
from .distribution import Distribution, _expand_param, _index_tensor, getF, nans_like
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
    def __init__(
            self, nan_prob: Tensor, distribution: Distribution, F=None
    ) -> None:
        F = F if F else getF(nan_prob)
        mixture_probs = F.concat(nan_prob, 1 - nan_prob, dim=-1)
        super().__init__(mixture_probs=mixture_probs, components=[distribution,
                                                                  Deterministic(value=nans_like(nan_prob))],
                         F=F)

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        log_mix_weights = F.log(self.mixture_probs)

        # compute log probabilities of components
        component_log_likelihood = F.stack(
            *[c.log_prob(x) for c in self.components], axis=-1
        )

        # log likelihood of first component for NaN is -inf and not not NaN
        minus_inf = -component_log_likelihood.ones_like() / 0.
        component_log_likelihood = F.where(F.contrib.isnan(component_log_likelihood),
                                           minus_inf,
                                           component_log_likelihood
                                           )
        # compute mixture log probability by log-sum-exp
        summands = F.sum(F.stack(log_mix_weights, component_log_likelihood, axis=-1), axis=-1)
        max_val = F.max_axis(summands, axis=-1, keepdims=True)

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
