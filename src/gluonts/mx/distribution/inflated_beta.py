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
from .distribution import _expand_param, _index_tensor, getF
from .mixture import MixtureDistribution, MixtureDistributionOutput
from .deterministic import Deterministic, DeterministicOutput
from .beta import Beta, BetaOutput


class InflatedBeta(MixtureDistribution):
    r"""
    A zero and/or one Inflated Beta distribution.
    A mixture of one or two deterministic distributions centered at 0 and/or 1 and a Beta distribution.

    Parameters
    ----------
    zero_probability
        A tensor of probabilities.
        The entries should all be non-negative and smaller than 1-one_probability.

    one_probability
        A tensor of probabilities.
        The entries should all be non-negative and smaller than 1-zero_probability.

    alpha
        Tensor containing the alpha shape parameters of the Beta Distribution, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta shape parameters of the Beta Distribution, of shape `(*batch_shape, *event_shape)`.
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    @validated()
    def __init__(
        self,
        zero_probability: Tensor,
        one_probability: Tensor,
        alpha: Tensor,
        beta: Tensor,
        F=None,
    ) -> None:

        # does this work so easily? dont the zeros make problems :S
        # better take in parameters, zero inflated = true
        F = getF(one_probability)

        mixture_probs = F.stack(
            1 - one_probability - zero_probability,
            zero_probability,
            one_probability,
            axis=-1,
        )
        beta_distribution = Beta(alpha=alpha, beta=beta)
        super().__init__(
            mixture_probs=mixture_probs,
            components=[
                beta_distribution,
                Deterministic(value=F.zeros_like(zero_probability)),
                Deterministic(value=F.ones_like(one_probability)),
            ],
        )

    @property
    def F(self):
        return getF(self.mixture_probs)

    def __getitem__(self, item):
        return MixtureDistribution(
            _index_tensor(self.mixture_probs, item),
            [c[item] for c in self.components],
        )

    @property
    def batch_shape(self) -> Tuple:
        return self.components[0].batch_shape

    @property
    def event_shape(self) -> Tuple:
        return self.components[0].event_shape

    @property
    def event_dim(self) -> int:
        return self.components[0].event_dim

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        log_mix_weights = F.log(self.mixture_probs)

        # compute log probabilities of components
        component_log_likelihood = F.stack(
            *[c.log_prob(x) for c in self.components], axis=-1
        )

        # compute mixture log probability by log-sum-exp
        summands = log_mix_weights + component_log_likelihood
        max_val = F.max_axis(summands, axis=-1, keepdims=True)
        sum_exp = F.sum(
            F.exp(F.broadcast_minus(summands, max_val)), axis=-1, keepdims=True
        )
        log_sum_exp = F.log(sum_exp) + max_val

        return log_sum_exp.squeeze(axis=-1)

    @property
    def mean(self) -> Tensor:
        F = self.F
        mean_values = F.stack(*[c.mean for c in self.components], axis=-1)
        return F.sum(
            F.broadcast_mul(mean_values, self.mixture_probs, axis=-1), axis=-1
        )

    def cdf(self, x: Tensor) -> Tensor:
        F = self.F
        cdf_values = F.stack(*[c.cdf(x) for c in self.components], axis=-1)
        erg = F.sum(
            F.broadcast_mul(cdf_values, self.mixture_probs, axis=-1), axis=-1
        )
        return erg

    @property
    def stddev(self) -> Tensor:
        F = self.F
        sq_mean_values = F.square(
            F.stack(*[c.mean for c in self.components], axis=-1)
        )
        sq_std_values = F.square(
            F.stack(*[c.stddev for c in self.components], axis=-1)
        )

        return F.sqrt(
            F.sum(
                F.broadcast_mul(
                    sq_mean_values + sq_std_values, self.mixture_probs, axis=-1
                ),
                axis=-1,
            )
            - F.square(self.mean)
        )

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        F = self.F
        samples_list = [c.sample(num_samples, dtype) for c in self.components]
        samples = F.stack(*samples_list, axis=-1)

        mixture_probs = _expand_param(self.mixture_probs, num_samples)

        idx = F.random.multinomial(mixture_probs)

        for _ in range(self.event_dim):
            idx = idx.expand_dims(axis=-1)
        idx = idx.broadcast_like(samples_list[0])

        selected_samples = F.pick(data=samples, index=idx, axis=-1)

        return selected_samples


class MixtureArgs(gluon.HybridBlock):
    def __init__(
        self,
        distr_outputs: List[DistributionOutput],
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_components = len(distr_outputs)
        self.component_projections: List[gluon.HybridBlock] = []
        with self.name_scope():
            self.proj_mixture_probs = gluon.nn.HybridSequential()
            self.proj_mixture_probs.add(
                gluon.nn.Dense(
                    self.num_components, prefix=f"{prefix}_pi_", flatten=False
                )
            )
            self.proj_mixture_probs.add(gluon.nn.HybridLambda("softmax"))

            for k, do in enumerate(distr_outputs):
                self.component_projections.append(
                    do.get_args_proj(prefix=str(k))
                )
                self.register_child(self.component_projections[-1])

    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor, ...]:
        mixture_probs = self.proj_mixture_probs(x)
        component_args = [c_proj(x) for c_proj in self.component_projections]
        return tuple([mixture_probs] + component_args)


class MixtureDistributionOutput(DistributionOutput):
    @validated()
    def __init__(self, distr_outputs: List[DistributionOutput]) -> None:
        self.num_components = len(distr_outputs)
        self.distr_outputs = distr_outputs

    def get_args_proj(self, prefix: Optional[str] = None) -> MixtureArgs:
        return MixtureArgs(self.distr_outputs, prefix=prefix)

    # Overwrites the parent class method.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        **kwargs,
    ) -> MixtureDistribution:
        mixture_probs = distr_args[0]
        component_args = distr_args[1:]
        return MixtureDistribution(
            mixture_probs=mixture_probs,
            components=[
                do.distribution(args, loc=loc, scale=scale)
                for do, args in zip(self.distr_outputs, component_args)
            ],
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_outputs[0].event_shape
