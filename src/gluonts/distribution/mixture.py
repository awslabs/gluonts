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

# Third-party imports
from mxnet import gluon

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor

# Relative imports
from .distribution import Distribution, _expand_param, getF
from .distribution_output import DistributionOutput


class MixtureDistribution(Distribution):
    r"""
    A mixture distribution where each component is a Distribution.

    Paremeters
    ----------
    mixture_probs
        A tensor of mixing probabilities. The entries should all be positive
        and sum to 1 across the last dimension. Shape: (..., k), where k is
        the number of distributions to be mixed. All axis except the last one
        should either coincide with the ones from the component distributions,
        or be 1 (in which case, the mixing coefficient is shared across
        the axis).
    components
        A list of k Distribution objects representing the mixture components.
        Distributions can be of different types. Each component's support
        should be made of tensors of shape (..., d).
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    is_reparameterizable = False

    def __init__(
        self, mixture_probs: Tensor, components: List[Distribution], F=None
    ) -> None:

        self.F = F if F else getF(mixture_probs)
        # TODO: handle case with all components of the same type more efficiently when sampling
        # self.all_same = len(set(c.__class__.__name__ for c in components)) == 1
        self.mixture_probs = mixture_probs
        self.components = components

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
        stddev_values = F.stack(*[c.stddev for c in self.components], axis=-1)
        return F.sum(
            F.broadcast_mul(stddev_values, self.mixture_probs, axis=-1),
            axis=-1,
        )

    def sample(self, num_samples: Optional[int] = None) -> Tensor:
        samples_list = [c.sample(num_samples) for c in self.components]
        samples = self.F.stack(*samples_list, axis=-1)

        mixture_probs = _expand_param(self.mixture_probs, num_samples)

        idx = self.F.random.multinomial(mixture_probs)

        for _ in range(self.event_dim):
            idx = idx.expand_dims(axis=-1)
        idx = idx.broadcast_like(samples_list[0])

        selected_samples = self.F.pick(data=samples, index=idx, axis=-1)

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
                    self.num_components, prefix=f'{prefix}_pi_', flatten=False
                )
            )
            self.proj_mixture_probs.add(gluon.nn.HybridLambda('softmax'))

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
        self, distr_args, scale: Optional[Tensor] = None, **kwargs
    ) -> MixtureDistribution:
        mixture_probs = distr_args[0]
        component_args = distr_args[1:]
        return MixtureDistribution(
            mixture_probs=mixture_probs,
            components=[
                do.distribution(args, scale=scale)
                for do, args in zip(self.distr_outputs, component_args)
            ],
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_outputs[0].event_shape
