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

from typing import Optional, Tuple

from mxnet import gluon

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import Deterministic
from .distribution import Distribution, getF, nans_like
from .distribution_output import DistributionOutput
from .mixture import MixtureDistribution


class NanMixture(MixtureDistribution):
    r"""
    A mixture distribution of a NaN-valued Deterministic distribution and Distribution

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

        F = getF(nan_prob)

        mixture_probs = F.stack(1 - nan_prob, nan_prob, axis=-1)
        super().__init__(
            mixture_probs=mixture_probs,
            components=[
                distribution,
                Deterministic(value=nans_like(nan_prob)),
            ],
        )

    @property
    def distribution(self):
        return self.components[0]

    @property
    def nan_prob(self):
        return self.mixture_probs.slice_axis(axis=-1, begin=1, end=2).squeeze(
            axis=-1
        )

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F

        # masking data NaN's with ones to prevent NaN gradients
        x_non_nan = F.where(x != x, F.ones_like(x), x)

        # calculate likelihood for values which are not NaN
        non_nan_dist_log_likelihood = F.where(
            x != x,
            -x.ones_like() / 0.0,
            self.components[0].log_prob(x_non_nan),
        )

        log_mix_weights = F.log(self.mixture_probs)

        # stack log probabilities of components
        component_log_likelihood = F.stack(
            *[non_nan_dist_log_likelihood, self.components[1].log_prob(x)],
            axis=-1,
        )
        # compute mixture log probability by log-sum-exp
        summands = log_mix_weights + component_log_likelihood
        max_val = F.max_axis(summands, axis=-1, keepdims=True)

        sum_exp = F.sum(
            F.exp(F.broadcast_minus(summands, max_val)), axis=-1, keepdims=True
        )

        log_sum_exp = F.log(sum_exp) + max_val
        return log_sum_exp.squeeze(axis=-1)


class NanMixtureArgs(gluon.HybridBlock):
    def __init__(
        self,
        distr_output: DistributionOutput,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.component_projection: gluon.HybridBlock
        with self.name_scope():
            self.proj_nan_prob = gluon.nn.HybridSequential()
            self.proj_nan_prob.add(
                gluon.nn.Dense(1, prefix=f"{prefix}_pi_", flatten=False)
            )
            self.proj_nan_prob.add(gluon.nn.HybridLambda("sigmoid"))

            self.component_projection = distr_output.get_args_proj()

            self.register_child(self.component_projection)

    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor, ...]:
        nan_prob = self.proj_nan_prob(x)
        component_args = self.component_projection(x)
        return tuple([nan_prob.squeeze(axis=-1), component_args])


class NanMixtureOutput(DistributionOutput):
    distr_cls: type = NanMixture

    @validated()
    def __init__(self, distr_output: DistributionOutput) -> None:
        self.distr_output = distr_output

    def get_args_proj(self, prefix: Optional[str] = None) -> NanMixtureArgs:
        return NanMixtureArgs(self.distr_output, prefix=prefix)

    # Overwrites the parent class method.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        **kwargs,
    ) -> MixtureDistribution:
        nan_prob = distr_args[0]
        component_args = distr_args[1]
        return NanMixture(
            nan_prob=nan_prob,
            distribution=self.distr_output.distribution(
                component_args, loc=loc, scale=scale
            ),
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_output.event_shape
