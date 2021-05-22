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

from typing import List, Optional, Tuple, Union

import numpy as np
from mxnet import autograd

from gluonts.mx import Tensor
from gluonts.mx.distribution.bijection import AffineTransformation, Bijection
from gluonts.mx.distribution.distribution import Distribution, getF
from gluonts.mx.distribution.distribution_output import DistributionOutput
from gluonts.mx.distribution.transformed_distribution import (
    TransformedDistribution,
    sum_trailing_axes,
)


class TPPDistribution(Distribution):
    """
    Distribution used in temporal point processes.

    This class must implement new methods log_intensity, log_survival
    that are necessary for computing log-likelihood of TPP realizations. Also,
    sample_conditional is necessary for sampling TPPs.

    """

    def log_intensity(self, x: Tensor) -> Tensor:
        r"""
        Logarithm of the intensity (a.k.a. hazard) function.

        The intensity is defined as :math:`\lambda(x) = p(x) / S(x)`.
        """
        raise NotImplementedError()

    def log_survival(self, x: Tensor) -> Tensor:
        r"""
        Logarithm of the survival function `\log S(x) = \log(1 - CDF(x))`.
        """
        raise NotImplementedError()

    def log_prob(self, x: Tensor) -> Tensor:
        return self.log_intensity(x) + self.log_survival(x)

    def cdf(self, y: Tensor) -> Tensor:
        return 1.0 - self.log_survival(y).exp()

    def sample(
        self,
        num_samples=None,
        dtype=np.float32,
        lower_bound: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError()


class TPPTransformedDistribution(TransformedDistribution):
    """
    TransformedDistribution used in temporal point processes.

    This class must implement new methods log_intensity, log_survival
    that are necessary for computing log-likelihood of TPP realizations. Also,
    sample_conditional is necessary for sampling TPPs.

    Additionally, the sequence of transformations passed to the constructor
    must be increasing.

    """

    # Necessary for Mypy to understand that base_distribution is TPPDistribution
    base_distribution: TPPDistribution

    def __init__(
        self, base_distribution: TPPDistribution, transforms: List[Bijection]
    ) -> None:
        self.base_distribution = base_distribution
        self.transforms = transforms
        self._check_signs(transforms)
        self.is_reparameterizable = self.base_distribution.is_reparameterizable

        # use these to cache shapes and avoid recomputing all steps
        # the reason we cannot do the computations here directly
        # is that this constructor would fail in mx.symbol mode
        self._event_dim: Optional[int] = None
        self._event_shape: Optional[Tuple] = None
        self._batch_shape: Optional[Tuple] = None

    def _check_signs(self, transforms):
        """
        Make sure that the transformations are all increasing.

        This condition significantly simplifies the log_survival and
        sample_conditional functions.
        """
        sign = 1.0
        for t in transforms:
            sign = sign * t.sign
        if (sign != 1.0).asnumpy().any():
            raise ValueError("The transformations must be increasing.")

    def log_intensity(self, y: Tensor) -> Tensor:
        r"""
        Logarithm of the intensity (a.k.a. hazard) function.

        The intensity is defined as :math:`\lambda(y) = p(y) / S(y)`.
        """
        F = getF(y)
        lp = 0.0
        x = y
        for t in self.transforms[::-1]:
            x = t.f_inv(y)
            ladj = t.log_abs_det_jac(x, y)
            lp -= sum_trailing_axes(F, ladj, self.event_dim - t.event_dim)
            y = x
        return self.base_distribution.log_intensity(x) + lp

    def log_survival(self, y: Tensor) -> Tensor:
        r"""
        Logarithm of the survival function :math:`\log S(y) = \log(1 - CDF(y))`.
        """
        x = y
        for t in self.transforms[::-1]:
            x = t.f_inv(x)
        return self.base_distribution.log_survival(x)

    def cdf(self, y: Tensor) -> Tensor:
        return 1.0 - self.log_survival(y).exp()

    def sample(
        self,
        num_samples=None,
        dtype=np.float32,
        lower_bound: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Draw samples from the distribution.


        Parameters
        ----------
        num_samples
            Number of samples to generate.
        dtype
            Data type of the generated samples.
        lower_bound
            If None, generate samples as usual. If lower_bound is provided,
            all generated samples will be larger than the specified values.
            That is, we sample from `p(x | x > lower_bound)`.
            Shape: `(*batch_size)`

        Returns
        -------
        x
            Transformed samples drawn from the base distribution.
            Shape: `(num_samples, *batch_size)`

        """
        with autograd.pause():
            if lower_bound is not None:
                z = lower_bound
                for t in self.transforms[::-1]:
                    z = t.f_inv(z)
                lower_bound = z
            x = self.base_distribution.sample(
                num_samples=num_samples, lower_bound=lower_bound
            )
            # Apply transformations to the sample
            for t in self.transforms:
                x = t.f(x)
        return x.astype(dtype)


class TPPDistributionOutput(DistributionOutput):
    r"""
    Class to construct a distribution given the output of a network.

    Two differences compared to the base class DistributionOutput:
    1. Location param cannot be specified (all distributions must start at 0).
    2. The return type is either TPPDistribution or TPPTransformedDistribution.
    """
    distr_cls: type

    def distribution(
        self,
        distr_args,
        loc=None,
        scale: Optional[Tensor] = None,
    ) -> Union[TPPDistribution, TPPTransformedDistribution]:
        r"""
        Construct the associated distribution, given the collection of
        constructor arguments and, optionally, a scale tensor.

        Parameters
        ----------
        distr_args
            Constructor arguments for the underlying TPPDistribution type.
        loc
            Location parameter, specified here for compatibility with the
            superclass. Should never be specified.
        scale
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        """
        if loc is not None:
            raise ValueError(
                "loc should never be used for TPPDistributionOutput"
            )
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TPPTransformedDistribution(
                distr, [AffineTransformation(scale=scale)]
            )
