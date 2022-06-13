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

from torch.distributions import (
    TransformedDistribution,
    AffineTransform,
    Distribution,
)


class AffineTransformed(TransformedDistribution):
    """
    Represents the distribution of an affinely transformed random variable.

    This is the distribution of ``Y = scale * X + loc``, where ``X`` is a
    random variable distributed according to ``base_distribution``.

    Parameters
    ----------
    base_distribution
        Original distribution
    loc
        Translation parameter of the affine transformation.
    scale
        Scaling parameter of the affine transformation.
    """

    def __init__(self, base_distribution: Distribution, loc=None, scale=None):

        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

        super().__init__(
            base_distribution, [AffineTransform(self.loc, self.scale)]
        )

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()
