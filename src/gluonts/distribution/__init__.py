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

# Relative imports
from . import bijection
from .binned import Binned, BinnedOutput
from .box_cox_transform import (
    BoxCoxTransformOutput,
    InverseBoxCoxTransformOutput,
)
from .categorical import Categorical, CategoricalOutput
from .dirichlet import Dirichlet, DirichletOutput
from .dirichlet_multinomial import (
    DirichletMultinomial,
    DirichletMultinomialOutput,
)
from .distribution import Distribution
from .distribution_output import DistributionOutput
from .gaussian import Gaussian, GaussianOutput
from .beta import Beta, BetaOutput
from .laplace import Laplace, LaplaceOutput
from .lowrank_multivariate_gaussian import (
    LowrankMultivariateGaussian,
    LowrankMultivariateGaussianOutput,
)
from .mixture import MixtureDistribution, MixtureDistributionOutput
from .multivariate_gaussian import (
    MultivariateGaussian,
    MultivariateGaussianOutput,
)
from .neg_binomial import NegativeBinomial, NegativeBinomialOutput
from .piecewise_linear import (
    PiecewiseLinear,
    PiecewiseLinearOutput,
    TransformedPiecewiseLinear,
)
from .student_t import StudentT, StudentTOutput
from .transformed_distribution import TransformedDistribution
from .transformed_distribution_output import TransformedDistributionOutput
from .uniform import Uniform, UniformOutput
from .logit_normal import LogitNormal, LogitNormalOutput
from .gamma import Gamma, GammaOutput
from .poisson import Poisson, PoissonOutput

__all__ = [
    "Distribution",
    "DistributionOutput",
    "StudentTOutput",
    "StudentT",
    "GammaOutput",
    "Gamma",
    "BetaOutput",
    "Beta",
    "GaussianOutput",
    "Gaussian",
    "LaplaceOutput",
    "Laplace",
    "MultivariateGaussian",
    "MultivariateGaussianOutput",
    "LowrankMultivariateGaussian",
    "LowrankMultivariateGaussianOutput",
    "MixtureDistributionOutput",
    "MixtureDistribution",
    "NegativeBinomialOutput",
    "NegativeBinomial",
    "UniformOutput",
    "Uniform",
    "Binned",
    "BinnedOutput",
    "PiecewiseLinear",
    "PiecewiseLinearOutput",
    "Poisson",
    "PoissonOutput",
    "TransformedPiecewiseLinear",
    "TransformedDistribution",
    "TransformedDistributionOutput",
    "InverseBoxCoxTransformOutput",
    "BoxCoxTransformOutput",
    "bijection",
    "Dirichlet",
    "DirichletOutput",
    "DirichletMultinomial",
    "DirichletMultinomialOutput",
    "Categorical",
    "CategoricalOutput",
    "LogitNormal",
    "LogitNormalOutput",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
