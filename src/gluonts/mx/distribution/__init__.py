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

from . import bijection
from .beta import Beta, BetaOutput
from .binned import Binned, BinnedOutput
from .box_cox_transform import (
    BoxCoxTransformOutput,
    InverseBoxCoxTransformOutput,
)
from .categorical import Categorical, CategoricalOutput
from .deterministic import Deterministic, DeterministicOutput
from .dirichlet import Dirichlet, DirichletOutput
from .dirichlet_multinomial import (
    DirichletMultinomial,
    DirichletMultinomialOutput,
)
from .distribution import Distribution
from .distribution_output import DistributionOutput
from .gamma import Gamma, GammaOutput
from .gaussian import Gaussian, GaussianOutput
from .genpareto import GenPareto, GenParetoOutput
from .inflated_beta import (
    OneInflatedBeta,
    OneInflatedBetaOutput,
    ZeroAndOneInflatedBeta,
    ZeroAndOneInflatedBetaOutput,
    ZeroInflatedBeta,
    ZeroInflatedBetaOutput,
)
from .laplace import Laplace, LaplaceOutput, LaplaceFixedVarianceOutput
from .logit_normal import LogitNormal, LogitNormalOutput
from .lowrank_multivariate_gaussian import (
    LowrankMultivariateGaussian,
    LowrankMultivariateGaussianOutput,
)
from .mixture import MixtureDistribution, MixtureDistributionOutput
from .multivariate_gaussian import (
    MultivariateGaussian,
    MultivariateGaussianOutput,
)
from .nan_mixture import NanMixture, NanMixtureOutput
from .neg_binomial import (
    NegativeBinomial,
    NegativeBinomialOutput,
    ZeroInflatedNegativeBinomialOutput,
)
from .piecewise_linear import (
    FixedKnotsPiecewiseLinearOutput,
    PiecewiseLinear,
    PiecewiseLinearOutput,
    TransformedPiecewiseLinear,
)
from .poisson import Poisson, PoissonOutput, ZeroInflatedPoissonOutput
from .student_t import StudentT, StudentTOutput
from .transformed_distribution import TransformedDistribution
from .transformed_distribution_output import TransformedDistributionOutput
from .uniform import Uniform, UniformOutput

__all__ = [
    "Distribution",
    "DistributionOutput",
    "StudentTOutput",
    "StudentT",
    "GammaOutput",
    "Gamma",
    "BetaOutput",
    "Beta",
    "ZeroAndOneInflatedBeta",
    "ZeroAndOneInflatedBetaOutput",
    "ZeroInflatedBeta",
    "ZeroInflatedBetaOutput",
    "OneInflatedBeta",
    "OneInflatedBetaOutput",
    "GenParetoOutput",
    "GenPareto",
    "GaussianOutput",
    "Gaussian",
    "LaplaceOutput",
    "LaplaceFixedVarianceOutput",
    "Laplace",
    "MultivariateGaussian",
    "MultivariateGaussianOutput",
    "LowrankMultivariateGaussian",
    "LowrankMultivariateGaussianOutput",
    "MixtureDistributionOutput",
    "MixtureDistribution",
    "NanMixture",
    "NanMixtureOutput",
    "NegativeBinomialOutput",
    "ZeroInflatedNegativeBinomialOutput",
    "NegativeBinomial",
    "UniformOutput",
    "Uniform",
    "Binned",
    "BinnedOutput",
    "PiecewiseLinear",
    "PiecewiseLinearOutput",
    "Poisson",
    "PoissonOutput",
    "ZeroInflatedPoissonOutput",
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
    "Deterministic",
    "DeterministicOutput",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
