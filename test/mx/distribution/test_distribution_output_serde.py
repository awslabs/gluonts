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

import mxnet as mx
import pytest

from gluonts.core.serde import encode, decode, dump_json
from gluonts.mx.model.tpp.distribution import (
    LoglogisticOutput,
    WeibullOutput,
)
from gluonts.mx.distribution import (
    BetaOutput,
    CategoricalOutput,
    DeterministicOutput,
    DistributionOutput,
    DirichletMultinomialOutput,
    DirichletOutput,
    EmpiricalDistributionOutput,
    EmpiricalDistribution,
    GammaOutput,
    GaussianOutput,
    GenParetoOutput,
    LaplaceOutput,
    LogitNormalOutput,
    LowrankMultivariateGaussianOutput,
    MultivariateGaussianOutput,
    NegativeBinomialOutput,
    OneInflatedBetaOutput,
    PiecewiseLinearOutput,
    PoissonOutput,
    StudentTOutput,
    UniformOutput,
    ZeroAndOneInflatedBetaOutput,
    ZeroInflatedBetaOutput,
    ZeroInflatedNegativeBinomialOutput,
    ZeroInflatedPoissonOutput,
)


@pytest.mark.parametrize(
    "distr_output",
    [
        BetaOutput(),
        CategoricalOutput(num_cats=3),
        DeterministicOutput(value=42.0),
        DirichletMultinomialOutput(dim=3, n_trials=5),
        DirichletOutput(dim=4),
        EmpiricalDistributionOutput(
            num_samples=10, distr_output=GaussianOutput()
        ),
        GammaOutput(),
        GaussianOutput(),
        GenParetoOutput(),
        LaplaceOutput(),
        LogitNormalOutput(),
        LoglogisticOutput(),
        LowrankMultivariateGaussianOutput(dim=5, rank=2),
        MultivariateGaussianOutput(dim=4),
        NegativeBinomialOutput(),
        OneInflatedBetaOutput(),
        PiecewiseLinearOutput(num_pieces=10),
        PoissonOutput(),
        StudentTOutput(),
        UniformOutput(),
        WeibullOutput(),
        ZeroAndOneInflatedBetaOutput(),
        ZeroInflatedBetaOutput(),
        ZeroInflatedNegativeBinomialOutput(),
        ZeroInflatedPoissonOutput(),
    ],
)
def test_distribution_output_serde(distr_output: DistributionOutput):
    distr_output_copy = decode(encode(distr_output))

    assert isinstance(distr_output_copy, type(distr_output))
    assert dump_json(distr_output_copy) == dump_json(distr_output)
