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

import importlib
import sys
import warnings

import gluonts.mx.distribution

warnings.warn(
    "gluonts.distribution is deprecated. Use gluonts.mx.distribution instead.",
    DeprecationWarning,
    stacklevel=2,
)


sys.modules["gluonts.distribution"] = gluonts.mx.distribution


for submodule in (
    "beta",
    "bijection",
    "bijection_output",
    "binned",
    "box_cox_transform",
    "categorical",
    "dirichlet",
    "dirichlet_multinomial",
    "distribution",
    "distribution_output",
    "gamma",
    "gaussian",
    "laplace",
    "lds",
    "logit_normal",
    "lowrank_gp",
    "lowrank_multivariate_gaussian",
    "mixture",
    "multivariate_gaussian",
    "neg_binomial",
    "piecewise_linear",
    "poisson",
    "student_t",
    "transformed_distribution",
    "transformed_distribution_output",
    "uniform",
):
    sys.modules[f"gluonts.distribution.{submodule}"] = importlib.import_module(
        f"gluonts.mx.distribution.{submodule}"
    )
