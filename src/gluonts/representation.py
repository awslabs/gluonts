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

import gluonts.mx.representation

warnings.warn(
    "gluonts.representation is deprecated. Use gluonts.mx.representation instead.",
    DeprecationWarning,
    stacklevel=2,
)


sys.modules["gluonts.representation"] = gluonts.mx.representation


for submodule in (
    "binning_helpers",
    "custom_binning",
    "dim_expansion",
    "discrete_pit",
    "embedding",
    "global_relative_binning",
    "hybrid_representation",
    "local_absolute_binning",
    "mean_scaling",
    "representation",
    "representation_chain",
):
    sys.modules[
        f"gluonts.representation.{submodule}"
    ] = importlib.import_module(f"gluonts.mx.representation.{submodule}")
