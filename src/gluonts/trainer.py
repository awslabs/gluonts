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

import gluonts.mx.trainer

warnings.warn(
    "gluonts.trainer is deprecated. Use gluonts.mx.trainer instead.",
    DeprecationWarning,
    stacklevel=2,
)


sys.modules["gluonts.trainer"] = gluonts.mx.trainer

for submodule in (
    "_base",
    "learning_rate_scheduler",
    "model_averaging",
    "model_iteration_averaging",
):
    sys.modules[f"gluonts.trainer.{submodule}"] = importlib.import_module(
        f"gluonts.mx.trainer.{submodule}"
    )
