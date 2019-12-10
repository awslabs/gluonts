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
from pathlib import Path


# Default GluonTS version for when the framework version is not specified.
# This is no longer updated so as to not break existing workflows.
GLUONTS_VERSION = "0.4.1"

# Framework related
FRAMEWORK_NAME = "GluonTS"
LOWEST_MMS_VERSION = "1.4"
LOWEST_SCRIPT_MODE_VERSION = "0", "4", "1"
LATEST_GLUONTS_VERSION = "0.4.1"
PYTHON_VERSION = "py3"

# Training related
ENTRY_POINTS_FOLDER = Path(__file__).parent.resolve() / "entry_point_scripts"
TRAIN_SCRIPT = "train_entry_point.py"
MONITORED_METRICS = "mean_wQuantileLoss", "ND", "RMSE"
NUM_SAMPLES = 100
QUANTILES = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
