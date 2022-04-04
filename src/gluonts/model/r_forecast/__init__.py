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

from ._predictor import (
    RBasePredictor,
    R_IS_INSTALLED,
    RPY2_IS_INSTALLED,
)
from ._univariate_predictor import (
    SUPPORTED_UNIVARIATE_METHODS,
    RForecastPredictor,
    UNIVARIATE_POINT_FORECAST_METHODS,
    UNIVARIATE_QUANTILE_FORECAST_METHODS,
    UNIVARIATE_SAMPLE_FORECAST_METHODS,
)
from ._hierarchical_predictor import (
    HIERARCHICAL_POINT_FORECAST_METHODS,
    HIERARCHICAL_SAMPLE_FORECAST_METHODS,
    RHierarchicalForecastPredictor,
    SUPPORTED_HIERARCHICAL_METHODS,
)

__all__ = [
    "HIERARCHICAL_POINT_FORECAST_METHODS",
    "HIERARCHICAL_SAMPLE_FORECAST_METHODS",
    "RBasePredictor",
    "RHierarchicalForecastPredictor",
    "RForecastPredictor",
    "R_IS_INSTALLED",
    "RPY2_IS_INSTALLED",
    "SUPPORTED_HIERARCHICAL_METHODS",
    "SUPPORTED_UNIVARIATE_METHODS",
    "UNIVARIATE_POINT_FORECAST_METHODS",
    "UNIVARIATE_QUANTILE_FORECAST_METHODS",
    "UNIVARIATE_SAMPLE_FORECAST_METHODS",
]
