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


import pytest

from gluonts.ext.r_forecast.util import (
    interval_to_quantile_level,
    quantile_to_interval_level,
)


@pytest.mark.parametrize(
    "quantile_level", [0.5, 0.15, 0.99, 0.4, 0.85, 0.35, 0.3, 0.5, 0.95, 0.13]
)
def test_quantile_to_interval_level_and_back(quantile_level: float):
    interval_level, side = quantile_to_interval_level(quantile_level)
    output = interval_to_quantile_level(interval_level, side)
    assert output == quantile_level
