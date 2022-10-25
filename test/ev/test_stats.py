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

import numpy as np
import pytest
from gluonts.ev.stats import (
    absolute_error,
    absolute_label,
    absolute_percentage_error,
    coverage,
    error,
    quantile_loss,
    squared_error,
    symmetric_absolute_percentage_error,
)

NAN = np.full(5, np.nan)
ZEROES = np.zeros(5)
# LINEAR = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
# EXPONENTIAL = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
# CONSTANT = np.array([0.4] * 5)


@pytest.mark.parametrize(
    "label, forecast, metrics",
    [
        (
            ZEROES,
            ZEROES,
            [
                (absolute_label, ZEROES, {}),
                (error, ZEROES, {"forecast_type": "mean"}),
                (absolute_error, ZEROES, {"forecast_type": "mean"}),
                (squared_error, ZEROES, {"forecast_type": "mean"}),
                (quantile_loss, ZEROES, {"q": 0.5}),
                (coverage, ZEROES, {"q": 0.5}),
                (absolute_percentage_error, NAN, {"forecast_type": "mean"}),
                (
                    symmetric_absolute_percentage_error,
                    NAN,
                    {"forecast_type": "mean"},
                ),
            ],
        ),
    ],
)
def test_metrics_without_seasonal_error(label, forecast, metrics):
    data = {"label": label, "0.5": forecast, "mean": forecast}

    for metric, expected, kwargs in metrics:
        np.testing.assert_almost_equal(metric(data, **kwargs), expected)


def test_metrics_with_seasonal_error():
    pass  # TODO
