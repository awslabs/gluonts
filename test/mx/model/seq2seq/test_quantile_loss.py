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

# Third-party imports
from mxnet import nd
import pytest
import numpy as np
from pandas import Period

# First-party imports
from gluonts.mx.block.quantile_output import QuantileLoss, crps_weights_pwl
from gluonts.model.forecast import QuantileForecast


@pytest.mark.parametrize(
    "quantile_weights, correct_qt_loss",
    [
        (None, [1.0, 1.8]),
        ([0.5, 0.5], 1.4),
    ],
)
def test_compute_quantile_loss(quantile_weights, correct_qt_loss) -> None:
    y_true = nd.ones(shape=(10, 10, 10))
    y_pred = nd.zeros(shape=(10, 10, 10, 2))

    quantiles = [0.5, 0.9]

    loss = QuantileLoss(quantiles, quantile_weights)
    tol = 1e-5
    if not quantile_weights:
        for idx, q in enumerate(quantiles):
            assert (
                nd.mean(
                    loss.compute_quantile_loss(
                        nd.ndarray, y_true, y_pred[:, :, :, idx], q
                    )
                )
                - correct_qt_loss[idx]
                < tol
            ), f"computing quantile loss at quantile {q} fails!"
    else:
        assert (
            nd.mean(loss(y_true, y_pred)) - correct_qt_loss < tol
        ), "computing weighted quantile loss fails!"


@pytest.mark.parametrize(
    "quantiles, true_quantile_weight",
    [
        ([], []),
        ([0.5], [1]),
        ([0.5, 0.9], [0.2, 0.2]),
        ([0.1, 0.5, 0.9], [0.2, 0.4, 0.2]),
        ([0.3, 0.5, 0.9], [0.1, 0.3, 0.2]),
        (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
        ),
    ],
)
def test_crps_pwl_quantile_weights(quantiles, true_quantile_weight) -> None:
    assert len(quantiles) == len(true_quantile_weight), (
        f"length quantiles {quantiles} "
        f"and quantile_weights {true_quantile_weight} "
        "do not match."
    )
    tol = 1e-5
    quantile_weights = crps_weights_pwl(quantiles)
    assert (
        sum(
            abs(quantile_weights[i] - true_quantile_weight[i])
            for i in range(len(quantiles))
        )
        < tol
    ), "inaccurate computation of quantile weights"


@pytest.mark.parametrize(
    "quantile_predictions, inference_quantiles, inferred_quantile_predictions",
    [
        (
            {
                "0.5": np.array([0.0, 0.0]),
            },
            [0.1, 0.5, 0.9],
            {
                "0.1": np.array([np.nan, np.nan]),
                "0.5": np.array([0.0, 0.0]),
                "0.9": np.array([np.nan, np.nan]),
            },
        ),
        (
            {
                "0.1": np.array([-0.4, -0.8]),
                "0.5": np.array([0.0, 0.0]),
                "0.9": np.array([0.4, 0.8]),
            },
            [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            {
                "0.01": np.array([-0.97227044, -1.94454088]),
                "0.1": np.array([-0.4, -0.8]),
                "0.3": np.array([-0.2, -0.4]),
                "0.5": np.array([0.0, 0.0]),
                "0.7": np.array([0.2, 0.4]),
                "0.9": np.array([0.4, 0.8]),
                "0.99": np.array([0.97227044, 1.94454088]),
            },
        ),
    ],
)
def test_infer_quantile_forecast(
    quantile_predictions,
    inference_quantiles,
    inferred_quantile_predictions,
):
    tol = 1e-5
    forecast_keys = []
    output = []
    for key, value in quantile_predictions.items():
        forecast_keys.append(key)
        output.append(value)
    output = np.array(output)
    quantile_forecast = QuantileForecast(
        output,
        start_date=Period("01-01-2019 04:00:00", freq="h"),
        forecast_keys=forecast_keys,
    )
    if len(forecast_keys) == 1:
        for q in inference_quantiles:
            if forecast_keys[0] == str(q):
                assert (
                    sum(
                        inferred_quantile_predictions[str(q)]
                        - quantile_forecast.quantile(q)
                    )
                    < tol
                ), "infer_quantile_forecast failed for singleton quantile."

    else:
        assert (
            sum(
                sum(
                    inferred_quantile_predictions[str(q)]
                    - quantile_forecast.quantile(q)
                )
                for q in inference_quantiles
            )
            < tol
        ), "infer_quantile_forecast failed."
