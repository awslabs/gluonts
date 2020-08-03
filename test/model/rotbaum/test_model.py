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
import numpy as np
from itertools import chain

# First-party imports
from gluonts.model.rotbaum import TreePredictor

# TODO: switch to using backtest_metrics rather than separate quantile_loss
# TODO: function.


def test_accuracy(accuracy_test, dsinfo):
    def quantile_loss(true, pred, quantile):
        denom = sum(np.abs(true))
        num = sum(
            [
                (1 - quantile) * abs(y_hat - y)
                if y_hat > y
                else quantile * abs(y_hat - y)
                for y_hat, y in zip(pred, true)
            ]
        )
        if denom != 0:
            return 2 * num / denom
        else:
            return None

    record = {}
    predictor = TreePredictor(
        context_length=2, prediction_length=dsinfo["prediction_length"]
    )
    predictor_instance = predictor(dsinfo.train_ds)
    data_for_pred = [
        {
            "start": ts["start"],
            "target": ts["target"][: -dsinfo["prediction_length"]],
        }
        for ts in dsinfo.test_ds
    ]
    data_true = [
        {
            "start": ts["start"],
            "target": ts["target"][-dsinfo["prediction_length"] :],
        }
        for ts in dsinfo.test_ds
    ]
    for quantile in [0.1, 0.5, 0.9]:
        predictions = list(predictor_instance.predict(data_for_pred))
        preds = [t.quantile(quantile) for t in predictions]
        record[f"quantile_loss_{quantile}"] = quantile_loss(
            list(chain(*[ts["target"] for ts in data_true])),
            list(chain(*preds)),
            quantile,
        )
    if dsinfo["name"] == "constant":
        for q in [0.1, 0.5, 0.9]:
            assert record[f"quantile_loss_{q}"] == 0
    if dsinfo["name"] == "synthetic":
        assert 1.1 < record[f"quantile_loss_{0.5}"] < 1.2
