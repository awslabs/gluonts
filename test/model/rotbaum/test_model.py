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
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

def test_accuracy(accuracy_test, dsinfo):

    predictor = TreePredictor(
        context_length=2, prediction_length=dsinfo["prediction_length"], freq = dsinfo["freq"]
    )
    predictor(dsinfo.train_ds)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dsinfo.test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=1,  # number of sample paths we want for evaluation
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=0)
    record, item_metrics = evaluator(iter(ts_it), iter(forecast_it), num_series=len(dsinfo.test_ds))

    if dsinfo["name"] == "constant":
        for q in [0.1, 0.5, 0.9]:
            print(record[f"wQuantileLoss[{q}]"])
            assert record[f"wQuantileLoss[{q}]"] == 0
    if dsinfo["name"] == "synthetic":
        print(record[f"wQuantileLoss[0.5]"])
        assert 1.1 < record[f"wQuantileLoss[0.5]"] < 1.2