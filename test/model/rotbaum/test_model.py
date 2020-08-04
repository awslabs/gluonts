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
from gluonts.evaluation.backtest import backtest_metrics

def test_accuracy(accuracy_test, dsinfo):

    predictor = TreePredictor(
        context_length=2, prediction_length=dsinfo["prediction_length"], freq = dsinfo["freq"]
    )
    predictor(dsinfo.train_ds)


    agg_metrics, item_metrics = backtest_metrics(
            test_dataset=dsinfo.test_ds,
            predictor=predictor,
            evaluator=Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=0),
        )

    if dsinfo["name"] == "constant":
        for q in [0.1, 0.5, 0.9]:
            assert agg_metrics[f"wQuantileLoss[{q}]"] == 0
    if dsinfo["name"] == "synthetic":
        assert 1.1 < agg_metrics[f"wQuantileLoss[0.5]"] < 1.2
        accuracy = 10.0
        assert agg_metrics["ND"] <= accuracy