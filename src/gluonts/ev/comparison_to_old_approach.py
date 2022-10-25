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


import json
from multiprocessing import freeze_support

from toolz import take

from gluonts.dataset.split import TestTemplate, OffsetSplitter
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.ev.evaluator import MetricGroup
from gluonts.ev.metrics import (
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    Coverage,
    SumAbsoluteError,
    SumQuantileLoss,
    WeightedSumQuantileLoss,
    mean_absolute_label,
    sum_absolute_label,
)


def main():
    dataset = get_dataset("electricity")

    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    dataset_test = list(take(100, dataset.test))

    test_template = TestTemplate(
        dataset=dataset_test,
        splitter=OffsetSplitter(offset=-prediction_length),
    )

    test_data = test_template.generate_instances(
        prediction_length=prediction_length
    )

    predictor = NPTSPredictor(prediction_length=prediction_length, freq=freq)

    # --- EVALUATION STARTS HERE ---
    # the old way
    forecast_it, ts_it = make_evaluation_predictions(dataset_test, predictor)
    tss = list(ts_it)
    forecasts = list(forecast_it)
    evaluator = Evaluator(quantiles=(0.1, 0.5, 0.9))
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print("OLD RESULTS:")
    print(json.dumps(agg_metrics, indent=4))

    # the new way
    quantile_levels = (0.1, 0.5, 0.9)

    aggregated_metrics = {
        "MSE": MSE(),
        "abs_error": SumAbsoluteError(),
        "abs_target_sum": sum_absolute_label,
        "abs_target_mean": mean_absolute_label,
        "MASE": MASE(),
        "MAPE": MAPE(),
        "sMAPE": SMAPE(),
        "MSIS": MSIS(),
        **{
            f"QuantileLoss[{q}]": SumQuantileLoss(q=q) for q in quantile_levels
        },
        **{
            f"wQuantileLoss[{q}]": WeightedSumQuantileLoss(q=q)
            for q in quantile_levels
        },
        **{f"Coverage[{q}]": Coverage(q=q) for q in quantile_levels},
        "RMSE": RMSE(),
        "NRMSE": NRMSE(),
        "ND": ND(),
    }

    metric_group = MetricGroup()
    metric_group.add_metrics(aggregated_metrics, axis=None)

    res = metric_group.evaluate(test_data, predictor)

    print("\nNEW RESULTS:")
    for name, value in res.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    freeze_support()
    main()


"""
OLD RESULTS:
{
    "MSE": 31870.83876720834,
    "abs_error": 140896.0,
    "abs_target_sum": 1389123.0,
    "abs_target_mean": 578.80125,
    "seasonal_error": 74.21931779257851,
    "MASE": 0.6946276981713059,
    "MAPE": 0.11611727679465304,
    "sMAPE": 0.11837255492400349,
    "MSIS": 4.657175423001957,
    "QuantileLoss[0.1]": 62127.00000000001,
    "Coverage[0.1]": 0.14291666666666666,
    "QuantileLoss[0.5]": 140896.0,
    "Coverage[0.5]": 0.5629166666666666,
    "QuantileLoss[0.9]": 68531.79999999999,
    "Coverage[0.9]": 0.8975000000000002,
    "RMSE": 178.5240565503942,
    "NRMSE": 0.3084375794806839,
    "ND": 0.10142802329239384,
    "wQuantileLoss[0.1]": 0.04472390133919027,
    "wQuantileLoss[0.5]": 0.10142802329239384,
    "wQuantileLoss[0.9]": 0.04933458016316769,
    "mean_absolute_QuantileLoss": 90518.26666666666,
    "mean_wQuantileLoss": 0.06516216826491726,
    "MAE_Coverage": 0.03611111111111104,
    "OWA": NaN
}

NEW RESULTS:
MSE: 32591.889823208327
abs_error: 141585.0
abs_target_sum: 1389123.0
abs_target_mean: 578.80125
MASE: 0.692172930083296
MAPE: 0.11479859845921345
sMAPE: 0.11430782972432606
MSIS: 4.629788317158182
QuantileLoss[0.1]: 31596.200000000004
QuantileLoss[0.5]: 70792.5
QuantileLoss[0.9]: 34470.499999999985
wQuantileLoss[0.1]: 0.022745430030314092
wQuantileLoss[0.5]: 0.05096200984362076
wQuantileLoss[0.9]: 0.02481457725485791
Coverage[0.1]: 0.14041666666666666
Coverage[0.5]: 0.5629166666666666
Coverage[0.9]: 0.8970833333333333
RMSE: 180.53224039824113
NRMSE: 0.311907136341259
ND: 0.11980707971864261
"""