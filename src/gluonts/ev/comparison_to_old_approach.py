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
    "MSE": 32252.758071166663,
    "abs_error": 141563.0,
    "abs_target_sum": 1389123.0,
    "abs_target_mean": 578.80125,
    "seasonal_error": 74.21931779257851,
    "MASE": 0.6872127982322291,
    "MAPE": 0.11397109846527713,
    "sMAPE": 0.1166518161631813,
    "MSIS": 4.652401797678681,
    "QuantileLoss[0.1]": 61223.80000000001,
    "Coverage[0.1]": 0.14541666666666667,
    "QuantileLoss[0.5]": 141563.0,
    "Coverage[0.5]": 0.5570833333333334,
    "QuantileLoss[0.9]": 69770.19999999998,
    "Coverage[0.9]": 0.9029166666666667,
    "RMSE": 179.59052890162852,
    "NRMSE": 0.3102801331227749,
    "ND": 0.10190818235678194,
    "wQuantileLoss[0.1]": 0.044073706935958884,
    "wQuantileLoss[0.5]": 0.10190818235678194,
    "wQuantileLoss[0.9]": 0.05022607789231046,
    "mean_absolute_QuantileLoss": 90852.33333333333,
    "mean_wQuantileLoss": 0.06540265572835043,
    "MAE_Coverage": 0.03513888888888891,
    "OWA": NaN
}

NEW RESULTS:
MSE: 32243.589991541678
abs_error: 141901.0
abs_target_sum: 1389123.0
abs_target_mean: 578.80125
MASE: 0.6862165457798155
MAPE: 0.11101808615099994
sMAPE: 0.11336362799976307
MSIS: 4.670753223398399
QuantileLoss[0.1]: 62895.79999999999
QuantileLoss[0.5]: 141901.0
QuantileLoss[0.9]: 70042.59999999996
wQuantileLoss[0.1]: 0.04527734405088677
wQuantileLoss[0.5]: 0.10215150134293363
wQuantileLoss[0.9]: 0.05042217283854631
Coverage[0.1]: 0.14416666666666667
Coverage[0.5]: 0.5683333333333334
Coverage[0.9]: 0.9029166666666667
RMSE: 179.56500213444065
NRMSE: 0.3102360303030456
ND: 0.1194727968653604
"""
