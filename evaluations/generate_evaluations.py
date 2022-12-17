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

"""
This example evaluates models in gluon-ts.
Evaluations are stored for each model/dataset in a json file and all results can then
be displayed with `show_results.py`.
"""
import json
import os
from pathlib import Path
from typing import Dict

from gluonts.dataset.repository.datasets import get_dataset, dataset_names
from gluonts.evaluation import backtest_metrics
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

metrics_persisted = ["mean_wQuantileLoss", "ND", "RMSE"]
datasets = dataset_names

Estimators = [
    SeasonalNaivePredictor,
    # model.simple_feedforward.SimpleFeedForwardEstimator,
    # model.deepar.DeepAREstimator,
    # model.NPTSPredictor,
    # model.seq2seq.MQCNNEstimator,
    # TransformerEstimator,
]

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def persist_evaluation(
    estimator_name: str,
    dataset: str,
    evaluation: Dict[str, float],
    evaluation_path: str = "./",
):
    """
    Saves an evaluation dictionary into `evaluation_path`
    """
    path = Path(evaluation_path) / dataset / f"{estimator_name}.json"

    os.makedirs(path.parent, exist_ok=True)

    evaluation = {
        m: v for m, v in evaluation.items() if m in metrics_persisted
    }
    evaluation["dataset"] = dataset
    evaluation["estimator"] = estimator_name

    with open(path, "w") as f:
        f.write(json.dumps(evaluation, indent=4, sort_keys=True))


if __name__ == "__main__":

    for dataset_name in datasets:
        for Estimator in Estimators:
            dataset = get_dataset(
                dataset_name=dataset_name,
                regenerate=False,
                path=Path("../datasets/"),
            )

            estimator = Estimator(
                prediction_length=dataset.metadata.prediction_length,
                freq=dataset.metadata.freq,
            )

            estimator_name = type(estimator).__name__

            print(f"evaluating {estimator_name} on {dataset_name}")

            agg_metrics, item_metrics = backtest_metrics(
                train_dataset=dataset.train,
                test_dataset=dataset.test,
                forecaster=estimator,
            )

            persist_evaluation(
                estimator_name=estimator_name,
                dataset=dataset_name,
                evaluation=agg_metrics,
                evaluation_path=dir_path,
            )
