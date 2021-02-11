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
This example show how to intialize the network with parameters from a model that was previously trained.
"""

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import backtest_metrics, Evaluator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
import pandas as pd


if __name__ == "__main__":
    print(f"datasets available: {dataset_recipes.keys()}")

    # we pick m4_hourly as it only contains a few hundred time series
    dataset = get_dataset("m4_hourly", regenerate=False)

    # First train a model
    estimator = SimpleFeedForwardEstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(epochs=10, num_batches_per_epoch=10),
    )

    train1_output = estimator.train_model(dataset.train)

    # callback to overwrite parameters of the new model with the already trained model
    def copy_params(net):
        params1 = train1_output.trained_net.collect_params()
        params2 = net.collect_params()
        for p1, p2 in zip(params1.values(), params2.values()):
            p2.set_data(p1.data())

    estimator = SimpleFeedForwardEstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(
            epochs=5, num_batches_per_epoch=10, post_initialize_cb=copy_params
        ),
    )

    new_pred = estimator.train(dataset.train)

    ev = Evaluator(num_workers=0)
    agg_metrics1, _ = backtest_metrics(
        dataset.test, train1_output.predictor, evaluator=ev
    )
    agg_metrics2, _ = backtest_metrics(dataset.test, new_pred, evaluator=ev)

    df = pd.DataFrame([agg_metrics1, agg_metrics2], index=["model1", "model2"])
    print(df)
