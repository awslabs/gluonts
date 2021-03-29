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
from pprint import pprint

import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)

if __name__ == "__main__":
    dataset = get_dataset("constant", regenerate=False)

    estimator = SimpleFeedForwardEstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10),
    )

    predictor = estimator.train(dataset.train)

    # create the rolled dataset to use for forecasting and evaluation
    dataset_rolled = generate_rolling_dataset(
        dataset=dataset.test,
        start_time=pd.Timestamp("2000-01-01-15", freq="1H"),
        end_time=pd.Timestamp("2000-01-02-04", freq="1H"),
        strategy=StepStrategy(
            prediction_length=dataset.metadata.prediction_length,
        ),
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset_rolled, predictor=predictor, num_samples=len(dataset_rolled)
    )

    agg_metrics, _ = Evaluator()(ts_it, forecast_it)

    pprint(agg_metrics)
