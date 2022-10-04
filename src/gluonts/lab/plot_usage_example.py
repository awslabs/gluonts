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

from matplotlib import pyplot as plt
from gluonts.model.forecast import Quantile
from gluonts.mx import SimpleFeedForwardEstimator, Trainer
from gluonts.model.npts import NPTSPredictor
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.lab.plot import plot
from gluonts.evaluation import make_evaluation_predictions

dataset = get_dataset("m4_hourly")

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    trainer=Trainer(
        ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100
    ),
)

predictor = estimator.train(dataset.train)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)

ts_entry = tss[0]
forecast_entry = forecasts[0]

plot(
    forecast=forecast_entry,
    timeseries=ts_entry[-100 : -dataset.metadata.prediction_length + 1],
    date_format="%d-%m-%Y %H:%M",
    train_test_seperator=forecast_entry.index[0].to_timestamp()
)
