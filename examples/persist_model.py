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
This example shows how to serialize and deserialize a model
"""
import os
import pprint

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.support.util import get_download_path
from gluonts.mx.trainer import Trainer
from gluonts.model.predictor import Predictor

if __name__ == "__main__":

    dataset = get_dataset("exchange_rate")

    estimator = SimpleFeedForwardEstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10),
    )

    predictor = estimator.train(dataset.train)

    # save the trained model in a path ~/.mxnet/gluon-ts/feedforward/
    # or $MXNET_HOME/feedforward if MXNET_HOME is defined
    model_path = get_download_path() / "feedforward"
    os.makedirs(model_path, exist_ok=True)

    predictor.serialize(model_path)

    # loads it back and evaluate predictions accuracy with the deserialized model
    predictor_deserialized = Predictor.deserialize(model_path)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor_deserialized, num_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint.pprint(agg_metrics)
