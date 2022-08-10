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
import pytest

# First-party imports
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.callback import (
    CallbackList,
    TerminateOnNaN,
    WarmStart,
    TrainingHistory,
)
from gluonts.mx.trainer.model_iteration_averaging import (
    NTA,
    ModelIterationAveraging,
)
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx import SimpleFeedForwardEstimator


def test_callbacklist():

    cb1 = TrainingHistory()
    cb2 = TerminateOnNaN()
    cb3 = TrainingHistory()

    list0 = CallbackList([cb2])
    list1 = CallbackList([cb1, cb2])
    list2 = CallbackList([cb3])

    list1 = CallbackList(list1.callbacks + list2.callbacks)
    list0 = CallbackList(list0.callbacks + list2.callbacks)

    assert len(list1.callbacks) == 3
    assert len(list0.callbacks) == 2


def test_callbacks():
    n_epochs = 4

    history = TrainingHistory()
    iter_avg = ModelIterationAveraging(avg_strategy=NTA(epochs=2 * n_epochs))

    dataset = "m4_hourly"
    dataset = get_dataset(dataset)
    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        trainer=Trainer(epochs=n_epochs, callbacks=[history, iter_avg]),
    )

    predictor = estimator.train(dataset.train, num_workers=None)

    assert len(history.loss_history) == n_epochs

    ws = WarmStart(predictor=predictor)

    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        trainer=Trainer(epochs=n_epochs, callbacks=[history, iter_avg, ws]),
    )
    predictor = estimator.train(dataset.train, num_workers=None)

    assert len(history.loss_history) == n_epochs * 2
