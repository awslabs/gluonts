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
    ModelIterationAveraging,
)
from gluonts.mx.trainer.model_iteration_averaging import NTA
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator


def test_callbacklist_include():

    cb1 = TrainingHistory()
    cb2 = TerminateOnNaN()
    cb3 = TrainingHistory()

    list0 = CallbackList([cb2])
    list1 = CallbackList([cb1, cb2])
    list2 = CallbackList([cb3])

    list1.include(list2)
    list0.include(list2)

    assert len(list1.callbacks) == 2 and len(list1.callbacks) == 2
    assert len(list1.callbacks) == 2 and len(list1.callbacks) == 2


def test_callbacks():
    history = TrainingHistory()
    iter_avg = ModelIterationAveraging(avg_strategy=NTA())

    dataset = "m4_hourly"
    dataset = get_dataset(dataset)
    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq
    n_epochs = 4

    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        freq=freq,
        trainer=Trainer(epochs=n_epochs, callbacks=[history, iter_avg]),
    )

    predictor = estimator.train(dataset.train, num_workers=None)

    assert len(history.loss_history) == n_epochs

    ws = WarmStart(predictor=predictor)

    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        freq=freq,
        trainer=Trainer(epochs=n_epochs, callbacks=[history, iter_avg, ws]),
    )
    predictor = estimator.train(dataset.train, num_workers=None)

    assert len(history.loss_history) == n_epochs * 2
