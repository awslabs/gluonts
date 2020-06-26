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
import mxnet as mx
import mxnet.gluon.nn as nn
import numpy as np
import pytest
import pandas as pd
import math

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.model_iteration_averaging import (
    IterationAveragingStrategy,
    NTA_V1,
    NTA_V2,
    Alpha_Suffix,
)

def initialize_model() -> nn.HybridBlock:
    # dummy training data
    N = 10  # number of time series
    T = 100  # number of timesteps
    prediction_length = 24
    freq = "1H"
    custom_dataset = np.random.normal(size=(N, T))
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series
    train_ds = ListDataset([{'target': x, 'start': start}
                            for x in custom_dataset[:, :-prediction_length]],
                        freq=freq)
    # create a simple model
    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=prediction_length,
        context_length=T,
        freq=freq,
        trainer=Trainer(ctx="cpu",
                        epochs=1,
                        learning_rate=1e-3,
                        num_batches_per_epoch=1
                    )
    )
    # train model
    predictor = estimator.train(train_ds)

    return predictor.prediction_net

@pytest.mark.parametrize("n", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20])
def test_NTA_V1(n: int):
    model = initialize_model()
    params = model.collect_params()
    avg_strategy = NTA_V1(n=n)
    loss_list = [5, 4, 3, 2, 3, 3, 3, 3, 3, 3, 3]
    for i, loss in enumerate(loss_list):
        for k,v in params.items():
            for arr in v.list_data():
                arr[:] = i
        avg_strategy.update_average_trigger(loss)
        avg_strategy.apply(model)
    # nothing is cached yet, thus load_cached_model won't change anything
    # test cached model
    avg_strategy.load_cached_model(model)
    for k,v in params.items():
        for arr in v.list_data():
            # the last model should have 10 in all coordinates
            assert mx.nd.norm(arr - 10).asscalar() < 1e-30
    # test averaged model
    avg_strategy.load_averaged_model(model)
    if n <= 0 or n > 6:
        # average never happends, model is not changed
        for k,v in params.items():
            for arr in v.list_data():
                # the last model should have 10 in all coordinates
                assert mx.nd.norm(arr - 10).asscalar() < 1e-30
    else:
        for k,v in params.items():
            for arr in v.list_data():
                # NTA_V1 takes the average on the last 7-n iterations
                assert mx.nd.norm(arr - (4+n+10)/2.).asscalar() < 1e-30
    # test cached model
    avg_strategy.load_cached_model(model)
    for k,v in params.items():
        for arr in v.list_data():
            # the last model should have 10 in all coordinates
            assert mx.nd.norm(arr - 10).asscalar() < 1e-30

@pytest.mark.parametrize("n", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20])
def test_NTA_V2(n: int):
    model = initialize_model()
    params = model.collect_params()
    avg_strategy = NTA_V2(n=n)
    loss_list = [5, 4, 3, 2, 3, 3, 3, 3, 3, 3, 3]
    for i, loss in enumerate(loss_list):
        for k,v in params.items():
            for arr in v.list_data():
                arr[:] = i
        avg_strategy.update_average_trigger(loss)
        avg_strategy.apply(model)
    # nothing is cached yet, thus load_cached_model won't change anything
    # test cached model
    avg_strategy.load_cached_model(model)
    for k,v in params.items():
        for arr in v.list_data():
            # the last model should have 10 in all coordinates
            assert mx.nd.norm(arr - 10).asscalar() < 1e-30
    # test averaged model
    avg_strategy.load_averaged_model(model)
    if n <= 0 or n >= len(loss_list):
        # average never happends, model is not changed
        for k,v in params.items():
            for arr in v.list_data():
                # the last model should have 10 in all coordinates
                assert mx.nd.norm(arr - 10).asscalar() < 1e-30
    else:
        for k,v in params.items():
            for arr in v.list_data():
                # NTA_V2 takes the average once the loss increases, no matter what n is taken 
                # (the first n iterations are ignored)
                if n <= 4:
                    val = 7
                else:
                    val = (n+10) / 2.
                assert mx.nd.norm(arr - val).asscalar() < 1e-30
    # test cached model
    avg_strategy.load_cached_model(model)
    for k,v in params.items():
        for arr in v.list_data():
            # the last model should have 10 in all coordinates
            assert mx.nd.norm(arr - 10).asscalar() < 1e-30

@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_Alpha_Suffix(alpha: float):
    model = initialize_model()
    params = model.collect_params()
    loss_list = [5, 4, 3, 2, 3, 3, 3, 3, 3, 3, 3]
    avg_strategy = Alpha_Suffix(epochs=len(loss_list), alpha=alpha)
    for i, loss in enumerate(loss_list):
        for k,v in params.items():
            for arr in v.list_data():
                arr[:] = i
        avg_strategy.update_average_trigger(i+1)
        avg_strategy.apply(model)
    # nothing is cached yet, thus load_cached_model won't change anything
    # test cached model
    avg_strategy.load_cached_model(model)
    for k,v in params.items():
        for arr in v.list_data():
            # the last model should have 10 in all coordinates
            assert mx.nd.norm(arr - 10).asscalar() < 1e-30
    # test averaged model
    avg_strategy.load_averaged_model(model)
    n = max(int(math.ceil(len(loss_list)*(1-alpha))), 1)
    if n > len(loss_list):
        # average never happends, model is not changed
        for k,v in params.items():
            for arr in v.list_data():
                # the last model should have 10 in all coordinates
                assert mx.nd.norm(arr - 10).asscalar() < 1e-30
    else:
        for k,v in params.items():
            for arr in v.list_data():
                val = (n+9) / 2.
                assert mx.nd.norm(arr - val).asscalar() < 1e-30
    # test cached model
    avg_strategy.load_cached_model(model)
    for k,v in params.items():
        for arr in v.list_data():
            # the last model should have 10 in all coordinates
            assert mx.nd.norm(arr - 10).asscalar() < 1e-30