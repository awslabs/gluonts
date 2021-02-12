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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.mx.model.forecast import DistributionForecast
from gluonts.mx.trainer import Trainer
from gluonts.nursery.streaming_models.predictor import (
    StreamPredictor,
    StateAwarePredictorWrapper,
)
from gluonts.nursery.streaming_models.rnn import StreamingRnnEstimator


@pytest.mark.parametrize(
    "estimator, training_data",
    [
        (
            StreamingRnnEstimator(
                freq="1min",
                lead_time=5 * 60,
                train_window_length=100,
                trainer=Trainer(epochs=2, num_batches_per_epoch=2),
            ),
            ListDataset(
                [
                    {
                        "start": "2020-01-01 00:00:00",
                        "target": np.random.normal(
                            loc=[100] * 2000, scale=10, size=(2000,)
                        ).tolist(),
                    }
                    for _ in range(3)
                ],
                freq="1min",
            ),
        )
    ],
)
def test_StateAwarePredictorWrapper(estimator, training_data):
    predictor = estimator.train(training_data)

    stream_predictor_orig = StateAwarePredictorWrapper(
        predictor=predictor,
        state_initializer=estimator.get_state_initializer(),
    )

    with tempfile.TemporaryDirectory() as directory:
        stream_predictor_orig.serialize(Path(directory))
        stream_predictor = StreamPredictor.deserialize(Path(directory))

    test_idx = pd.date_range(
        start="2020-06-01 00:00:00", freq="1min", periods=500
    )
    test_target = np.random.normal(
        loc=[100] * len(test_idx),
        scale=10,
    ).tolist()

    state = stream_predictor.initial_state()

    forecast_all, _ = stream_predictor.step(
        data={"start": test_idx[0], "target": test_target},
        state=state,
    )

    state = stream_predictor.initial_state()

    forecast_mean = []
    forecast_p90 = []

    for ts, v in zip(test_idx, test_target):
        forecast, state = stream_predictor.step(
            data={"start": ts, "target": [v]},
            state=state,
        )

        exp_forecast_start = ts + (stream_predictor.lead_time + 1) * ts.freq

        assert (
            isinstance(forecast, DistributionForecast)
            and len(forecast.index) == 1
            and forecast.start_date == exp_forecast_start
        )

        forecast_mean.append(forecast.mean[0])

    assert np.allclose(forecast_all.mean, forecast_mean)
