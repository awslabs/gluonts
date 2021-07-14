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

from typing import Tuple

import mxnet as mx
import numpy as np
import pandas as pd
import pytest
from mxnet import nd

from gluonts.model.tpp import (
    PointProcessGluonPredictor,
    PointProcessSampleForecast,
)

from gluonts.mx import Tensor
from gluonts.transform import (
    ContinuousTimeInstanceSplitter,
    ContinuousTimePredictionSampler,
)

from .common import point_process_dataset, point_process_dataset_2


class MockTPPPredictionNet(mx.gluon.HybridBlock):
    def __init__(
        self,
        num_parallel_samples: int = 100,
        prediction_interval_length: float = 5.0,
        context_interval_length: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_parallel_samples = num_parallel_samples
        self.prediction_interval_length = prediction_interval_length
        self.context_interval_length = context_interval_length

    def hybridize(self, active=True, **kwargs):
        if active:
            raise NotImplementedError()

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self, F, past_target: Tensor, past_valid_length: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Return two tensors, of shape
        (batch_size, num_samples, max_prediction_length, target_dim)
        and (batch_size, num_samples) respectively.
        """
        batch_size = past_target.shape[0]
        assert past_valid_length.shape[0] == batch_size

        target_shape = (self.num_parallel_samples, batch_size, 25)
        pred_target = nd.stack(
            nd.random.uniform(shape=target_shape),
            nd.random.randint(0, 10, shape=target_shape).astype(np.float32),
            axis=-1,
        )
        pred_valid_length = nd.random.randint(
            15, 25 + 1, shape=target_shape[:2]
        )

        return pred_target, pred_valid_length


@pytest.fixture
def predictor_factory():
    def get_predictor(**kwargs) -> PointProcessGluonPredictor:
        default_kwargs = dict(
            input_names=["past_target", "past_valid_length"],
            prediction_net=MockTPPPredictionNet(
                prediction_interval_length=5.0
            ),
            batch_size=128,
            prediction_interval_length=5.0,
            freq="H",
            ctx=mx.cpu(),
            input_transform=ContinuousTimeInstanceSplitter(
                1,
                5,
                ContinuousTimePredictionSampler(
                    allow_empty_interval=False, min_past=1
                ),
            ),
        )

        default_kwargs.update(**kwargs)

        return PointProcessGluonPredictor(**default_kwargs)

    return get_predictor


@pytest.mark.parametrize(
    "dataset_tuple", [(point_process_dataset, 1), (point_process_dataset_2, 3)]
)
def test_tpp_pred_dataset_2_shapes_ok(dataset_tuple, predictor_factory):
    dataset, ds_length = dataset_tuple

    predictor = predictor_factory()
    forecasts = [fc for fc in predictor.predict(dataset(), 50)]

    assert len(forecasts) == ds_length

    for forecast in forecasts:
        # each forecast should have 3dim samples, 1d valid lengths
        assert isinstance(forecast, PointProcessSampleForecast)

        assert forecast.samples.shape == (50, 25, 2)
        assert forecast.valid_length.shape == (50,)

        assert (
            forecast.prediction_interval_length
            == predictor.prediction_interval_length
        )

        assert forecast.start_date == pd.Timestamp("2011-01-01 03:00:00")
        assert forecast.end_date == pd.Timestamp("2011-01-01 08:00:00")
