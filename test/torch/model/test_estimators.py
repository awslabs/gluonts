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
from functools import partial
from itertools import islice
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from gluonts.dataset.repository import get_dataset
from gluonts.model.predictor import Predictor
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.deep_npts import (
    DeepNPTSEstimator,
    DeepNPTSNetworkDiscrete,
    DeepNPTSNetworkSmooth,
)
from gluonts.torch.model.forecast import DistributionForecast
from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.d_linear import DLinearEstimator
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.model.lag_tst import LagTSTEstimator
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.torch.modules.loss import NegativeLogLikelihood, QuantileLoss
from gluonts.torch.distributions import ImplicitQuantileNetworkOutput


@pytest.mark.parametrize(
    "estimator_constructor",
    [
        lambda dataset: DeepAREstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
            loss=NegativeLogLikelihood(beta=0.1),
            scaling=False,
        ),
        lambda dataset: DeepAREstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            context_length=1,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
            scaling=False,
        ),
        lambda dataset: MQF2MultiHorizonEstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda dataset: SimpleFeedForwardEstimator(
            prediction_length=dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda dataset: DLinearEstimator(
            prediction_length=dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda dataset: LagTSTEstimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda dataset: TemporalFusionTransformerEstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda dataset: DeepNPTSEstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            context_length=2 * dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            epochs=2,
        ),
        lambda dataset: PatchTSTEstimator(
            prediction_length=dataset.metadata.prediction_length,
            patch_len=16,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
    ],
)
@pytest.mark.parametrize("use_validation_data", [False, True])
def test_estimator_constant_dataset(
    estimator_constructor, use_validation_data: bool
):
    constant = get_dataset("constant")

    estimator = estimator_constructor(constant)

    if use_validation_data:
        predictor = estimator.train(
            training_data=constant.train,
            validation_data=constant.train,
        )
    else:
        predictor = estimator.train(
            training_data=constant.train,
        )

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    forecasts = predictor_copy.predict(constant.test)

    for f in islice(forecasts, 5):
        if isinstance(f, DistributionForecast):
            f = f.to_sample_forecast()
        f.mean


@pytest.mark.parametrize(
    "estimator_constructor",
    [
        lambda freq, prediction_length: DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            num_feat_dynamic_real=3,
            num_feat_static_real=1,
            num_feat_static_cat=2,
            cardinality=[2, 2],
            trainer_kwargs=dict(max_epochs=2),
            loss=NegativeLogLikelihood(beta=0.1),
        ),
        lambda freq, prediction_length: DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            num_feat_dynamic_real=3,
            num_feat_static_real=1,
            num_feat_static_cat=2,
            cardinality=[2, 2],
            trainer_kwargs=dict(max_epochs=2),
            loss=QuantileLoss(),
            distr_output=ImplicitQuantileNetworkOutput(),
        ),
        lambda freq, prediction_length: MQF2MultiHorizonEstimator(
            freq=freq,
            prediction_length=prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            num_feat_dynamic_real=3,
            num_feat_static_real=1,
            num_feat_static_cat=2,
            cardinality=[2, 2],
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda freq, prediction_length: TemporalFusionTransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            dynamic_dims=[3],
            static_dims=[1],
            static_cardinalities=[2, 2],
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda freq, prediction_length: DeepNPTSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=2 * prediction_length,
            batch_norm=True,
            network_type=partial(DeepNPTSNetworkDiscrete, use_softmax=True),
            use_feat_static_cat=True,
            cardinality=[2, 2],
            num_feat_static_real=1,
            num_feat_dynamic_real=0,
            input_scaling=None,
            dropout_rate=0.0,
            batch_size=4,
            num_batches_per_epoch=3,
            epochs=2,
        ),
        lambda freq, prediction_length: DeepNPTSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=2 * prediction_length,
            batch_norm=False,
            network_type=partial(DeepNPTSNetworkDiscrete, use_softmax=False),
            use_feat_static_cat=True,
            cardinality=[2, 2],
            num_feat_static_real=1,
            num_feat_dynamic_real=0,
            input_scaling="min_max_scaling",
            dropout_rate=0.0,
            batch_size=4,
            num_batches_per_epoch=3,
            epochs=2,
        ),
        lambda freq, prediction_length: DeepNPTSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=2 * prediction_length,
            batch_norm=True,
            network_type=DeepNPTSNetworkSmooth,
            use_feat_static_cat=True,
            cardinality=[2, 2],
            num_feat_static_real=1,
            num_feat_dynamic_real=0,
            input_scaling="standard_normal_scaling",
            dropout_rate=0.1,
            batch_size=4,
            num_batches_per_epoch=3,
            epochs=2,
        ),
    ],
)
def test_estimator_with_features(estimator_constructor):
    freq = "1h"
    prediction_length = 12

    training_dataset = [
        {
            "start": pd.Period("2021-01-01 00:00:00", freq=freq),
            "target": np.ones(200, dtype=np.float32),
            "feat_static_cat": np.array([0, 1], dtype=np.float32),
            "feat_static_real": np.array([42.0], dtype=np.float32),
            "feat_dynamic_real": np.ones((3, 200), dtype=np.float32),
            "__unused__": np.ones(3, dtype=np.float32),
        },
        {
            "start": pd.Period("2021-02-01 00:00:00", freq=freq),
            "target": np.ones(100, dtype=np.float32),
            "feat_static_cat": np.array([1, 0], dtype=np.float32),
            "feat_static_real": np.array([1.0], dtype=np.float32),
            "feat_dynamic_real": np.ones((3, 100), dtype=np.float32),
            "__unused__": np.ones(5, dtype=np.float32),
        },
    ]

    prediction_dataset = [
        {
            "start": pd.Period("2021-01-01 00:00:00", freq=freq),
            "target": np.ones(200, dtype=np.float32),
            "feat_static_cat": np.array([0, 1], dtype=np.float32),
            "feat_static_real": np.array([42.0], dtype=np.float32),
            "feat_dynamic_real": np.ones(
                (3, 200 + prediction_length), dtype=np.float32
            ),
            "__unused__": np.ones(3, dtype=np.float32),
        },
        {
            "start": pd.Period("2021-02-01 00:00:00", freq=freq),
            "target": np.ones(100, dtype=np.float32),
            "feat_static_cat": np.array([1, 0], dtype=np.float32),
            "feat_static_real": np.array([1.0], dtype=np.float32),
            "feat_dynamic_real": np.ones(
                (3, 100 + prediction_length), dtype=np.float32
            ),
            "__unused__": np.ones(5, dtype=np.float32),
        },
    ]

    estimator = estimator_constructor(freq, prediction_length)

    predictor = estimator.train(
        training_data=training_dataset,
        validation_data=training_dataset,
    )

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    forecasts = predictor_copy.predict(prediction_dataset)

    for f in islice(forecasts, 5):
        f.mean
