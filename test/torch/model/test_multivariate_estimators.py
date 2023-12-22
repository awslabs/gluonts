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
from itertools import islice
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.model.predictor import Predictor
from gluonts.torch.model.forecast import DistributionForecast
from gluonts.torch.model.i_transformer import ITransformerEstimator


@pytest.mark.parametrize(
    "estimator_constructor",
    [
        lambda dataset: ITransformerEstimator(
            prediction_length=dataset.metadata.prediction_length,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
    ],
)
@pytest.mark.parametrize("use_validation_data", [False, True])
def test_multivariate_estimator_constant_dataset(
    estimator_constructor, use_validation_data: bool
):
    constant = get_dataset("constant")
    estimator = estimator_constructor(constant)

    train_grouper = MultivariateGrouper(
        max_target_dim=int(constant.metadata.feat_static_cat[0].cardinality)
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(constant.test) / len(constant.train)),
        max_target_dim=int(constant.metadata.feat_static_cat[0].cardinality),
    )
    dataset_train = train_grouper(constant.train)
    dataset_test = test_grouper(constant.test)

    if use_validation_data:
        predictor = estimator.train(
            training_data=dataset_train, validation_data=dataset_test
        )
    else:
        predictor = estimator.train(training_data=dataset_train)

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    assert predictor == predictor_copy

    forecasts = predictor_copy.predict(dataset_test)

    for f in islice(forecasts, 5):
        if isinstance(f, DistributionForecast):
            f = f.to_sample_forecast()
        f.mean


@pytest.mark.parametrize(
    "estimator_constructor",
    [
        lambda freq, prediction_length: ITransformerEstimator(
            prediction_length=prediction_length,
            batch_size=4,
            trainer_kwargs=dict(max_epochs=2),
        ),
    ],
)
def test_multivariate_estimator_with_features(estimator_constructor):
    freq = "1h"
    prediction_length = 12
    estimator = estimator_constructor(freq, prediction_length)

    training_dataset = [
        {
            "start": pd.Period("2021-01-01 00:00:00", freq=freq),
            "target": np.ones((3, 200), dtype=np.float32),
            "feat_static_cat": np.array([0, 1], dtype=np.float32),
            "feat_static_real": np.array([42.0], dtype=np.float32),
            "feat_dynamic_real": np.ones((3, 200), dtype=np.float32),
            "__unused__": np.ones(3, dtype=np.float32),
        },
        {
            "start": pd.Period("2021-02-01 00:00:00", freq=freq),
            "target": np.ones((3, 100), dtype=np.float32),
            "feat_static_cat": np.array([1, 0], dtype=np.float32),
            "feat_static_real": np.array([1.0], dtype=np.float32),
            "feat_dynamic_real": np.ones((3, 100), dtype=np.float32),
            "__unused__": np.ones(5, dtype=np.float32),
        },
    ]

    prediction_dataset = [
        {
            "start": pd.Period("2021-01-01 00:00:00", freq=freq),
            "target": np.ones((3, 200), dtype=np.float32),
            "feat_static_cat": np.array([0, 1], dtype=np.float32),
            "feat_static_real": np.array([42.0], dtype=np.float32),
            "feat_dynamic_real": np.ones(
                (3, 200 + prediction_length), dtype=np.float32
            ),
            "__unused__": np.ones(3, dtype=np.float32),
        },
        {
            "start": pd.Period("2021-02-01 00:00:00", freq=freq),
            "target": np.ones((3, 100), dtype=np.float32),
            "feat_static_cat": np.array([1, 0], dtype=np.float32),
            "feat_static_real": np.array([1.0], dtype=np.float32),
            "feat_dynamic_real": np.ones(
                (3, 100 + prediction_length), dtype=np.float32
            ),
            "__unused__": np.ones(5, dtype=np.float32),
        },
    ]

    predictor = estimator.train(
        training_data=training_dataset,
        validation_data=training_dataset,
    )

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    assert predictor == predictor_copy

    forecasts = predictor_copy.predict(prediction_dataset)

    for f in islice(forecasts, 5):
        f.mean
