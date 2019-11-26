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

# Standard library imports
from functools import partial

# Third-party imports
import numpy as np
import pytest

# First-party imports
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer


common_estimator_hps = dict(
    freq="D",
    prediction_length=3,
    trainer=Trainer(epochs=3, num_batches_per_epoch=2, batch_size=4),
)


@pytest.mark.parametrize(
    "estimator, datasets",
    [
        # No features
        (
            partial(DeepAREstimator, **common_estimator_hps),
            make_dummy_datasets_with_features(),
        ),
        # Single static categorical feature
        (
            partial(
                DeepAREstimator,
                **common_estimator_hps,
                use_feat_static_cat=True,
                cardinality=[5],
            ),
            make_dummy_datasets_with_features(cardinality=[5]),
        ),
        # Multiple static categorical features
        (
            partial(
                DeepAREstimator,
                **common_estimator_hps,
                use_feat_static_cat=True,
                cardinality=[3, 10, 42],
            ),
            make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        ),
        # Multiple static categorical features (ignored)
        (
            partial(DeepAREstimator, **common_estimator_hps),
            make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        ),
        # Single dynamic real feature
        (
            partial(
                DeepAREstimator,
                **common_estimator_hps,
                use_feat_dynamic_real=True,
            ),
            make_dummy_datasets_with_features(num_feat_dynamic_real=1),
        ),
        # Multiple dynamic real feature
        (
            partial(
                DeepAREstimator,
                **common_estimator_hps,
                use_feat_dynamic_real=True,
            ),
            make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        ),
        # Multiple dynamic real feature (ignored)
        (
            partial(DeepAREstimator, **common_estimator_hps),
            make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        ),
        # Both static categorical and dynamic real features
        (
            partial(
                DeepAREstimator,
                **common_estimator_hps,
                use_feat_dynamic_real=True,
                use_feat_static_cat=True,
                cardinality=[3, 10, 42],
            ),
            make_dummy_datasets_with_features(
                cardinality=[3, 10, 42], num_feat_dynamic_real=3
            ),
        ),
        # Both static categorical and dynamic real features (ignored)
        (
            partial(DeepAREstimator, **common_estimator_hps),
            make_dummy_datasets_with_features(
                cardinality=[3, 10, 42], num_feat_dynamic_real=3
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_deepar_smoke(estimator, datasets, dtype):
    estimator = estimator(dtype=dtype)
    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    assert all([forecast.samples.dtype == dtype for forecast in forecasts])
    assert len(forecasts) == len(dataset_test)
