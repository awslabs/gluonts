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
import pytest

# First-party imports
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features
from gluonts.trainer import Trainer


common_estimator_hps = dict(
    freq="D",
    prediction_length=3,
    trainer=Trainer(
        epochs=3, num_batches_per_epoch=2, batch_size=1, hybridize=True
    ),
    past_length=10,
    add_trend=True,
)


@pytest.mark.parametrize(
    "estimator, datasets",
    [
        # No features
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[1],
                use_feat_static_cat=False,
            ),
            make_dummy_datasets_with_features(),
        ),
        # Single static categorical feature
        (
            partial(
                DeepStateEstimator, **common_estimator_hps, cardinality=[5]
            ),
            make_dummy_datasets_with_features(cardinality=[5]),
        ),
        # Multiple static categorical features
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[3, 10, 42],
            ),
            make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        ),
        # Multiple static categorical features for one of which cardinality = 1
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[3, 1, 42],
            ),
            make_dummy_datasets_with_features(cardinality=[3, 1, 42]),
        ),
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[1],
                use_feat_static_cat=False,
                use_feat_dynamic_real=True,
            ),
            make_dummy_datasets_with_features(num_feat_dynamic_real=1),
        ),
        # Multiple dynamic real feature
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[1],
                use_feat_static_cat=False,
                use_feat_dynamic_real=True,
            ),
            make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        ),
        # Multiple dynamic real feature (ignored)
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[1],
                use_feat_static_cat=False,
            ),
            make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        ),
        # Both static categorical and dynamic real features
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                use_feat_dynamic_real=True,
                use_feat_static_cat=True,
                cardinality=[3, 10, 42],
            ),
            make_dummy_datasets_with_features(
                cardinality=[3, 10, 42], num_feat_dynamic_real=3
            ),
        ),
    ],
)
def test_deepstate_smoke(estimator, datasets):
    # TODO: pass `dtype` below once you add `dtype` support to `DeepStateEstimator`
    estimator = estimator()
    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)


# This test makes sure that `DeepStateEstimator` raises exception when it is misused.
#   * cardinality is not passed explicitly
#   * product of cardinalities is not greater than 1 even though `use_feat_static_cat == True`
@pytest.mark.parametrize(
    "estimator, datasets",
    [
        # Static categorical feature ignored
        (
            partial(DeepStateEstimator, **common_estimator_hps),
            make_dummy_datasets_with_features(cardinality=[3]),
        ),
        # Static categorical features are ignored by wrongly setting cardinality = [1, 1, 1]
        (
            partial(
                DeepStateEstimator,
                **common_estimator_hps,
                cardinality=[1, 1, 1],
            ),
            make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        ),
    ],
)
def test_deepstate_exceptions_with_feat_static_cat(estimator, datasets):
    with pytest.raises(Exception):
        estimator()
