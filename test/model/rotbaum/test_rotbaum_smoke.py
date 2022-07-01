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

import pytest

from gluonts.model.rotbaum import TreeEstimator

from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features

# TODO: Add support for categorical and dynamic features.


@pytest.mark.parametrize(
    "datasets",
    [
        # No features
        make_dummy_datasets_with_features(),
        # Single static categorical feature
        make_dummy_datasets_with_features(cardinality=[5]),
        # Multiple static categorical features
        make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        # Multiple static categorical features
        make_dummy_datasets_with_features(cardinality=[3, 10, 42]),
        # Single dynamic real feature
        make_dummy_datasets_with_features(num_feat_dynamic_real=1),
        # Multiple dynamic real feature
        make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        # Multiple dynamic real feature
        make_dummy_datasets_with_features(num_feat_dynamic_real=3),
        # Both static categorical and dynamic real features
        make_dummy_datasets_with_features(
            cardinality=[3, 10, 42], num_feat_dynamic_real=3
        ),
        # Both static categorical and dynamic real features
        make_dummy_datasets_with_features(
            cardinality=[3, 10, 42], num_feat_dynamic_real=3
        ),
    ],
)
def test_rotbaum_smoke(datasets):
    dataset_train, dataset_test = datasets
    hps = {
        "freq": "D",
        "prediction_length": 3,
        "context_length": 3,
    }
    estimator = TreeEstimator.from_inputs(dataset_train, **hps)

    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    assert len(forecasts) == len(dataset_test)
