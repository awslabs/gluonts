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
import numpy as np

from gluonts.ext.rotbaum import TreeEstimator, TreePredictor

from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features
from gluonts.dataset.common import ListDataset

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


def test_short_history_item_pred():
    prediction_length = 7
    freq = "D"

    dataset = ListDataset(
        data_iter=[
            {
                "start": "2017-10-11",
                "item_id": "item_1",
                "target": np.array(
                    [
                        1.0,
                        9.0,
                        2.0,
                        0.0,
                        0.0,
                        1.0,
                        5.0,
                        3.0,
                        4.0,
                        2.0,
                        0.0,
                        0.0,
                        1.0,
                        6.0,
                    ]
                ),
                "feat_static_cat": np.array([0.0, 0.0], dtype=float),
                "past_feat_dynamic_real": np.array(
                    [
                        [1.0222e06 for i in range(14)],
                        [750.0 for i in range(14)],
                    ]
                ),
            },
            {
                "start": "2017-10-11",
                "item_id": "item_2",
                "target": np.array([7.0, 0.0, 0.0, 23.0, 13.0]),
                "feat_static_cat": np.array([0.0, 1.0], dtype=float),
                "past_feat_dynamic_real": np.array(
                    [[0 for i in range(5)], [750.0 for i in range(5)]]
                ),
            },
        ],
        freq=freq,
    )

    predictor = TreePredictor(
        freq=freq,
        prediction_length=prediction_length,
        quantiles=[0.1, 0.5, 0.9],
        max_n_datapts=50000,
        method="QuantileRegression",
        use_past_feat_dynamic_real=True,
        use_feat_dynamic_real=False,
        use_feat_dynamic_cat=False,
        use_feat_static_real=False,
        cardinality="auto",
    )
    predictor = predictor.train(dataset)
    forecasts = list(predictor.predict(dataset))
    assert forecasts[1].quantile(0.5).shape[0] == prediction_length
