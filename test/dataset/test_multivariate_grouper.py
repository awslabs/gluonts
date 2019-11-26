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
import pytest
import numpy as np

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

UNIVARIATE_TS = [
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-07", "target": [5, 6, 7, 8]},
    ],
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-08", "target": [5, 6, 7, 8]},
    ],
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-07", "target": [0]},
    ],
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-01", "target": [0]},
    ],
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-08", "target": [5, 6, 7, 8]},
    ],
]

MULTIVARIATE_TS = [
    [{"start": "2014-09-07", "target": [[1, 2, 3, 4], [5, 6, 7, 8]]}],
    [
        {
            "start": "2014-09-07",
            "target": [[1, 2, 3, 4, 2.5], [6.5, 5, 6, 7, 8]],
        }
    ],
    [{"start": "2014-09-07", "target": [[1, 2, 3, 4], [0, 0, 0, 0]]}],
    [
        {
            "start": "2014-09-01",
            "target": [
                [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1, 2, 3, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }
    ],
    [{"start": "2014-09-07", "target": [[1, 2, 3, 4, 0], [0, 5, 6, 7, 8]]}],
]

TRAIN_FILL_RULE = [np.mean, np.mean, np.mean, np.mean, lambda x: 0.0]


@pytest.mark.parametrize(
    "univariate_ts, multivariate_ts, train_fill_rule",
    zip(UNIVARIATE_TS, MULTIVARIATE_TS, TRAIN_FILL_RULE),
)
def test_multivariate_grouper_train(
    univariate_ts, multivariate_ts, train_fill_rule
) -> None:
    univariate_ds = ListDataset(univariate_ts, freq="1D")
    multivariate_ds = ListDataset(
        multivariate_ts, freq="1D", one_dim_target=False
    )

    grouper = MultivariateGrouper(train_fill_rule=train_fill_rule)
    assert (
        list(grouper(univariate_ds))[0]["target"]
        == list(multivariate_ds)[0]["target"]
    ).all()

    assert (
        list(grouper(univariate_ds))[0]["start"]
        == list(multivariate_ds)[0]["start"]
    )


UNIVARIATE_TS_TEST = [
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-07", "target": [5, 6, 7, 8]},
        {"start": "2014-09-08", "target": [0, 1, 2, 3]},
        {"start": "2014-09-08", "target": [4, 5, 6, 7]},
    ],
    [
        {"start": "2014-09-07", "target": [1, 2, 3, 4]},
        {"start": "2014-09-07", "target": [5, 6, 7, 8]},
        {"start": "2014-09-08", "target": [0, 1, 2, 3]},
        {"start": "2014-09-08", "target": [4, 5, 6, 7]},
    ],
]

MULTIVARIATE_TS_TEST = [
    [
        {"start": "2014-09-07", "target": [[1, 2, 3, 4], [5, 6, 7, 8]]},
        {"start": "2014-09-07", "target": [[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]},
    ],
    [
        {"start": "2014-09-07", "target": [[5, 6, 7, 8]]},
        {"start": "2014-09-07", "target": [[0, 4, 5, 6, 7]]},
    ],
]

TEST_FILL_RULE = [lambda x: 0.0, lambda x: 0.0]
MAX_TARGET_DIM = [2, 1]


@pytest.mark.parametrize(
    "univariate_ts, multivariate_ts, test_fill_rule, max_target_dim",
    zip(
        UNIVARIATE_TS_TEST,
        MULTIVARIATE_TS_TEST,
        TEST_FILL_RULE,
        MAX_TARGET_DIM,
    ),
)
def test_multivariate_grouper_test(
    univariate_ts, multivariate_ts, test_fill_rule, max_target_dim
) -> None:
    univariate_ds = ListDataset(univariate_ts, freq="1D")
    multivariate_ds = ListDataset(
        multivariate_ts, freq="1D", one_dim_target=False
    )

    grouper = MultivariateGrouper(
        test_fill_rule=test_fill_rule,
        num_test_dates=2,
        max_target_dim=max_target_dim,
    )

    for grouped_data, multivariate_data in zip(
        grouper(univariate_ds), multivariate_ds
    ):
        assert (grouped_data["target"] == multivariate_data["target"]).all()

        assert grouped_data["start"] == multivariate_data["start"]
