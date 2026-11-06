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

import numpy as np
import pandas as pd
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import (
    InputDataset,
    LabelDataset,
    OffsetSplitter,
    TestData,
)

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


@pytest.mark.xfail(
    reason="This test is known to fail with numpy>=1.24, and a fix is pending"
)
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


def test_multivariate_grouper_with_test_data() -> None:
    df = pd.DataFrame(
        data={
            "target": list(range(100)) + list(range(100)),
            "item_id": ["var_1"] * 100 + ["var_2"] * 100,
        },
        index=list(pd.date_range(start="1970-01-01", periods=100, freq="1D"))
        * 2,
    )
    gluon_ds = PandasDataset.from_long_dataframe(
        dataframe=df,
        target="target",
        item_id="item_id",
        freq="1D",
    )

    splitter = OffsetSplitter(offset=70)
    train_ds, test_template = splitter.split(gluon_ds)

    windows = 29
    prediction_length = 1
    test_ds = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=1,
    )

    train_grouper = MultivariateGrouper(max_target_dim=2)
    test_grouper = MultivariateGrouper(
        num_test_dates=windows, max_target_dim=2
    )

    # 1. Training data grouping
    train_gr_data = train_grouper(train_ds)
    train_entries = list(train_gr_data)
    assert len(train_entries) == 1
    assert train_entries[0]["target"].shape == (2, 70)

    # 2. TestData grouping directly
    test_gr_data = test_grouper(test_ds)
    assert isinstance(test_gr_data, TestData)
    assert len(test_gr_data) == windows

    pairs = list(test_gr_data)
    assert len(pairs) == windows
    for i, (inp, lbl) in enumerate(pairs):
        assert inp["target"].shape == (2, 70 + i)
        assert lbl["target"].shape == (2, prediction_length)

    # 3. TestData.input grouping via property
    assert isinstance(test_gr_data.input, InputDataset)
    input_entries = list(test_gr_data.input)
    assert len(input_entries) == windows
    assert input_entries[0]["target"].shape == (2, 70)
    assert input_entries[-1]["target"].shape == (2, 70 + windows - 1)

    # 4. TestData.label grouping via property
    assert isinstance(test_gr_data.label, LabelDataset)
    label_entries = list(test_gr_data.label)
    assert len(label_entries) == windows
    assert label_entries[0]["target"].shape == (2, prediction_length)

    # 5. Passing InputDataset directly to test_grouper
    grouped_from_input = test_grouper(test_ds.input)
    assert isinstance(grouped_from_input, InputDataset)
    assert len(list(grouped_from_input)) == windows
    assert next(iter(grouped_from_input))["target"].shape == (2, 70)

    # 6. Passing LabelDataset directly to test_grouper
    grouped_from_label = test_grouper(test_ds.label)
    assert isinstance(grouped_from_label, LabelDataset)
    assert len(list(grouped_from_label)) == windows
    assert next(iter(grouped_from_label))["target"].shape == (
        2,
        prediction_length,
    )

    # 7. Passing series-major DataEntry list to test_grouper
    univariate_inputs = list(test_ds.input)
    grouped_inputs = test_grouper(univariate_inputs)
    assert len(grouped_inputs) == windows
    assert grouped_inputs[0]["target"].shape == (2, 70)
    assert grouped_inputs[-1]["target"].shape == (2, 70 + windows - 1)

    # 8. Passing series-major (input, label) pairs list to test_grouper
    univariate_pairs = list(test_ds)
    grouped_pairs = test_grouper(univariate_pairs)
    assert len(grouped_pairs) == windows
    assert grouped_pairs[0][0]["target"].shape == (2, 70)
    assert grouped_pairs[0][1]["target"].shape == (2, prediction_length)

    # 9. Validation error when num_test_dates mismatches TestData.windows
    mismatch_grouper = MultivariateGrouper(num_test_dates=5, max_target_dim=2)
    with pytest.raises(ValueError, match="num_test_dates"):
        mismatch_grouper(test_ds)
