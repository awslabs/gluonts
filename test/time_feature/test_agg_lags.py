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

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.transform import AddAggregateLags


expected_lags = {
    "prediction_length_2": {
        "train": np.array(
            [
                [0, 0, 0, 0, 1, 1, 2, 2, 3],
                [0, 0, 0, 0, 0, 0, 1, 1, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        "test": np.array(
            [
                [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
                [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    },
    "prediction_length_1": {
        "train": np.array(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 4],
                [0, 0, 0, 0, 1, 1, 2, 2, 3],
                [0, 0, 0, 0, 0, 0, 1, 1, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        "test": np.array(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                [0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
                [0, 0, 0, 0, 0, 0, 1, 1, 2, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    },
}

valid_lags = {
    "prediction_length_2": [2, 3, 4, 6],
    "prediction_length_1": [1, 2, 3, 4, 6],
}


@pytest.mark.parametrize("pred_length", [2, 1])
def test_agg_lags(pred_length):
    # create dummy dataset
    target = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5])
    start = pd.Timestamp("01-01-2019", freq="1H")
    freq = "1H"
    ds = ListDataset(
        [{FieldName.TARGET: target, FieldName.START: start}], freq=freq
    )

    # 2H aggregate lags
    lags_2H = [1, 2, 3, 4, 6]

    add_agg_lags = AddAggregateLags(
        target_field=FieldName.TARGET,
        output_field="lags_2H",
        pred_length=pred_length,
        base_freq=freq,
        agg_freq="2H",
        agg_lags=lags_2H,
    )

    # transformation check
    assert add_agg_lags.ratio == 2
    assert (
        add_agg_lags.valid_lags
        == valid_lags[f"prediction_length_{pred_length}"]
    )

    train_entry = next(add_agg_lags(iter(ds), is_train=True))
    assert np.array_equal(
        train_entry["lags_2H"],
        expected_lags[f"prediction_length_{pred_length}"]["train"],
    )

    test_entry = next(add_agg_lags(iter(ds), is_train=False))
    assert np.array_equal(
        test_entry["lags_2H"],
        expected_lags[f"prediction_length_{pred_length}"]["test"],
    )
