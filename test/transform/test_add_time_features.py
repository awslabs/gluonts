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

from typing import Dict, List

import pandas as pd
import numpy as np
import pytest

from gluonts.dataset.common import Dataset, ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.transform.feature import AddTimeFeatures
from gluonts.time_feature import (
    TimeFeature,
    month_of_year,
    time_features_from_frequency_str,
)


def compute_time_features(
    entry: Dict,
    time_features: List[TimeFeature],
    pred_length: int = 0,
    dtype=np.float32,
):
    assert pred_length >= 0
    index = to_pandas(entry, freq=entry["start"].freq).index

    if pred_length > 0:
        index = index.union(
            pd.period_range(
                index[-1] + 1, index[-1] + pred_length, freq=index.freq
            )
        )

    feature_arrays = [feat(index) for feat in time_features]
    return np.vstack(feature_arrays).astype(dtype)


@pytest.mark.parametrize(
    "dataset, time_features",
    [
        (
            ListDataset(
                data_iter=[
                    {"start": "2021-01-01 00:00:06", "target": [1.0] * 50},
                    {"start": "2021-01-12 00:45:17", "target": [1.0] * 50},
                    {"start": "2021-02-18 12:00:28", "target": [1.0] * 50},
                    {"start": "2021-05-27 07:10:39", "target": [1.0] * 50},
                ],
                freq=freq_str,
            ),
            time_features_from_frequency_str(freq_str) + [month_of_year],
        )
        for freq_str in [
            "2M",
            "3W",
            "3W-MON",
            "3W-TUE",
            "3W-WED",
            "3W-THU",
            "3W-FRI",
            "3W-SAT",
            "2D",
            "5H",
        ]
    ],
)
@pytest.mark.parametrize("pred_length", [0, 1, 2, 10])
def test_AddTimeFeatures_correctness(
    dataset: Dataset, time_features: List[TimeFeature], pred_length: int
):
    transform = AddTimeFeatures(
        start_field="start",
        target_field="target",
        output_field="features",
        time_features=time_features,
        pred_length=pred_length,
    )
    for entry, transformed_entry in zip(
        dataset, transform(dataset, is_train=(pred_length == 0))
    ):
        expected_features = compute_time_features(
            entry, time_features, pred_length
        )
        assert np.allclose(expected_features, transformed_entry["features"])
