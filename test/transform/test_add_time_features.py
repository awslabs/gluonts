from typing import Dict, List

import pandas as pd
import numpy as np
import pytest

from gluonts.dataset.common import Dataset, ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.transform.feature import AddTimeFeatures
from gluonts.time_feature import TimeFeature, WeekOfYear, MonthOfYear


def simple_add_time_features(
    entry: Dict,
    time_features: List[TimeFeature],
    pred_length: int = 0,
    dtype=np.float32,
):
    assert pred_length >= 0
    index = to_pandas(entry, freq=entry["start"].freq).index
    if pred_length > 0:
        index = index.union(
            pd.date_range(
                index[-1] + index.freq,
                index[-1] + pred_length * index.freq,
                freq=index.freq,
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
                    {"start": "2021-01-01 00:00:00", "target": [1.0] * 20},
                    {"start": "2021-02-15 00:45:00", "target": [1.0] * 20},
                    {"start": "2021-05-06 12:00:00", "target": [1.0] * 20},
                ],
                freq="3M",
            ),
            [MonthOfYear()],
        ),
        (
            ListDataset(
                data_iter=[
                    {"start": "2021-01-01 00:00:00", "target": [1.0] * 20},
                    {"start": "2021-02-15 00:45:00", "target": [1.0] * 20},
                    {"start": "2021-05-06 12:00:00", "target": [1.0] * 20},
                ],
                freq="3W",
            ),
            [WeekOfYear(), MonthOfYear()],
        ),
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
        expected_features = simple_add_time_features(
            entry, time_features, pred_length
        )
        assert np.allclose(expected_features, transformed_entry["features"])
