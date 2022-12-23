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
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.dataset.hierarchical import HierarchicalTimeSeries
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.trainer import Trainer


NUM_BOTTOM_TS = 4
FREQ = "H"
PERIODS = 168 * 2
S = np.vstack(([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], np.eye(4)))
PREDICTION_LENGTH = 24


def random_ts(num_ts: int, periods: int, freq: str):
    index = pd.period_range(start="22-03-2020", periods=periods, freq=freq)

    return pd.concat(
        [
            pd.Series(data=np.random.random(size=len(index)), index=index)
            for _ in range(num_ts)
        ],
        axis=1,
    )


@pytest.mark.parametrize(
    "features_df",
    [
        None,
        random_ts(
            num_ts=S.shape[0], periods=PERIODS + PREDICTION_LENGTH, freq=FREQ
        ),
    ],
)
def test_train_prediction(features_df: Optional[pd.DataFrame]):
    if features_df is not None:
        use_feat_dynamic_real = True
        features_df_train = features_df.iloc[:-PREDICTION_LENGTH, :]
    else:
        use_feat_dynamic_real = False
        features_df_train = None

    # HTS
    ts_at_bottom_level = random_ts(
        num_ts=NUM_BOTTOM_TS,
        periods=PERIODS,
        freq="H",
    )
    hts = HierarchicalTimeSeries(
        ts_at_bottom_level=ts_at_bottom_level,
        S=S,
    )

    dataset = hts.to_dataset(feat_dynamic_real=features_df_train)

    estimator = DeepVARHierarchicalEstimator(
        freq=hts.freq,
        prediction_length=PREDICTION_LENGTH,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1, hybridize=False),
        target_dim=hts.num_ts,
        S=hts.S,
        use_feat_dynamic_real=use_feat_dynamic_real,
    )

    predictor = estimator.train(dataset)

    predictor_input = hts.to_dataset(feat_dynamic_real=features_df)
    forecasts = list(predictor.predict(predictor_input))

    assert len(forecasts) == len(dataset)
    assert all(
        [
            forecast.samples.shape == (100, PREDICTION_LENGTH, hts.num_ts)
            for forecast in forecasts
        ]
    )
