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
from gluonts.dataset.hierarchical import HierarchicalDataset
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.trainer import Trainer


NUM_BOTTOM_TS = 4
FREQ = "H"
PERIODS = 168 * 2
S = np.vstack(
    (
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        np.identity(4),
    )
)
PREDICTION_LENGTH = 24


@pytest.mark.parametrize(
    "use_feat_dynamic_real",
    [True, False],
)
def test_train_prediction(use_feat_dynamic_real: Optional[pd.DataFrame]):
    entry = {
        "start": pd.Period("22-03-2020"),
        "target": np.random.random(size=(NUM_BOTTOM_TS, PERIODS)),
    }
    if use_feat_dynamic_real:
        entry["feat_dynamic_real"] = np.random.random(size=(3, PERIODS))

    dataset = HierarchicalDataset([entry], S=S)
    estimator = DeepVARHierarchicalEstimator(
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1, hybridize=False),
        S=S,
        use_feat_dynamic_real=use_feat_dynamic_real,
    )
    predictor = estimator.train(dataset)

    if use_feat_dynamic_real:
        entry["feat_dynamic_real"] = np.random.random(
            size=(3, PERIODS + PREDICTION_LENGTH)
        )

    dataset = HierarchicalDataset([entry], S=S)
    forecasts = list(predictor.predict(dataset))

    assert len(forecasts) == len(dataset)
    assert all(
        forecast.samples.shape == (100, PREDICTION_LENGTH, len(S))
        for forecast in forecasts
    )
