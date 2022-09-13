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

import pandas as pd
import numpy as np

from gluonts.mx import SimpleFeedForwardEstimator, Trainer


def test_incremental_training_smoke_mx():
    estimator = SimpleFeedForwardEstimator(
        prediction_length=6,
        trainer=Trainer(epochs=2),
    )

    dataset = [
        {"start": pd.Period("2022-03-04 00", "1H"), "target": np.ones((100,))},
        {"start": pd.Period("2022-04-05 00", "1H"), "target": np.ones((100,))},
    ]

    predictor = estimator.train(dataset)
    _ = estimator.train_from(predictor, dataset)
