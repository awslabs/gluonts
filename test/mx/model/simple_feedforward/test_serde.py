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

import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from gluonts.model import Predictor
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer


def test_simplefeedforward_symbol_block_serde():
    with tempfile.TemporaryDirectory(
        prefix="gluonts-predictor-temp-"
    ) as temp_dir:
        dataset = [
            {
                "start": pd.Period("2022-01-01", freq="D"),
                "target": np.random.normal(size=(200)),
            }
        ]

        estimator = SimpleFeedForwardEstimator(
            prediction_length=10,
            trainer=Trainer(
                epochs=2,
                num_batches_per_epoch=5,
            ),
        )

        predictor = estimator.train(dataset)
        predictor = predictor.as_symbol_block_predictor(dataset=dataset)

        model_path = Path(temp_dir) / "model"
        model_path.mkdir()
        predictor.serialize(model_path)

        new_symbol_predictor = Predictor.deserialize(model_path)

        assert len(list(new_symbol_predictor.predict(dataset))) == 1
