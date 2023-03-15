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
import torch
import pytest

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import Evaluator


FREQ = "D"
PRED_LENGTH = 5
TRAINER_KARGS = {"max_epochs": 1, "accelerator": "gpu", "devices": 1}


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="only run the test on GPU"
)
@pytest.mark.parametrize(
    "model",
    [
        SimpleFeedForwardEstimator(
            prediction_length=PRED_LENGTH, trainer_kwargs=TRAINER_KARGS
        ),
        DeepAREstimator(
            freq=FREQ,
            prediction_length=PRED_LENGTH,
            trainer_kwargs=TRAINER_KARGS,
        ),
    ],
)
def test_torch_model_on_gpu(
    model: PyTorchLightningEstimator,
):
    n = 100
    date = "2015-04-07 00:00:00"

    df = pd.DataFrame(
        np.random.randn(n).astype(np.float32),
        index=pd.period_range(date, periods=n, freq=FREQ),
    )
    dataset = PandasDataset(df, target=0)

    predictor = model.train(dataset)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset, predictor=predictor, num_samples=100
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(
        ts_it,
        forecast_it,
    )
