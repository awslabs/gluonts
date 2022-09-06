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

"""
This example shows how to fit a model and evaluate its predictions.
"""

# third party imports
import pandas as pd
import pytest

# first party imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx import DeepAREstimator
from gluonts.mx.trainer import Trainer


def create_dynamic_dataset(
    start: str, length: int, num_dynamic: int
) -> ListDataset:
    """Create a ListDataset with dynamic values equal to the target."""
    return ListDataset(
        [
            {
                FieldName.TARGET: list(range(length)),
                FieldName.START: pd.Timestamp(start),
                FieldName.FEAT_DYNAMIC_REAL: [list(range(length))]
                * num_dynamic,
                FieldName.FEAT_DYNAMIC_CAT: [list(range(length))]
                * num_dynamic,
            }
        ],
        freq="D",
    )


@pytest.mark.parametrize(
    "train_length, test_length, prediction_length, target_start, rolling_start, num_dynamic_feat",
    [
        (10, 15, 2, "01-01-2019", "01-13-2019", 1),
        (10, 15, 2, "01-01-2019", "01-11-2019", 2),
    ],
)
def test_dynamic_integration(
    train_length: int,
    test_length: int,
    prediction_length: int,
    target_start: str,
    rolling_start: str,
    num_dynamic_feat: int,
):
    """
    Trains an estimator on a rolled dataset with dynamic features.
    Tests https://github.com/awslabs/gluonts/issues/1390
    """
    train_ds = create_dynamic_dataset(
        target_start, train_length, num_dynamic_feat
    )
    rolled_ds = generate_rolling_dataset(
        dataset=create_dynamic_dataset(
            target_start, test_length, num_dynamic_feat
        ),
        strategy=StepStrategy(prediction_length=prediction_length),
        start_time=pd.Timestamp(rolling_start),
    )
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=2 * prediction_length,
        use_feat_dynamic_real=True,
        trainer=Trainer(epochs=1),
    )
    predictor = estimator.train(training_data=train_ds)
    forecast_it, ts_it = make_evaluation_predictions(
        rolled_ds, predictor=predictor, num_samples=100
    )
    training_agg_metrics, _ = Evaluator(num_workers=0)(ts_it, forecast_it)
    # it should have failed by this point if the dynamic features were wrong
    assert training_agg_metrics
