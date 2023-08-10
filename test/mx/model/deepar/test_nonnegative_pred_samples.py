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
import pytest

from gluonts.mx import DeepAREstimator
from gluonts.mx.distribution import StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features


@pytest.mark.parametrize("datasets", [make_dummy_datasets_with_features()])
@pytest.mark.parametrize("distr_output", [StudentTOutput()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("impute_missing_values", [False, True])
@pytest.mark.parametrize("symbol_block_predictor", [False, True])
def test_deepar_nonnegative_pred_samples(
    distr_output,
    datasets,
    dtype,
    impute_missing_values,
    symbol_block_predictor,
):
    estimator = DeepAREstimator(
        distr_output=distr_output,
        dtype=dtype,
        impute_missing_values=impute_missing_values,
        nonnegative_pred_samples=True,
        freq="D",
        prediction_length=3,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1),
    )

    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)

    if symbol_block_predictor:
        predictor = predictor.as_symbol_block_predictor(dataset=dataset_test)

    forecasts = list(predictor.predict(dataset_test))
    assert all([forecast.samples.dtype == dtype for forecast in forecasts])
    assert len(forecasts) == len(dataset_test)

    for forecast in forecasts:
        assert (forecast.samples >= 0).all()
