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

from functools import partial

import numpy as np
import pytest

from gluonts.mx import DeepAREstimator
from gluonts.mx.distribution import StudentTOutput
from gluonts.mx.trainer import Trainer

from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features

common_estimator_hps = dict(
    freq="D",
    prediction_length=3,
    trainer=Trainer(epochs=1, num_batches_per_epoch=1),
)


@pytest.mark.parametrize(
    "estimator, datasets",
    [
        (
            partial(DeepAREstimator, **common_estimator_hps),
            make_dummy_datasets_with_features(),
        ),
    ],
)
@pytest.mark.parametrize("distr_output", [StudentTOutput()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("impute_missing_values", [False, True])
@pytest.mark.parametrize("nonnegative_pred_samples", [True])
def test_deepar_nonnegative_pred_samples(
    distr_output,
    estimator,
    datasets,
    dtype,
    impute_missing_values,
    nonnegative_pred_samples,
):
    estimator = estimator(
        distr_output=distr_output,
        dtype=dtype,
        impute_missing_values=impute_missing_values,
        nonnegative_pred_samples=nonnegative_pred_samples,
    )
    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))
    assert all([forecast.samples.dtype == dtype for forecast in forecasts])
    assert len(forecasts) == len(dataset_test)

    if nonnegative_pred_samples:
        assert all(
            [(forecast.samples < 0).sum() == 0 for forecast in forecasts]
        )
