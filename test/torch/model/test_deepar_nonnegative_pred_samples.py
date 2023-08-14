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

import pytest

from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput, NormalOutput
from gluonts.testutil.dummy_datasets import make_dummy_datasets_with_features


@pytest.mark.parametrize("datasets", [make_dummy_datasets_with_features()])
@pytest.mark.parametrize("distr_output", [StudentTOutput(), NormalOutput()])
def test_deepar_nonnegative_pred_samples(
    distr_output,
    datasets,
):
    estimator = DeepAREstimator(
        distr_output=distr_output,
        nonnegative_pred_samples=True,
        freq="D",
        prediction_length=3,
        trainer_kwargs={"max_epochs": 1},
    )

    dataset_train, dataset_test = datasets
    predictor = estimator.train(dataset_train)
    forecasts = list(predictor.predict(dataset_test))

    assert len(forecasts) == len(dataset_test)

    for forecast in forecasts:
        assert (forecast.samples >= 0).all()
