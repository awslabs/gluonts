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

from gluonts.model.canonical._estimator import (
    CanonicalRNNEstimator,
    MLPForecasterEstimator,
)


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        num_cells=2,
        num_layers=1,
        context_length=2,
        use_symbol_block_predictor=False,
    )


@pytest.fixture(
    params=[MLPForecasterEstimator, CanonicalRNNEstimator], ids=["mlp", "rnn"]
)
def Estimator(request):
    return request.param


@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(Estimator, accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(hybridize=hybridize)

    accuracy_test(Estimator, hyperparameters, accuracy=10.0)


def test_repr(Estimator, repr_test, hyperparameters):
    repr_test(Estimator, hyperparameters)


@pytest.mark.xfail
def test_serialize(Estimator, serialize_test, hyperparameters):
    serialize_test(Estimator, hyperparameters)
