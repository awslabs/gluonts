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

from gluonts.model.seq2seq import (
    MQCNNEstimator,
    MQRNNEstimator,
    Seq2SeqEstimator,
)


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        context_length=dsinfo.prediction_length,
        num_batches_per_epoch=1,
        quantiles=[0.1, 0.5, 0.9],
        use_symbol_block_predictor=True,
    )


@pytest.fixture(
    params=[MQCNNEstimator, MQRNNEstimator], ids=["mqcnn", "mqrnn"]
)
def Estimator(request):
    return request.param


@pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.5]])
@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(
    Estimator, accuracy_test, hyperparameters, hybridize, quantiles
):
    hyperparameters.update(
        num_batches_per_epoch=100, hybridize=hybridize, quantiles=quantiles
    )

    accuracy_test(Estimator, hyperparameters, accuracy=0.25)


def test_repr(Estimator, repr_test, hyperparameters):
    repr_test(Estimator, hyperparameters)


def test_serialize(Estimator, serialize_test, hyperparameters):
    serialize_test(Estimator, hyperparameters)
