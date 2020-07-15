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

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.distribution import GaussianOutput


@pytest.fixture()
def hyperparameters():
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        num_hidden_dimensions=[3],
        num_batches_per_epoch=1,
        use_symbol_block_predictor=True,
        distr_output=GaussianOutput(),
    )


@pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize("sampling", [True, False])
def test_accuracy(accuracy_test, hyperparameters, hybridize, sampling):
    hyperparameters.update(
        num_batches_per_epoch=200, hybridize=hybridize, sampling=sampling
    )

    accuracy_test(SimpleFeedForwardEstimator, hyperparameters, accuracy=0.3)


def test_repr(repr_test, hyperparameters):
    repr_test(SimpleFeedForwardEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(SimpleFeedForwardEstimator, hyperparameters)
