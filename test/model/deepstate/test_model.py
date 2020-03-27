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

from gluonts.model.deepstate import DeepStateEstimator


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=False,
        num_cells=2,
        num_layers=1,
        context_length=2,
        past_length=dsinfo.prediction_length,
        cardinality=[1],
        use_feat_static_cat=False,
        num_batches_per_epoch=1,
        use_symbol_block_predictor=False,
    )


def test_accuracy(accuracy_test, hyperparameters):
    hyperparameters.update(num_batches_per_epoch=100)

    accuracy_test(DeepStateEstimator, hyperparameters, accuracy=0.5)


def test_repr(repr_test, hyperparameters):
    repr_test(DeepStateEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(DeepStateEstimator, hyperparameters)
