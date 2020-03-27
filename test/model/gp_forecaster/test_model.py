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

from gluonts.model.gp_forecaster import GaussianProcessEstimator


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-2,
        hybridize=True,
        cardinality=dsinfo.cardinality,
        num_batches_per_epoch=1,
        time_features=dsinfo.time_features,
        use_symbol_block_predictor=False,
        # FIXME: test_shell fails with use_symbol_block_predictor=True
        # FIXME and float_type = np.float64
    )


@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(num_batches_per_epoch=200, hybridize=hybridize)

    accuracy_test(GaussianProcessEstimator, hyperparameters, accuracy=0.2)


def test_repr(repr_test, hyperparameters):
    repr_test(GaussianProcessEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(GaussianProcessEstimator, hyperparameters)
