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
import sys

import pytest

from gluonts.mx.model.renewal import DeepRenewalProcessEstimator


@pytest.fixture()
def hyperparameters():
    return dict(
        num_cells=10,
        num_layers=3,
        context_length=10,
        epochs=1,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="don't run on windows")
@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy_smoke_test(accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(num_batches_per_epoch=80, hybridize=hybridize)
    accuracy_test(DeepRenewalProcessEstimator, hyperparameters, accuracy=10)


@pytest.mark.parametrize("hybridize", [True, False])
def test_serialize(serialize_test, hyperparameters, hybridize):
    hyperparameters.update(hybridize=hybridize)
    serialize_test(DeepRenewalProcessEstimator, hyperparameters)
