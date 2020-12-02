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

from itertools import chain

import numpy as np
import pytest

from gluonts.model.rotbaum import TreeEstimator


@pytest.fixture()
def hyperparameters(dsinfo):
    return dict(
        context_length=2,
        quantiles=[0.1, 0.5, 0.9],
        num_workers=0,
    )


@pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.5]])
def test_accuracy(accuracy_test, hyperparameters, quantiles):
    hyperparameters.update(quantiles=quantiles, max_workers=32)

    accuracy_test(TreeEstimator, hyperparameters, accuracy=0.20)


def test_repr(repr_test, hyperparameters):
    repr_test(TreeEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(TreeEstimator, hyperparameters)
