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

from gluonts.model.npts import NPTSEstimator

hyperparameters = dict(
    kernel_type="uniform",
    use_default_features=True,
)


def test_accuracy(accuracy_test):
    accuracy_test(NPTSEstimator, hyperparameters, accuracy=0.0)


def test_repr(repr_test):
    repr_test(NPTSEstimator, hyperparameters)


def test_serialize(serialize_test):
    serialize_test(NPTSEstimator, hyperparameters)
