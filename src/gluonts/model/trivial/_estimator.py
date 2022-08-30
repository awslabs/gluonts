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

from gluonts.core import serde
from gluonts.model.estimator import DummyEstimator

from gluonts.model.trivial.constant import ConstantPredictor
from gluonts.model.trivial.identity import IdentityPredictor
from gluonts.model.trivial.mean import MeanPredictor, MovingAveragePredictor


@serde.dataclass
class ConstantEstimator(DummyEstimator):
    predictor_cls: type = ConstantPredictor


@serde.dataclass
class IdentityEstimator(DummyEstimator):
    predictor_cls: type = IdentityPredictor


@serde.dataclass
class MeanEstimator(DummyEstimator):
    predictor_cls: type = MeanPredictor


@serde.dataclass
class MovingAverageEstimator(DummyEstimator):
    predictor_cls: type = MovingAveragePredictor
