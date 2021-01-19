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
from gluonts.core.component import equals
from gluonts.model.deepar import DeepAREstimator


def test_nested_params():
    deepar = DeepAREstimator(prediction_length=7, freq="D")

    assert equals(deepar, serde.flat.decode(serde.flat.encode(deepar)))

    deepar2 = serde.flat.clone(deepar, {"trainer.epochs": 999})
    assert deepar2.trainer.epochs == 999
