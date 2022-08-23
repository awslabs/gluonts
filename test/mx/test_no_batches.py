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

from gluonts.exceptions import GluonTSDataError
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer


@pytest.mark.parametrize("dataset", [[]])
def test_deepar_no_batches(dataset):
    estimator = DeepAREstimator(
        prediction_length=10,
        freq="H",
        trainer=Trainer(epochs=1, num_batches_per_epoch=1),
    )

    with pytest.raises(GluonTSDataError):
        estimator.train(dataset)
