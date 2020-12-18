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

import pandas as pd
import pytest

from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.mx.distribution.inflated_beta import ZeroAndOneInflatedBetaOutput
from gluonts.mx.trainer import Trainer


@pytest.mark.parametrize("hybridize", [False, True])
def test_symbol_and_array(hybridize: bool):
    # Tests for cases like the one presented in issue 1211, in which the Inflated
    # Beta outputs used a method only available to arrays and not to symbols.
    # We simply go through a short training to ensure no exceptions are raised.
    data = [
        {
            "target": [0, 0.0460043, 0.263906, 0.4103112, 1],
            "start": pd.to_datetime("1999-01-04"),
        },
        {
            "target": [1, 0.65815564, 0.44982578, 0.58875054, 0],
            "start": pd.to_datetime("1999-01-04"),
        },
    ]
    dataset = common.ListDataset(data, freq="W-MON", one_dim_target=True)

    trainer = Trainer(epochs=1, num_batches_per_epoch=2, hybridize=hybridize)

    estimator = deepar.DeepAREstimator(
        freq="W",
        prediction_length=2,
        trainer=trainer,
        distr_output=ZeroAndOneInflatedBetaOutput(),
        context_length=2,
        batch_size=1,
        scaling=False,
    )

    estimator.train(dataset)
