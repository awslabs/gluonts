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

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.model.deepstate import DeepStateEstimator
from gluonts.mx.trainer import Trainer


def test_deepstate_on_tsf_dataset():
    dataset = get_dataset("covid_deaths")

    estimator = DeepStateEstimator(
        cardinality=[len(dataset)],
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        trainer=Trainer(
            ctx="cpu",
            epochs=1,
            learning_rate=1e-3,
            num_batches_per_epoch=1,
            hybridize=False,
        ),
    )
    estimator.train(dataset.train)
