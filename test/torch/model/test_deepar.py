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

import tempfile
from itertools import islice
from pathlib import Path

import pytorch_lightning as pl

from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.predictor import Predictor
from gluonts.torch.model.deepar import DeepAREstimator


def test_torch_deepar():
    constant = get_dataset("constant")

    estimator = DeepAREstimator(
        freq=constant.metadata.freq,
        prediction_length=constant.metadata.prediction_length,
        batch_size=4,
        num_batches_per_epoch=3,
        trainer=pl.Trainer(max_epochs=2),
    )

    predictor = estimator.train(
        training_data=constant.train,
        validation_data=constant.train,
        shuffle_buffer_length=5,
    )

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    forecasts = predictor_copy.predict(constant.test)

    for f in islice(forecasts, 5):
        f.mean
