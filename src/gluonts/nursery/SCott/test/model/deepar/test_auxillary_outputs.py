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

from itertools import islice

import torch

from pts import Trainer
from pts.dataset import constant_dataset, TrainDataLoader
from pts.model import get_module_forward_input_names
from pts.model.deepar import DeepAREstimator
from pts.modules import StudentTOutput

ds_info, train_ds, test_ds = constant_dataset()
freq = ds_info.metadata.freq
prediction_length = ds_info.prediction_length


def test_distribution():
    """
    Makes sure additional tensors can be accessed and have expected shapes
    """
    prediction_length = ds_info.prediction_length
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        input_size=48,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1),
        distr_output=StudentTOutput(),
    )

    train_output = estimator.train_model(train_ds)

    # todo adapt loader to anomaly detection use-case
    batch_size = 2
    num_samples = 3

    training_data_loader = TrainDataLoader(
        dataset=train_ds,
        transform=train_output.transformation,
        batch_size=batch_size,
        num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
        device=torch.device("cpu"),
    )

    seq_len = 2 * ds_info.prediction_length

    for data_entry in islice(training_data_loader, 1):
        input_names = get_module_forward_input_names(train_output.trained_net)

        distr = train_output.trained_net.distribution(
            *[data_entry[k] for k in input_names]
        )

        assert distr.sample((num_samples,)).shape == (
            num_samples,
            batch_size,
            seq_len,
        )
