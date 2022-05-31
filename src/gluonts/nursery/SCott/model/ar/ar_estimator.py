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

from typing import List, Optional

import torch
import torch.nn as nn

from pts import Trainer
from pts.dataset import FieldName
from pts.model import PTSEstimator, PTSPredictor, copy_parameters
from pts.modules import DistributionOutput, StudentTOutput
from pts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
)
from pts.transform.sampler import CustomUniformSampler
from .ar_network import (
    ARTrainingNetwork,
    ARPredictionNetwork,
)


class AREstimator(PTSEstimator):
    def __init__(
        self,
        prediction_length: int,
        freq: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_parallel_samples: int = 100,
    ) -> None:
        """
        Defines an estimator.

        All parameters should be serializable.
        """
        super().__init__(trainer=trainer)
        self.num_parallel_samples = num_parallel_samples
        self.freq = freq

        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

    # here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    # For a more complex transformation example, see the `pts.model.deepar`
    # transformation that includes time features, age feature, observed values
    # indicator, etc.
    def create_transformation(self, is_full_batch=False) -> Transformation:
        return Chain(
            [
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    # train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    train_sampler=CustomUniformSampler(),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    is_full_batch=is_full_batch,
                    time_series_fields=[],  # [FieldName.FEAT_DYNAMIC_REAL]
                )
            ]
        )

    # defines the network, we get to see one batch to initialize it.
    # the network should return at least one tensor that is used as a loss to minimize in the training loop.
    # several tensors can be returned for instance for analysis, see DeepARTrainingNetwork for an example.
    def create_training_network(
        self, device: torch.device
    ) -> ARTrainingNetwork:
        return ARTrainingNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
        ).to(device)

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> PTSPredictor:
        prediction_network = ARPredictionNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
