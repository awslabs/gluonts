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

from typing import Callable, List

import pytorch_lightning as pl
import torch
from torch import nn

from gluonts.dataset.artificial import default_synthetic
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.batchify import batchify
from gluonts.torch.model.forecast_generator import (
    DistributionForecastGenerator,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.distribution_output import (
    NormalOutput,
    StudentTOutput,
)
from gluonts.torch.support.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    InstanceSplitter,
)


def mean_abs_scaling(seq, min_scale=1e-5):
    return seq.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int,
        hidden_dimensions: List[int],
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
        scaling: Callable = mean_abs_scaling,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling

        dimensions = [context_length] + hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            self.__make_lin(
                dimensions[-1], prediction_length * hidden_dimensions[-1]
            )
        )

        self.nn = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1])

    @staticmethod
    def __make_lin(dim_in, dim_out):
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def forward(self, context):
        scale = self.scaling(context)
        scaled_context = context / scale
        nn_out = self.nn(scaled_context)
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros_like(scale), scale

    def get_predictor(self, input_transform, batch_size=32, device=None):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(
                self.distr_output
            ),
            device=None,
        )


class LightningFeedForwardNetwork(FeedForwardNetwork, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        context = batch["past_target"]
        target = batch["future_target"]

        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length

        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)

        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def test_simple_model():
    dsinfo, training_data, test_data = default_synthetic()

    freq = dsinfo.metadata.freq
    prediction_length = dsinfo.prediction_length
    context_length = 2 * prediction_length
    hidden_dimensions = [10, 10]

    net = LightningFeedForwardNetwork(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        hidden_dimensions=hidden_dimensions,
        distr_output=NormalOutput(),
        batch_norm=True,
        scaling=mean_abs_scaling,
    )

    transformation = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )

    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    data_loader = TrainDataLoader(
        training_data,
        batch_size=8,
        stack_fn=batchify,
        transform=transformation + training_splitter,
        num_batches_per_epoch=5,
    )

    trainer = pl.Trainer(max_epochs=3, callbacks=[], weights_summary=None)
    trainer.fit(net, train_dataloader=data_loader)

    prediction_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    predictor = net.get_predictor(transformation + prediction_splitter)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data, predictor=predictor
    )

    evaluator = Evaluator(quantiles=[0.5, 0.9], num_workers=None)

    agg_metrics, _ = evaluator(ts_it, forecast_it)
