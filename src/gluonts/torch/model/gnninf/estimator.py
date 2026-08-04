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

# If you use this code in your work please cite:
# Multivariate Time Series Forecasting with Latent Graph Inference
# (https://arxiv.org/abs/2203.03423)

import logging
import torch
from gluonts.dataset.common import Dataset
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.torch.model.predictor import PyTorchPredictor
from . import trans
from .module import GNNInfModule
from gluonts.model.estimator import Estimator

logger = logging.getLogger(__name__)


class GNNInfEstimator(Estimator):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        num_batches_per_epoch: int,
        epochs: int,
        lr: float,
        gnn_name: str,
        nf: int,
        gnn_layers: int,
        device: str = "cpu",
        lead_time: int = 0,
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.epochs = epochs
        self.lr = lr
        self.gnn_name = gnn_name  # gnn or bpgnn
        self.nf = nf
        self.gnn_layers = gnn_layers
        self.device = device

    def train_model(
        self, train_dataset: Dataset, window_loss: int = 10
    ) -> GNNInfModule:
        # Build dataloaders
        data_loader = trans.wrap_with_dataloader(
            train_dataset,
            self.batch_size,
            self.num_batches_per_epoch,
            self.prediction_length,
            self.context_length,
        )
        n_nodes = len(train_dataset.list_data[0]["target"])
        net = GNNInfModule(
            self.freq,
            self.prediction_length,
            self.context_length,
            n_nodes=n_nodes,
            gnn_name=self.gnn_name,
            nf=self.nf,
            gnn_layers=self.gnn_layers,
            device=self.device,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_arr = []
        n_samples = 0
        net.train()
        for epoch in range(self.epochs):
            for idx, batch in enumerate(data_loader):
                context = batch["past_target"].to(
                    self.device
                )  # shape (bs, context_length, num_nodes)
                target = batch["future_target"].to(
                    self.device
                )  # shape (bs, pred_length, num_nodes)
                optimizer.zero_grad()
                distr_args, loc, scale = net(context)
                distr = net.distr_output.distribution(distr_args, loc, scale)
                loss = -distr.log_prob(target.to(net.device))
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                loss_arr.append(loss.detach().item())
                n_samples += 1
                if idx % window_loss == 0:
                    avg_loss = sum(loss_arr[-window_loss:]) / len(
                        loss_arr[-window_loss:]
                    )
                    logger.info(
                        "Epoch %d \t Iteration %d/%d: \t "
                        "Avg Loss (10 last iterations): %.4f",
                        epoch,
                        idx,
                        data_loader.length,
                        avg_loss,
                    )

        return net

    def get_predictor(
        self, net: torch.nn.Module, input_transform, batch_size: int = 32
    ) -> PyTorchPredictor:
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq,
            input_names=["past_target"],
            prediction_net=net,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(net.distr_output),
            device=net.device,
        )

    def train(self, train_dataset: Dataset) -> PyTorchPredictor:
        pred_net = self.train_model(train_dataset=train_dataset)
        transformations = trans.get_pred_trans(
            self.context_length, self.prediction_length
        )
        return self.get_predictor(pred_net, transformations, self.batch_size)
