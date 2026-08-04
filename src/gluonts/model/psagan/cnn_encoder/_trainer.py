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

import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.model.psagan.cnn_encoder._loss import SimilarityScore
from gluonts.model.psagan.cnn_encoder.plot import PlotterSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class Trainer:
    @validated()
    def __init__(
        self,
        num_epochs: int = 1000,
        lr: float = 0.001,
        save_display_frq: int = 100,
        save_dir: pathlib.PosixPath = pathlib.Path.cwd(),
        save_model_dir: pathlib.PosixPath = pathlib.Path.cwd(),
        device: str = "cpu",
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_display_frq = save_display_frq
        self.save_dir = save_dir
        self.save_model_dir = save_model_dir
        assert device == "cpu" or device == "gpu"
        self.device = torch.device("cuda" if device == "gpu" else "cpu")

    def __call__(self, data_loader: DataLoader, network: nn.Module):
        optimizer = Adam(network.parameters(), lr=self.lr)
        # loss_func = TripletLoss()
        similarity_score = SimilarityScore().to(self.device)
        network = network.float()
        plotter = PlotterSaver(save_dir=self.save_dir)
        logger.info(pathlib.Path.cwd())

        with open(self.save_dir / "configuration.txt", "w") as f:
            f.write("Model = " + repr(network) + "\n")
            f.write("optimizer = " + repr(optimizer) + "\n")
            # f.write("Loss function = " + repr(loss_func) + "\n")
            f.close()

        loss_recorder_pos = np.empty(self.num_epochs)
        loss_recorder_neg = np.empty(self.num_epochs)
        loss_recorder_neg_pos = np.empty(self.num_epochs)

        for epoch in range(self.num_epochs):
            for idx, (
                context_sequence,
                sequence,
                negative_samples,
            ) in enumerate(data_loader):

                nb_sample = context_sequence.size(1)
                nb_negative_samples = negative_samples.size(1)
                batch_size = context_sequence.size(0)

                context_sequence = context_sequence.view(
                    batch_size * nb_sample, 1, -1
                )
                sequence = sequence.view(batch_size * nb_sample, 1, -1)
                negative_samples = negative_samples.view(
                    batch_size * nb_negative_samples, 1, -1
                )

                optimizer.zero_grad()

                context_sequence_emb = network(context_sequence.float()).view(
                    batch_size, nb_sample, -1
                )
                sequence_emb = network(sequence.float()).view(
                    batch_size, nb_sample, -1
                )
                negative_emb = network(negative_samples.float()).view(
                    batch_size, nb_negative_samples, -1
                )

                loss_pos = -similarity_score(
                    sequence_emb, context_sequence_emb
                ).mean()
                loss_neg = -similarity_score(
                    -context_sequence_emb.expand_as(negative_emb), negative_emb
                ).mean()
                loss_neg_pos = -similarity_score(
                    -sequence_emb.expand_as(negative_emb), negative_emb
                ).mean()
                loss = loss_pos + loss_neg + loss_neg_pos
                # loss = loss_func(
                #     context_sequence_emb, sequence_emb, negative_emb
                # ).mean()
                loss.backward()
                optimizer.step()
                loss_recorder_pos[epoch] += loss_pos.detach().item()
                loss_recorder_neg[epoch] += loss_neg.detach().item()
                loss_recorder_neg_pos[epoch] += loss_neg_pos.detach().item()

            loss_recorder_pos[epoch] /= len(data_loader)
            loss_recorder_neg[epoch] /= len(data_loader)
            loss_recorder_neg_pos[epoch] /= len(data_loader)

            if epoch % self.save_display_frq == 0:
                logger.info(
                    f"Epoch : {epoch}, Training Loss : {loss_recorder_pos[epoch] + loss_recorder_neg[epoch]}, \
                        Training Loss pos : {loss_recorder_pos[epoch]}, Training Loss neg : {loss_recorder_neg[epoch]}, \
                        Training Loss pos_neg : {loss_recorder_neg_pos[epoch]}   "
                )
                torch.save(
                    network.state_dict(),
                    self.save_model_dir / f"model_{epoch}.pth",
                )

            plotter.log_loss(loss_recorder_pos[epoch], prefix="Positive")
            plotter.log_loss(loss_recorder_neg[epoch], prefix="negative")
            plotter.log_loss(loss_recorder_neg_pos[epoch], prefix="pos_neg")
        plotter.save_plot(loss=loss_recorder_pos, prefix="Positive")
        plotter.save_plot(loss=loss_recorder_neg, prefix="Negative")
        plotter.save_plot(loss=loss_recorder_neg_pos, prefix="pos_neg")

        return network
