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
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from gluonts.core.component import validated
from gluonts.dataset.loader import TrainDataLoader
from gluonts.model.psagan._callbacks import (
    CudaCallback,
    GANLossCheck,
    JointTrainingPytorch,
    MinMaxScalingPytorch,
    PlotSamplesandTSNEPytorch,
    SaveModel,
    TimeCheck,
    TrainEvalCallback,
)
from gluonts.model.psagan._data import DataBunch
from gluonts.model.psagan._utils import Learner, Optimizers, Runner
from gluonts.model.psagan.cnn_encoder import CausalCNNEncoder
from gluonts.model.psagan.lars_optim import LARSWrapper
from gluonts.nursery.ContextFIDExperiment.res2tex import toLatex

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
        num_epochs: int,
        lr_generator: float,
        lr_discriminator: float,
        schedule: List[int],
        nb_step_discrim: int = 1,
        nb_epoch_fade_in_new_layer: int = 10,
        save_plot_dir: pathlib.PosixPath = pathlib.Path.cwd(),
        save_model_dir: pathlib.PosixPath = pathlib.Path.cwd(),
        EMA_value: float = 0.2,
        device: str = "cpu",
        use_loss: str = "wgan",
        momment_loss: float = 0.0,
        scaling_penalty: float = 0.0,
        betas_generator: Tuple[float, float] = (0.9, 0.999),
        betas_discriminator: Tuple[float, float] = (0.9, 0.999),
        scaling: str = "local",
        encoder_network_path: str = None,
        encoder_network_factor: float = None,
        LARS: bool = False,
    ):
        self.num_epochs = num_epochs
        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.schedule = schedule
        self.nb_step_discrim = nb_step_discrim
        self.nb_epoch_fade_in_new_layer = nb_epoch_fade_in_new_layer
        self.save_plot_dir = save_plot_dir
        self.save_model_dir = save_model_dir
        self.EMA_value = EMA_value
        assert device == "cpu" or device == "gpu", "Device must be cpu or gpu"
        self.device = device
        self.use_loss = use_loss
        self.momment_loss = momment_loss
        self.betas_generator = betas_generator
        self.betas_discriminator = betas_discriminator
        assert (
            scaling == "local" or scaling == "global" or scaling == "NoScale"
        ), "scaling has to be \
            local or global. If it is local, then the whole time series will be\
            mix-max scaled. If it is local, then each subseries of length \
            target_len will be min-max scaled independenty. If is is NoScale\
            then no scaling is applied to the dataset."
        self.scaling = scaling
        self.scaling_penalty = scaling_penalty
        self.encoder_network_factor = encoder_network_factor
        self.encoder_network_path = (
            Path(encoder_network_path)
            if encoder_network_path is not None
            else None
        )
        self.LARS = LARS

    def _create_encoder_network(self):
        logger.info(
            f"encoder network factor {self.encoder_network_factor}, encoder network path : {self.encoder_network_path}"
        )
        if (
            self.encoder_network_factor is not None
            and self.encoder_network_path is not None
        ):
            logger.info("Hey Loading the embedder")
            embed_model_path = list(
                list(self.encoder_network_path.glob("model_*"))[0].glob(
                    "model_*"
                )
            )[0]
            embed_config_path = list(
                list(self.encoder_network_path.glob("out_*"))[0].glob(
                    "configuration.txt"
                )
            )[0]
            latex_obj = toLatex(path_to_config_file_CNN=embed_config_path)

            embedder = CausalCNNEncoder(
                in_channels=latex_obj._get_in_channels_CNN(),
                channels=latex_obj._get_channels_CNN(),
                depth=latex_obj._get_depth_CNN(),
                reduced_size=latex_obj._get_reduced_size_CNN(),
                out_channels=latex_obj._get_out_channels_CNN(),
                kernel_size=latex_obj._get_kernel_size_CNN(),
            )
            device = torch.device("cuda" if self.device == "gpu" else "cpu")
            embedder.load_state_dict(
                torch.load(embed_model_path, map_location=device)
            )
            embedder.eval()
            embedder.to(device)
            location = next(embedder.parameters()).device
            logger.info(
                f"Embedder network has been loaded and is located on {location}"
            )

            return embedder
        else:
            logger.info("Hey Not loading the embedder")
            logger.info("No embedder network has been loaded")
            return None

    def __call__(self, data_loader: TrainDataLoader, network: nn.Module):

        data = DataBunch(data_loader)

        # opt_pre = Adam(network.generator.parameters(), lr = self.lr_generator)

        # learner = Learner(network.generator, opt_pre, data, SupervisedLoss())

        # run_pre = Runner(
        #     [
        #         TrainEvalCallback(),
        #         TimeCheck(),
        #         MinMaxScaling(),
        #         InputforInterpolation(0.5),
        #         SupervisedPreTraining(),
        #         LossCheck(save_plot_dir=self.save_plot_dir,),
        #         SaveModelSingle()

        #     ]
        # )

        opt_gen = Adam(
            network.generator.parameters(),
            lr=self.lr_generator,
            betas=self.betas_generator,
        )
        opt_disc = Adam(
            network.discriminator.parameters(),
            lr=self.lr_discriminator,
            betas=self.betas_discriminator,
        )

        opt = Optimizers(
            opt_generator=LARSWrapper(opt_gen) if self.LARS else opt_gen,
            opt_discriminator=LARSWrapper(opt_disc) if self.LARS else opt_disc,
        )
        learner = Learner(network, opt, data)

        # run = Runner(
        #     [
        #         TrainEvalCallback(),
        #         TimeCheck(),
        #     ]
        #     + ([MinMaxScalingPytorch()] if self.scaling == "local" else [])
        #     + [
        #         CudaCallback(device=self.device),
        #         InputforInterpolation(0.3),
        #         SupervisedPreTraining(),
        #         LossCheck(save_plot_dir=self.save_plot_dir),
        #         SaveModelSingle(save_model_dir=self.save_model_dir),
        #         PlotInterpolatedSamples(0.3, save_plot_dir=self.save_plot_dir),
        #     ]
        # )

        run = Runner(
            [
                TrainEvalCallback(),
                TimeCheck(),
            ]
            + ([MinMaxScalingPytorch()] if self.scaling == "local" else [])
            + [
                CudaCallback(device=self.device),
                JointTrainingPytorch(
                    nb_step_discrim=self.nb_step_discrim,
                    schedule=self.schedule,
                    nb_epoch_fade_in_new_layer=self.nb_epoch_fade_in_new_layer,
                    device=self.device,
                    use_loss=self.use_loss,
                    momment_loss=self.momment_loss,
                    scaling_penalty=self.scaling_penalty,
                    encoder_network=self._create_encoder_network(),
                    encoder_network_factor=self.encoder_network_factor,
                ),
                GANLossCheck(
                    save_plot_dir=self.save_plot_dir,
                    EMA_value=self.EMA_value,
                    nb_step_discrim=self.nb_step_discrim,
                ),
                SaveModel(save_model_dir=self.save_model_dir),
                PlotSamplesandTSNEPytorch(
                    freq=1000, freq_tsne=1000, save_plot_dir=self.save_plot_dir
                ),
            ]
        )

        with open(self.save_plot_dir / "configuration.txt", "w") as f:
            f.write(f"{repr(learner)}")
            f.write(f"{repr(run)}")
            f.close()

        run.fit(self.num_epochs, learner)
        return run.model
