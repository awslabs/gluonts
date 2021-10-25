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

import datetime
import logging
import pathlib
from math import log2
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class Callback:
    """
    Mother class of all the callbacks. It serves three purposes:
    - __call__ execute the function
    - set_runner associate the runner to the callback so it can access the runners' attributes
    - __getattr__ fetch the attribute in the runner class if it is not found in the callback
    Sets also by default the _order attribute to 0. The callbacks are executed by the runner
    in a sorted manner with respect to the value of the _order attribute.
    """

    _order = 0

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)


class TrainEvalCallback(Callback):
    """
    Mandatory callback to include. Specify which epoch we are currently at in a float format
    and set the model in train or eval mode.
    """

    _order = 0

    def __repr__(self):
        return "TrainEvalCallback \n"

    def begin_fit(self):
        self.run.n_epochs = 0.0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.n_iters

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class CudaCallback(Callback):
    """Train on GPU"""

    _order = 0

    def __init__(self, device: str):
        assert device == "cpu" or device == "gpu"
        self.device = torch.device("cuda" if device == "gpu" else "cpu")

    def begin_fit(self):
        self.run.model.to(self.device)
        self.run.device = self.device

    # def begin_batch(self):
    # logger.info(f"past_target on {self.run.item["past_target"].device}")
    # self.run.item["noise_with_time_feat"] = self.run.item["noise_with_time_feat"].cuda()
    # self.run.item["past_time_feat"] = self.run.item["past_time_feat"].cuda()
    # self.run.item["past_target"] = self.run.item["past_target"].cuda()


# class CudaCallbackPytorch(Callback):
#     """Train on GPU"""

#     _order = 4
#     def __init__(self, device : str):
#         assert device == "cpu" or device =="gpu"
#         self.device = torch.device("cuda" if device=="gpu" else "cpu")
#     def begin_fit(self):
#         self.run.model.to(self.device)
#         self.run.data.train_dl.to(self.device)

#     # def begin_batch(self):
#         # logger.info(f"past_target on {self.run.item["past_target"].device}")
#         # self.run.item["noise_with_time_feat"] = self.run.item["noise_with_time_feat"].cuda()
#         # self.run.item["past_time_feat"] = self.run.item["past_time_feat"].cuda()
#         # self.run.item["past_target"] = self.run.item["past_target"].cuda()


class TimeCheck(Callback):
    """
    Print the time at which the function is called.
    """

    def __repr__(self):
        return "TimeCheck Callback \n"

    def begin_fit(self):
        logger.info(
            "Training started at {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )


class MinMaxScalingPytorch(Callback):
    """
    Callback to minmax scale the target field
    """

    _order = 5

    def __repr__(self):
        return "MinMaxScaling Callback \n"

    def _min_max_scaling(self, target):
        _min, _ = torch.min(target, dim=1)
        _max, _ = torch.max(target, dim=1)

        _min = _min.unsqueeze(dim=1)

        _max = _max.unsqueeze(dim=1)
        scaling_factor = _max - _min
        scaling_factor = torch.where(
            scaling_factor != 0,
            scaling_factor,
            torch.ones_like(scaling_factor),
        )
        scaled_target = (target - _min) / scaling_factor
        return scaled_target

    def begin_batch(self):
        self.run.item[0] = self._min_max_scaling(self.run.item[0])


class InputforInterpolation(Callback):
    """Callback that creates the input used for interpolation"""

    _order = 4

    def __repr__(self):
        return f" Supervised Pre Training Callback \n \
            \t replace factor of {self.replace_factor}"

    def __init__(self, replace_factor: float):
        super(InputforInterpolation, self).__init__()
        assert (
            0 <= replace_factor <= 1
        ), " Replace factor must be a float between 0 and 1"
        self.replace_factor = replace_factor

    def _min_max_scaling(self, target):
        _min, _ = torch.min(target, dim=1)
        _max, _ = torch.max(target, dim=1)

        _min = _min.unsqueeze(dim=1)

        _max = _max.unsqueeze(dim=1)
        scaling_factor = _max - _min
        scaling_factor = torch.where(
            scaling_factor != 0,
            scaling_factor,
            torch.ones_like(scaling_factor),
        )
        scaled_target = (target - _min) / scaling_factor
        return scaled_target

    def forward_pass(self):
        shape = self.run.item[0].unsqueeze(1).shape
        modified_input = self.run.item[0].view(-1).clone().detach()
        nb_elmnt_change = int(shape[0] * shape[2] * self.replace_factor)
        idx_to_change = torch.randperm(len(modified_input))[:nb_elmnt_change]
        noise = torch.randn(nb_elmnt_change, device=self.run.device)
        modified_input[idx_to_change] = noise
        modified_input = self._min_max_scaling(modified_input.view(shape))
        self.run.interpolate_target = torch.cat(
            (self.run.item[1], modified_input),
            dim=1,
        )


class SupervisedPreTraining(Callback):
    """Callback that implements a pre-training of the generator using a supervised loss

    This callback trains the generator to perform a task of interpolation.
    It maps a sparse time series to its full version.
    """

    _order = 10

    def __repr__(self):
        return "Supervised Pre Training Callback"

    def __init__(self):
        self.L1Loss = nn.L1Loss()

    def forward_pass(self):
        self.reconstruction = self.run.model(
            x=self.run.interpolate_target, net="generator"
        )

    def compute_loss(self):
        self.run.loss_value = self.L1Loss(
            self.reconstruction.squeeze(1), self.run.item[0]
        )

    def backward(self):
        self.run.loss_value.backward()

    def step(self):
        self.run.opt.step(net="generator")

    def zero_grad(self):
        self.run.opt.zero_grad(net="generator")

    def begin_validate(self):
        return True


class LossCheck(Callback):
    """
    Callback to print the loss of the generator.
    Appends also the results to list accessible from the runner class.
    """

    _order = 15

    def __repr__(self):
        return f"LossCheck Callback: \n \
            \t save_plot_dir : {self.save_plot_dir}"

    def __init__(
        self,
        save_plot_dir: pathlib.PosixPath = pathlib.Path.cwd(),
    ):
        super(LossCheck, self).__init__()
        self.save_plot_dir = save_plot_dir

    def begin_fit(self):
        self.run.list_loss = np.empty(self.run.epochs)

    def begin_epoch(self):
        self.total_loss = 0.0

    def after_batch(self):
        self.total_loss += self.run.loss_value.detach().item()

    def after_train(self):
        logger.info(
            f"Epoch = {self.run.epoch}, \
            Generator Training Loss = {self.total_loss/ (self.run.n_iters )}"
        )
        self.run.list_loss[self.run.epoch] = self.total_loss / self.run.n_iters

    def _save_plot(self, loss: np.ndarray):
        file_name = "pretrainng_loss.png"
        plt.plot(loss, label="Loss")
        # plt.plot(EMA(loss, alpha=self.EMA_value), label="EMA loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss values")
        plt.legend()
        plt.savefig(self.save_plot_dir / file_name)
        plt.close()

    def after_fit(self):
        self._save_plot(self.run.list_loss)
        np.savetxt(self.save_plot_dir / "loss.txt", self.run.list_loss)


class PlotInterpolatedSamples(Callback):

    _order = 20

    def __repr__(self):
        return "PlotInterpolatedSamples"

    def __init__(
        self,
        replace_factor: float,
        freq: int = 1,
        save_plot_dir: pathlib.PosixPath = pathlib.Path.cwd(),
    ):
        self.freq = freq
        self.save_plot_dir = save_plot_dir
        self.replace_factor = replace_factor

    def _min_max_scaling(self, target):
        _min, _ = torch.min(target, dim=1)
        _max, _ = torch.max(target, dim=1)

        _min = _min.unsqueeze(dim=1)

        _max = _max.unsqueeze(dim=1)
        scaling_factor = _max - _min
        scaling_factor = torch.where(
            scaling_factor != 0,
            scaling_factor,
            torch.ones_like(scaling_factor),
        )
        scaled_target = (target - _min) / scaling_factor
        return scaled_target

    def _get_input(self):
        shape = self.run.item[0].unsqueeze(1).shape
        modified_input = self.run.item[0].view(-1).clone().detach()
        nb_elmnt_change = int(shape[0] * shape[2] * self.replace_factor)
        idx_to_change = torch.randperm(len(modified_input))[:nb_elmnt_change]
        noise = torch.randn(nb_elmnt_change, device=self.run.device)
        modified_input[idx_to_change] = noise
        modified_input = self._min_max_scaling(
            modified_input.view(shape).squeeze(1)
        ).unsqueeze(1)
        return torch.cat(
            (self.run.item[1], modified_input),
            dim=1,
        )

    def _save_samples(
        self,
        ts: torch.Tensor,
        reconstruction: torch.Tensor,
        _input: torch.Tensor,
    ):
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(50, 50))
        ax = [item for sublist in ax for item in sublist]
        reconstruction = reconstruction.cpu().numpy()
        _input = _input.cpu().numpy()
        for idx, original_ts in enumerate(ts[:16, :].cpu().numpy()):
            ax[idx].plot(original_ts, label="original")
            ax[idx].plot(reconstruction[idx, :], label="reconstruction")
            ax[idx].plot(_input[idx, -1, :], label="input")
            ax[idx].legend()

        path = Path(self.save_plot_dir / "samples")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"reconstruction_samples_{self.run.epoch}.pdf")
        plt.close()

    def after_val(self):
        if self.run.epoch % self.freq == 0:
            to_interpolate = self._get_input()
            reconstruction = self.run.model(
                to_interpolate, net="generator"
            ).squeeze(1)
            self._save_samples(
                self.run.item[0], reconstruction, to_interpolate
            )


class SaveModelSingle(Callback):
    """Callback to save models and optimizer state

    Attributes:
        save_model_dir:
            Path where to save the models.
    """

    _order = 20

    def __repr__(self):
        return f"Save Model CallBack {self.save_model_dir} \n"

    def __init__(self, save_model_dir: pathlib.PosixPath = pathlib.Path.cwd()):
        self.save_model_dir = save_model_dir

    def after_epoch(self):
        checkpoint = f"checkpoint_pretrain_{self.run.epoch}.pt"
        if self.run.epoch % 10 == 0:
            torch.save(
                {
                    "epoch": self.run.epoch,
                    "generator_model_state_dict": self.run.model.state_dict(),
                    "generator_opt_state_dict": self.run.opt.state_dict(),
                    "generator_loss": self.run.list_loss[self.run.epoch],
                },
                self.save_model_dir / checkpoint,
            )


class JointTrainingPytorch(Callback):
    """Callback that implements the training loop of the generator and discriminator.

    Attributes:
        nb_step_discrim:
            number of discriminator step generator step
        schedule:
            list of integers. Each integer is the epoch number from which the size of the time
            series will be doubled. If the list is empty then it is never doubled and
            it stays at the original size.
            e.g.:
                schedule = [10,20]. at epoch 9 depth = 0, epoch 11 depth = 1, epoch 21 depth = 2
    """

    _order = 10

    def __repr__(self):
        return f"Joint Training Callback : \n \
            \t nb_step_discrim : {self.nb_step_discrim} \n \
            \t schedule : {self.schedule}. \n \
            \t Pre-train schedule : {self.pretrain_schedule} \n \
            \t Initial number of step for discriminator: {self.d_step_temp} \n \
            \t Pre-train of discriminator for {self.nb_gen_step_goal} number of step of generator \n \
            \t Number of epochs used to fade in a new layer {self.nb_epoch_fade_in_new_layer}"

    def __init__(
        self,
        nb_step_discrim: int = 1,
        schedule: List[int] = None,
        nb_epoch_fade_in_new_layer: int = None,
        device: str = "cpu",
        use_loss: str = "wgan",
        momment_loss: float = 0,
        scaling_penalty: float = 0,
        encoder_network: nn.Module = None,
        encoder_network_factor: float = None,
    ):
        super(JointTrainingPytorch, self).__init__()
        self.nb_step_discrim = nb_step_discrim
        self.d_step = nb_step_discrim
        self.d_step_temp = 50
        self.nb_gen_step = 0
        self.nb_gen_step_goal = 25
        self.depth = 0
        self.schedule = schedule
        self.pretrain_schedule = []
        self.nb_epoch_fade_in_new_layer = nb_epoch_fade_in_new_layer
        for k in schedule:
            self.pretrain_schedule.append((k, k + nb_epoch_fade_in_new_layer))
        self.nb_stage = len(schedule) if schedule else 0
        self.nb_gen_step = 0
        assert (device == "cpu") or (
            device == "gpu"
        ), "Either device==cpu or device==gpu"
        self.device = device
        self.relu = nn.ReLU()
        assert (
            use_loss == "wgan" or use_loss == "hinge" or use_loss == "lsgan"
        ), "use_loss must be either wgan or hinge or lsgan"
        self.use_loss = use_loss
        self.momment_loss = momment_loss
        self.scaling_penalty = scaling_penalty
        if encoder_network is not None:
            for p in encoder_network.parameters():
                p.requires_grad = False
        self.encoder_network = encoder_network

        self.encoder_network_factor = encoder_network_factor
        logger.info(f"encoder_network: {self.encoder_network}")
        logger.info(f"encoder_network_factor: {self.encoder_network_factor}")

    def _residual(self):
        if self.nb_stage >= 0:
            if len(self.pretrain_schedule) > 0:
                self.start_epoch_test = self.pretrain_schedule[0][0]
                self.end_epoch_test = self.pretrain_schedule[0][1]
                if (
                    self.end_epoch_test
                    > self.run.epoch
                    > self.start_epoch_test
                ):
                    self.start_epoch = self.pretrain_schedule[0][0]
                    self.end_epoch = self.pretrain_schedule[0][1]
                    self.pretrain_schedule.pop(0)
        try:
            if self.end_epoch >= self.run.epoch >= self.start_epoch:
                residual_factor = self._linear_interpolation(
                    self.start_epoch, self.end_epoch, self.run.epoch
                )
                self.run.model.generator.residual_factor = residual_factor
                self.run.model.discriminator.residual_factor = residual_factor

                return True
            else:
                return False
        except Exception:
            return False

    def _increase_depth(self):
        # Piece of code to compute at which depth the tensor should flow
        if self.nb_stage > 0:
            self.update_epoch = self.schedule[0]
            if self.run.epoch > self.update_epoch:
                self.depth += 1
                self.nb_stage -= 1
                self.schedule.pop(0)

    def _linear_interpolation(self, alpha, beta, x):
        assert beta > alpha
        return (x - alpha) / (beta - alpha)

    def _momment_loss(self, preds, target):
        std_loss = torch.abs(preds.std(dim=1) - target.std(dim=1)).mean()
        mean_loss = torch.abs(preds.mean(dim=1) - target.mean(dim=1)).mean()
        momment_loss = std_loss + mean_loss
        return self.momment_loss * momment_loss

    def _wgan_loss(
        self,
        fake_ts: torch.Tensor,
        reduced_target: torch.Tensor = None,
        step: str = "generator",
    ):
        if step == "generator":
            loss = -self.run.model(
                x=fake_ts,
                tf=self.run.item[1],
                depth=self.depth,
                net="discriminator",
                residual=self._residual(),
                feat_static_cat=self.run.item[3],
            ).mean()
            if (
                self.encoder_network_factor is not None
                and self.encoder_network is not None
            ):
                loss += self.encoder_network_factor * self._encoder_loss(
                    fake_ts, reduced_target
                )

        elif step == "discriminator":
            loss = -(
                self.run.model(
                    x=reduced_target,
                    tf=self.run.item[1],
                    depth=self.depth,
                    net="discriminator",
                    residual=self._residual(),
                    feat_static_cat=self.run.item[3],
                )
                - self.run.model(
                    x=fake_ts.detach(),
                    tf=self.run.item[1],
                    depth=self.depth,
                    net="discriminator",
                    residual=self._residual(),
                    feat_static_cat=self.run.item[3],
                )
            ).mean()

        return loss

    def _hinge_loss_gan(
        self,
        fake_ts: torch.Tensor,
        reduced_target: torch.Tensor = None,
        step: str = "generator",
    ):

        if step == "generator":
            loss = -self.run.model(
                x=fake_ts,
                tf=self.run.item[1],
                depth=self.depth,
                net="discriminator",
                residual=self._residual(),
                feat_static_cat=self.run.item[3],
            ).mean()

            if (
                self.encoder_network_factor is not None
                and self.encoder_network is not None
            ):
                loss += self.encoder_network_factor * self._encoder_loss(
                    fake_ts, reduced_target
                )

        elif step == "discriminator":

            loss = (
                -self.relu(
                    -1
                    + self.run.model(
                        x=reduced_target,
                        tf=self.run.item[1],
                        depth=self.depth,
                        net="discriminator",
                        residual=self._residual(),
                        feat_static_cat=self.run.item[3],
                    )
                ).mean()
                - self.relu(
                    -1
                    - self.run.model(
                        x=fake_ts.detach(),
                        tf=self.run.item[1],
                        depth=self.depth,
                        net="discriminator",
                        residual=self._residual(),
                        feat_static_cat=self.run.item[3],
                    )
                ).mean()
            )

        return loss

    def _lsgan_loss_gan(
        self,
        fake_ts: torch.Tensor,
        reduced_target: torch.Tensor = None,
        step: str = "generator",
    ):

        if step == "generator":
            loss = (
                self.run.model(
                    x=fake_ts,
                    tf=self.run.item[1],
                    depth=self.depth,
                    net="discriminator",
                    residual=self._residual(),
                    feat_static_cat=self.run.item[3],
                )
                - 1
            )
            loss = 0.5 * torch.square(loss).mean()

            if (
                self.encoder_network_factor is not None
                and self.encoder_network is not None
            ):
                loss += self.encoder_network_factor * self._encoder_loss(
                    fake_ts, reduced_target
                )

        elif step == "discriminator":
            loss_fake = (
                0.5
                * torch.square(
                    self.run.model(
                        x=fake_ts.detach(),
                        tf=self.run.item[1],
                        depth=self.depth,
                        net="discriminator",
                        residual=self._residual(),
                        feat_static_cat=self.run.item[3],
                    )
                ).mean()
            )

            loss_real = (
                0.5
                * torch.square(
                    self.run.model(
                        x=reduced_target,
                        tf=self.run.item[1],
                        depth=self.depth,
                        net="discriminator",
                        residual=self._residual(),
                        feat_static_cat=self.run.item[3],
                    )
                    - 1
                ).mean()
            )

            loss = loss_real + loss_fake
        return loss

    def _scaling_penalty(self, preds):
        return self.scaling_penalty * torch.sum(
            self.relu(preds - 1) + self.relu(-preds)
        )

    def _encoder_loss(
        self,
        fake_ts: torch.Tensor,
        reduced_target: torch.Tensor,
    ):
        fake_embedding = self.encoder_network(fake_ts)
        real_embedding = self.encoder_network(reduced_target)
        loss = F.mse_loss(real_embedding, fake_embedding)
        cosine_loss = F.cosine_similarity(
            fake_embedding, real_embedding
        ).mean()
        return loss + cosine_loss

    def _loss_gan(
        self,
        fake_ts: torch.Tensor,
        reduced_target: torch.Tensor,
        step: str = "generator",
        loss_type: str = "lsgan",
    ):
        logger.debug(f"Loss used is {loss_type}")
        logger.debug(
            f"Shape of fake_ts tensor {fake_ts.shape}, Shape of reduced_target tensor is {reduced_target.shape}"
        )

        if loss_type == "wgan":
            loss = self._wgan_loss(
                fake_ts=fake_ts, reduced_target=reduced_target, step=step
            )
        elif loss_type == "hinge":
            loss = self._hinge_loss_gan(
                fake_ts=fake_ts, reduced_target=reduced_target, step=step
            )
        elif loss_type == "lsgan":
            loss = self._lsgan_loss_gan(
                fake_ts=fake_ts, reduced_target=reduced_target, step=step
            )

        return loss

    def forward_pass(self):
        # Piece of code so that the discriminator is updated self.d_step_temp times more than the generator at the begining
        if self.nb_gen_step < self.nb_gen_step_goal:
            self.nb_step_discrim = self.d_step_temp
        else:
            self.nb_step_discrim = self.d_step

        self._increase_depth()
        self.run.depth = self.depth
        if self.run.itr % (self.nb_step_discrim + 1) == 0:  # Generator step
            for p in self.run.model.discriminator.parameters():
                p.requires_grad = False
            generated = self.run.model(
                x=self.run.item[2],
                depth=self.depth,
                net="generator",
                residual=self._residual(),
                feat_static_cat=self.run.item[3],
                context=self.run.item[4],
            )
            reduce_factor = int(
                log2(self.run.model.discriminator.target_len)
            ) - int(log2(generated.size(2)))
            reduced_target = F.avg_pool1d(
                self.run.item[0].unsqueeze(1),
                kernel_size=2 ** reduce_factor,
            )

            # if self.use_loss == "wgan":
            #     self.run.score_g = self._wgan_loss(
            #         fake_ts=generated, reduced_target=reduced_target, step="generator"
            #     )
            # elif self.use_loss == "hinge":
            #     self.run.score_g = self._hinge_loss_gan(
            #         fake_ts=generated, reduced_target=reduced_target,step="generator"
            #     )
            # elif self.use_loss == "lsgan":
            #     self.run.score_g = self._lsgan_loss_gan(
            #         fake_ts=generated, reduced_target=reduced_target,step="generator"
            #     )

            self.run.score_g = self._loss_gan(
                fake_ts=generated,
                reduced_target=reduced_target,
                step="generator",
                loss_type=self.use_loss,
            )

            if self.momment_loss != 0:
                self.run.score_g += self._momment_loss(
                    generated.squeeze(1), reduced_target.squeeze(1)
                )

            if self.scaling_penalty != 0:
                self.run.score_g += self._scaling_penalty(generated.squeeze(1))

            self.nb_gen_step += 1
        else:  # Discriminator step
            with torch.no_grad():
                fake_ts = self.run.model(
                    self.run.item[2],
                    depth=self.depth,
                    net="generator",
                    residual=self._residual(),
                    feat_static_cat=self.run.item[3],
                    context=self.run.item[4],
                )
            reduce_factor = int(
                log2(self.run.model.discriminator.target_len)
            ) - int(log2(fake_ts.size(2)))
            reduced_target = F.avg_pool1d(
                self.run.item[0].unsqueeze(1),
                kernel_size=2 ** reduce_factor,
            )
            # if self.use_loss == "wgan":
            #     self.run.score_d = self._wgan_loss(
            #         fake_ts=fake_ts,
            #         reduced_target=reduced_target,
            #         step="discriminator",
            #     )
            # elif self.use_loss == "hinge":
            #     self.run.score_d = self._hinge_loss_gan(
            #         fake_ts=fake_ts,
            #         reduced_target=reduced_target,
            #         step="discriminator",
            #     )
            # elif self.use_loss == "lsgan":
            #     self.run.score_d = self._lsgan_loss_gan(
            #         fake_ts=fake_ts,
            #         reduced_target=reduced_target,
            #         step="discriminator",
            #     )

            self.run.score_d = self._loss_gan(
                fake_ts=fake_ts,
                reduced_target=reduced_target,
                step="discriminator",
                loss_type=self.use_loss,
            )

    def backward(self):
        if self.run.itr % (self.nb_step_discrim + 1) == 0:  # Generator step
            self.run.score_g.backward()
            for p in self.run.model.discriminator.parameters():
                p.requires_grad = True

        else:  # Discriminator step
            self.run.score_d.backward()

    def step(self):
        if self.run.itr % (self.nb_step_discrim + 1) == 0:  # Generator step
            self.run.opt.step(net="generator")
        else:  # Discriminator step
            self.run.opt.step(net="discriminator")

    def zero_grad(self):
        if self.run.itr % (self.nb_step_discrim + 1) == 0:  # Generator step
            self.run.opt.zero_grad("generator")
        else:  # Discriminator step
            self.run.opt.zero_grad("discriminator")

    def begin_validate(self):
        return True


class PlotSamplesandTSNEPytorch(Callback):
    """Plot samples every given epoch and t-SNE plot them too"""

    _order = 20

    def __init__(
        self,
        freq: int,
        freq_tsne: int,
        save_plot_dir: pathlib.PosixPath,
    ):
        super(PlotSamplesandTSNEPytorch, self).__init__()
        self.freq = freq
        self.freq_tsne = freq_tsne
        self.save_plot_dir = save_plot_dir

    def _get_noise(self):
        shape = self.run.item[0].unsqueeze(1).shape
        noise = torch.randn(shape, device=self.run.device)
        return torch.cat(
            (
                self.run.item[1],
                noise,
            ),
            dim=1,
        )

    def _save_samples(self, fake_ts: torch.Tensor, fake=True):
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(50, 50))
        ax = [item for sublist in ax for item in sublist]
        for idx, f_ts in enumerate(fake_ts[:16, :].cpu().numpy()):
            # f_ts = f_ts.cpu().numpy()
            ax[idx].plot(f_ts)
        if fake:
            path = Path(self.save_plot_dir / "samples")
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(path / f"fake_samples_{self.run.epoch}.pdf")
        else:
            path = Path(self.save_plot_dir / "samples")
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(path / f"real_samples_{self.run.epoch}.pdf")
        plt.close()

    def _compute_tsne(self, fake: torch.Tensor, real: torch.Tensor):
        length = fake.size(1)
        nb_fake = fake.size(0)
        if length > 50:
            fake = PCA(n_components=min(50, nb_fake)).fit_transform(
                fake.numpy()
            )
            real = PCA(n_components=min(50, nb_fake)).fit_transform(
                real.numpy()
            )
        full = np.concatenate((fake, real), axis=0)
        full_tsne = TSNE().fit_transform(full)
        fake_tsne = full_tsne[:nb_fake, :]
        real_tsne = full_tsne[nb_fake:, :]

        return fake_tsne, real_tsne

    def _plot_tsne(self, fake_tsne, real_tsne):
        path = Path(self.save_plot_dir / "TSNE")
        path.mkdir(parents=True, exist_ok=True)
        plt.plot(
            fake_tsne[:, 0],
            fake_tsne[:, 1],
            "o",
            label="fake samples",
            alpha=0.4,
        )
        plt.plot(
            real_tsne[:, 0],
            real_tsne[:, 1],
            "o",
            label="real samples",
            alpha=0.4,
        )
        plt.legend()
        plt.savefig(path / f"TSNE_{self.run.epoch}.pdf")
        plt.close()

    def after_val(self):
        if (
            self.run.epoch % self.freq == 0
            or self.run.epoch == self.run.epochs - 1
        ):
            noise = self._get_noise()
            fake_ts = self.run.model(
                noise,
                depth=self.run.depth,
                net="generator",
                residual=False,
                feat_static_cat=self.run.item[3],
                context=self.run.item[4],
            ).squeeze(1)
            self._save_samples(fake_ts)
            reduce_factor = int(
                log2(self.run.model.discriminator.target_len)
            ) - int(log2(fake_ts.size(1)))
            reduced_target = F.avg_pool1d(
                self.run.item[0].unsqueeze(1),
                kernel_size=2 ** reduce_factor,
            ).squeeze(1)
            self._save_samples(reduced_target, fake=False)
            if (
                self.run.epoch % self.freq_tsne == 0
                or self.run.epoch == self.run.epochs - 1
            ):
                fake_tsne, real_tsne = self._compute_tsne(
                    fake_ts.cpu(), reduced_target.cpu()
                )
                self._plot_tsne(fake_tsne, real_tsne)


class GANLossCheck(Callback):
    """
    Callback to print the loss of the generator, the discriminator and the embedder.
    Appends also the results to list accessible from the runner class.
    """

    _order = 25

    def __repr__(self):
        return f"GANLossCheck Callback: \n \
            \t save_plot_dir : {self.save_plot_dir} \n \
            \t nb_step_discrim : {self.nb_step_discrim} \n \
            \t EMA_values : {self.EMA_value}. \n"

    def __init__(
        self,
        save_plot_dir: pathlib.PosixPath,
        nb_step_discrim: int = 1,
        EMA_value: float = 0.2,
    ):
        super(GANLossCheck, self).__init__()
        self.nb_step_discrim = nb_step_discrim
        self.save_plot_dir = save_plot_dir
        self.EMA_value = EMA_value

    def begin_fit(self):
        (self.run.list_loss_gen, self.run.list_loss_discrim,) = (
            np.empty(self.run.epochs),
            np.empty(self.run.epochs),
        )

    def begin_epoch(self):
        self.total_loss_gen, self.total_loss_discrim = (
            0.0,
            0.0,
        )

    def after_batch(self):
        if self.run.itr % (self.nb_step_discrim + 1) == 0:
            self.total_loss_gen += self.run.score_g.detach().item()
        else:
            self.total_loss_discrim += self.run.score_d.detach().item()

    def after_train(self):
        if self.run.epoch % 1 == 0:
            logger.info(
                f"Epoch = {self.run.epoch}, \
                Generator Training Loss = {(self.nb_step_discrim + 1) * self.total_loss_gen / (self.run.n_iters * self.nb_step_discrim)}, \
                Discriminator Training Loss = {(self.nb_step_discrim + 1) * self.total_loss_discrim / self.run.n_iters} "
            )
        self.run.list_loss_gen[self.run.epoch] = (
            (self.nb_step_discrim + 1)
            * self.total_loss_gen
            / (self.run.n_iters * self.nb_step_discrim)
        )
        self.run.list_loss_discrim[self.run.epoch] = (
            (self.nb_step_discrim + 1)
            * self.total_loss_discrim
            / self.run.n_iters
        )

    def _save_plot(self, loss: np.ndarray, prefix: str):
        file_name = prefix + "_loss.png"
        plt.plot(loss, label="Loss")
        # plt.plot(EMA(loss, alpha=self.EMA_value), label="EMA loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(prefix + " Loss values")
        plt.legend()
        plt.savefig(self.save_plot_dir / file_name)
        plt.close()

    def after_fit(self):
        self._save_plot(self.run.list_loss_gen, "Generator")
        self._save_plot(self.run.list_loss_discrim, "Discriminator")
        np.savetxt(
            self.save_plot_dir / "Generator_loss.txt", self.run.list_loss_gen
        )
        np.savetxt(
            self.save_plot_dir / "Discriminator_loss.txt",
            self.run.list_loss_discrim,
        )


class SaveModel(Callback):
    """Callback to save models and optimizer state

    Attributes:
        save_model_dir:
            Path where to save the models.
    """

    _order = 20

    def __repr__(self):
        return f"Save Model CallBack {self.save_model_dir} \n"

    def __init__(self, save_model_dir: pathlib.PosixPath = pathlib.Path.cwd()):
        self.save_model_dir = save_model_dir

    def after_epoch(self):
        checkpoint = f"checkpoint_{self.run.epoch}.pt"
        if self.run.epoch % 1000 == 0 or self.run.epoch == self.run.epochs - 1:

            torch.save(
                {
                    "epoch": self.run.epoch,
                    "generator_model_state_dict": self.run.model.generator.state_dict(),
                    "discriminator_model_state_dict": self.run.model.discriminator.state_dict(),
                    "generator_opt_state_dict": self.run.opt.get_opt(
                        net="generator"
                    ).state_dict(),
                    "discriminator_opt_state_dict": self.run.opt.get_opt(
                        net="discriminator"
                    ).state_dict(),
                    "generator_loss": self.run.list_loss_gen[self.run.epoch],
                    "discriminator_loss": self.run.list_loss_discrim[
                        self.run.epoch
                    ],
                },
                self.save_model_dir / checkpoint,
            )
