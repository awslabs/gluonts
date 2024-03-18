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
import torch
import wandb
import os
import pytorch_lightning as pl
import time
import numpy as np

from duru.vdvae_conv import VDVAEConv
from duru.optim import linear_warmup
from utils import compute_gradient_norm
from plotting import plot_kl_cum_sum, plot_state_norm

import matplotlib.pyplot as plt

from duru.optim import optimizer_step_hvae_eval, init_optimizer_scheduler
from utils import get_sample_for_visualization, compute_blocks_per_res
from gluonts.torch.model.predictor import PyTorchPredictor


class VDVAEConvPL(VDVAEConv, pl.LightningModule):
    def __init__(self, H, hvae_eval=None):
        super().__init__(H=H)
        # Important: This property activates manual optimization in PL.
        self.automatic_optimization = False
        self.hvae_eval = hvae_eval  # just for the integration of PL; is None if hvae_eval model itself!

    def training_step(self, batch, batch_idx):
        t0_train = time.time()

        # retrieving optimizer in PL, and resetting gradients
        optimizer = self.optimizers()
        optimizer.zero_grad()

        try:
            if self.H.conditinoal:
                x_context, x_forecast = (
                    batch["past_target"],
                    batch["future_target"],
                )
                (
                    elbo,
                    distortion,
                    rate,
                    kl_list,
                    state_norm_enc_list,
                    state_norm_dec_list,
                ) = self.forward(x_context=x_context, x_forecast=x_forecast)
            else:
                x_forecast = batch["future_target"]
                (
                    elbo,
                    distortion,
                    rate,
                    kl_list,
                    state_norm_enc_list,
                    state_norm_dec_list,
                ) = self.forward(x_forecast=x_forecast)

            # TODO ------------------
            # TODO before calling forward!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x = preprocess_fn(
                x
            )  # TODO handle as part of the transformers of gluonts?
            # TODO -------------

            if "cuda" in self.H.device and torch.cuda.device_count() > 1:
                # more than one GPU are used with DataParallel --> chunks the batch into torch.cuda.device_count() parts, which are processed on a replicated model on each of the GPUs.
                # Warning from documentation: "When module returns a scalar (i.e., 0-dimensional tensor) in forward(), this wrapper will return a vector of length equal to number of devices used in data parallelism, containing the result from each device."
                # this is why a tensor is now suddenly returned for every value which has been a scalar before
                # we account for that by averaging these tensors (of length torch.cuda.device_count())
                elbo, distortion, rate = (
                    torch.mean(elbo),
                    torch.mean(distortion),
                    torch.mean(rate),
                )

            # backward call in PL
            self.manual_backward(elbo)

            # if H.grad_clip_threshold != -1:
            #     grad_norm_before_clipping = torch.nn.utils.clip_grad_norm_(hvae.parameters(), max_norm=H.grad_clip_threshold, norm_type=2.0, error_if_nonfinite=True).item()
            #     grad_clip_count += 1
            # else:
            #     # just compute the gradient norm
            grad_norm_before_clipping = compute_gradient_norm(
                self.parameters()
            )

        except ValueError as e:
            print(e)  # print out exception as if it was thrown
            self.skipped_updates_count += 1
            self.nan_skip_count += 1
            # Note: this implementation may cause very high values for counted logger metrics,
            # if exception is thrown and logging in this iteration should be done (because logging and resetting of counters is skipped)
            # TODO is this the correct way of doing it in PL?
            return None  # skipping the step, see https://github.com/Lightning-AI/lightning/issues/3323

        # -------------------------

        elbo_nan_count = torch.isnan(elbo).sum()
        distortion_nan_count = torch.isnan(distortion).sum()
        rate_nan_count = torch.isnan(rate).sum()
        self.elbo_has_nan_count += 1 if elbo_nan_count > 0 else 0
        self.distortion_has_nan_count += 1 if distortion_nan_count > 0 else 0
        self.rate_has_nan_count += 1 if rate_nan_count > 0 else 0

        # perform optimizer step (updating the parameters) if 1) gradient norm before clipping is below `H.grad_skip_threshold` 2) neither the distortion nor rate contains NANs
        # TODO change update step
        if (
            (
                self.H.grad_skip_threshold == -1
                or grad_norm_before_clipping < self.H.grad_skip_threshold
            )
            and distortion_nan_count == 0
            and rate_nan_count == 0
        ):
            optimizer.step()
            optimizer_step_hvae_eval(
                self, self.hvae_eval, self.H.ema_rate
            )  # first argument: the model itself (which has a parameters instance variable)
        else:
            self.skipped_updates_count += 1
            self.grad_skip_count += 1
        # regardless of whether the optimizer took a step, update the scheduler
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

        t1_train = time.time()
        time_in_sec_training_step = t1_train - t0_train  # time in seconds

        # train metrics logging
        if (
            self.i % self.H.iters_per_train_metrics == 0
            or self.i in self.early_iterations
        ):
            # Note: kl_list not logged
            log_dict = {
                "train/epoch": self.i / self.H.n_train_iters_per_epoch,
                "train/elbo": elbo,
                "train/distortion": distortion,
                "train/rate": rate,
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/elbo_has_nan_count": self.elbo_has_nan_count,
                "train/distortion_has_nan_count": self.distortion_has_nan_count,
                "train/rate_has_nan_count": self.rate_has_nan_count,
                "train/train_step_seconds": time_in_sec_training_step,
                "train/grad_norm_before_clipping": grad_norm_before_clipping,
                "train/skipped_updates": self.skipped_updates_count,
                "train/nan_skip_count": self.nan_skip_count,
                "train/grad_skip_count": self.grad_skip_count,
            }

            self.log_dict(log_dict, on_step=True)
            # wandb.log(log_dict, step=self.i)
            # reset counters
            (
                elbo_has_nan_count,
                distortion_has_nan_count,
                rate_has_nan_count,
                grad_clip_count,
                skipped_updates_count,
                nan_skip_count,
                grad_skip_count,
            ) = (0, 0, 0, 0, 0, 0, 0)

        # training data visualizations
        with torch.no_grad():
            if self.H.vis_train and self.i % self.H.iters_per_vis == 0:

                # TODO
                # if H.vis_train_recon:
                #     recon_x_list = vis_train_x_list
                #     if torch.cuda.device_count() > 1 and 'cuda' in H.device:
                #         recon_x_hat = hvae.module.get_recon(vis_train_x_input)
                #     else:
                #         recon_x_hat = hvae.get_recon(vis_train_x_input)
                #     recon_x_hat_list = [recon_x_hat[i].cpu().detach() for i in range(recon_x_hat.shape[0])]  # convert to list
                #     fig = plot_inputs_and_recons_torch_grid(x_list=recon_x_list, x_hat_list=recon_x_hat_list, recon_n_rows=H.recon_n_rows, recon_n_pairs_col=H.recon_n_pairs_col)
                #     wandb.log({'train_vis' + "/inputs and reconstructions": wandb.Image(plt)}, step=i)
                #     plt.close(fig=fig)  # close the figure
                if self.H.vis_train_cum_kl:
                    kl_cum_sum = torch.cumsum(
                        input=torch.stack(kl_list, dim=1), dim=1
                    ).cpu()  # this uses the KLs of one mini-batch, could instead accumulate a couple of batches; cpu (and later numpy) conversion required for plotting
                    fig = plot_kl_cum_sum(
                        kl_cum_sum=kl_cum_sum, dataset=self.H.dataset
                    )
                    log_key = "train_vis" + "/Cumulative, batch-averaged KLs"
                    wandb.log({log_key: wandb.Image(plt)}, step=self.i)
                    plt.close(fig=fig)  # close the figure

            if self.H.vis_eval and self.i % self.H.iters_per_vis == 0:
                if self.H.vis_uncond_samples:
                    ts_list = []
                    for temp in self.H.uncond_samples_temp_list:
                        uncond_samples = self.hvae_eval.get_mean(
                            self.H.uncond_samples_n_ts_per_temp, temp=temp
                        )
                        ts_list += [
                            uncond_samples[i].cpu().detach()
                            for i in range(uncond_samples.shape[0])
                        ]
                    # fig = plot_uncond_samples(img_list=img_list, uncond_samples_n_rows=len(H.uncond_samples_temp_list), uncond_samples_n_cols=H.uncond_samples_n_ts_per_temp)  # TODO
                    if self.H.test_id is not None:
                        type = "test_vis"
                    else:
                        type = "val_vis"
                    wandb.log(
                        {type + "/unconditional samples": wandb.Image(plt)},
                        step=self.i,
                    )
                    plt.close(fig=fig)  # close the figure

        # Models, optimizer and scheduler saving 'checkpoint'
        # TODO simplify logging ("checkpointing") with PL checkpointing
        if (
            self.i % self.H.iters_per_model_save == 0
            and not self.H.iters_per_model_save == -1
        ):  # -1 means not saving model
            prefix = "iter-%d-" % (self.i)
            torch.save(
                self.state_dict(),
                os.path.join(wandb.run.dir, prefix + "model.th"),
            )
            torch.save(
                self.hvae_eval.state_dict(),
                os.path.join(wandb.run.dir, prefix + "model_eval.th"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(wandb.run.dir, prefix + "optimizer.th"),
            )
            torch.save(
                lr_scheduler.state_dict(),
                os.path.join(wandb.run.dir, prefix + "scheduler.th"),
            )
            torch.save(
                self.i, os.path.join(wandb.run.dir, "last_save_iter.th")
            )  # save without prefix

        # in very end of iteration: increment iteration count
        self.i += 1

    # TODO how to pass None by default for x_context?
    def validation_step(self, x_forecast, x_context, batch_idx):
        t0_eval = time.time()

        # TODO probably all handled by pytorch lightning
        (
            elbo_list,
            elbo_filtered_list,
            distortion_list,
            rate_list,
            kl_list_list,
            state_norm_enc_list_list,
            state_norm_dec_list_list,
        ) = ([], [], [], [], [], [], [])
        # for j, x in enumerate(loader_eval):

        # TODO preprocessing ---------------------------------------------------
        if "cuda" in self.H.device:
            x = x.cuda(non_blocking=True)
        if self.H.conditional:
            x_context = preprocess_fn(x_context)
            x_forecast = preprocess_fn(x_forecast)
        else:
            x = preprocess_fn(x)

        if self.H.conditional:
            (
                elbo,
                distortion,
                rate,
                kl_list,
                state_norm_enc_list,
                state_norm_dec_list,
            ) = self.hvae_eval(x_context=x_context, x_forecast=x_forecast)
        else:
            (
                elbo,
                distortion,
                rate,
                kl_list,
                state_norm_enc_list,
                state_norm_dec_list,
            ) = self.hvae_eval(x_forecast=x_forecast)
        elbo_filtered, distortion_filtered, rate_filtered = (
            elbo[torch.isfinite(elbo)],
            distortion[torch.isfinite(distortion)],
            rate[torch.isfinite(rate)],
        )

        # if 'cuda' in self.H.device and torch.cuda.device_count() > 1:
        #     # more than one GPU are used with DataParallel --> chunks the batch into torch.cuda.device_count() parts, which are processed on a replicated model on each of the GPUs.
        #     # Warning from documentation: "When module returns a scalar (i.e., 0-dimensional tensor) in forward(), this wrapper will return a vector of length equal to number of devices used in data parallelism, containing the result from each device."
        #     # this is why a tensor is now suddenly returned for every value which has been a scalar before
        #     # we account for that by averaging these tensors (of length torch.cuda.device_count())
        #     elbo, distortion, rate = torch.mean(elbo), torch.mean(distortion), torch.mean(rate)

        # Note: might have to average across machines when using DataParallel --> see main.py, ll. 98
        # Note: if this blows up memory: could only append a certain number of KLs and state norms
        # TODO probably no need to average etc. --> probably done automatically via PL
        # append_to_list_((elbo, elbo_list), (distortion, distortion_list), (rate, rate_list),
        #                 (kl_list, kl_list_list), (state_norm_enc_list, state_norm_enc_list_list),
        #                 (state_norm_dec_list, state_norm_dec_list_list))

        # print("eval j: %d"%(j))

        # compute filtered metrics for elbo, distortion, rate (might contain infinite or nan values)
        # elbo_list, distortion_list, rate_list = [elbo.cpu().numpy() for elbo in elbo_list], [distortion.cpu().numpy() for distortion in distortion_list], [rate.cpu().numpy() for rate in rate_list]  # convert ot numpy
        # elbo_filtered_list, distortion_filtered_list, rate_filtered_list = finites_only(elbo_list, distortion_list, rate_list)
        # # compute means
        # elbo_mean, distortion_mean, rate_mean, elbo_filtered_mean, distortion_filtered_mean, rate_filtered_mean = np.mean(
        #     elbo_list), np.mean(distortion_list), np.mean(rate_list), np.mean(elbo_filtered_list), np.mean(
        #     distortion_filtered_list), np.mean(rate_filtered_list)

        with torch.no_grad():
            if self.H.vis_eval and self.i % self.H.iters_per_vis == 0:

                # TODO
                # if self.H.vis_eval_recon:
                #     recon_x_list = vis_eval_x_list
                #     recon_x_hat = hvae_eval.get_recon(vis_eval_x_input)
                #     recon_x_hat_list = [recon_x_hat[i].permute(2, 0, 1).cpu().detach() for i in range(recon_x_hat.shape[0])]  # convert to list
                #     fig = plot_inputs_and_recons_torch_grid(x_list=recon_x_list, x_hat_list=recon_x_hat_list, recon_n_rows=self.H.recon_n_rows, recon_n_pairs_col=self.H.recon_n_pairs_col)
                #     eval_type = 'test_vis' if self.H.test_id is not None else 'val_vis'
                #     wandb.log({eval_type + "/inputs and reconstructions": wandb.Image(plt)}, step=i)
                #     plt.close(fig=fig)  # close the figure

                # TODO
                # if self.H.vis_eval_posterior_prior:
                #     # 'front to back'
                #     fig = posterior_prior_front_to_back(self.H, hvae_eval, vis_eval_x, vis_eval_x_input)
                #     eval_type = 'test_vis' if self.H.test_id is not None else 'val_vis'
                #     wandb.log({eval_type + "/posterior-prior comparison": wandb.Image(plt)}, step=i)
                #     plt.close(fig=fig)  # close the figure
                #     # 'per resolution'
                #     fig = posterior_prior_per_res(self.H, hvae_eval, vis_eval_x, vis_eval_x_input)
                #     eval_type = 'test_vis' if self.H.test_id is not None else 'val_vis'
                #     wandb.log({eval_type + "/posterior-prior-per-res comparison": wandb.Image(plt)}, step=i)
                #     plt.close(fig=fig)  # close the figure

                if self.H.vis_eval_cum_kl:
                    # Note: when using the below code, and calling torch.cumsum with kl_list_concat instead of kl_list, one could use all eval batches instead of just one.
                    #               However, this would make the KL plot inconsistent between train and eval, and likewise, if the batch size is large enough, should not make a big difference anyway.
                    #               Hence not done for now.
                    # kl_list_list = kl_list_list[:self.H.cum_kl_n_batches]
                    # kl_list_concat = []
                    # for i, kl_list in enumerate(kl_list_list):
                    #     for j, kl in enumerate(kl_list):
                    #         if i == 0:
                    #             kl_list_concat.append(kl)
                    #         else:
                    #             kl_list_concat[j] = torch.cat((kl_list_concat[j], kl))

                    kl_cum_sum = torch.cumsum(
                        input=torch.stack(kl_list, dim=1), dim=1
                    ).cpu()  # this uses the KLs of one mini-batch, could instead accumulate a couple of batches; cpu (and later numpy) conversion required for plotting
                    fig = plot_kl_cum_sum(
                        kl_cum_sum=kl_cum_sum, dataset=self.H.dataset
                    )
                    eval_type = (
                        "test_vis" if self.H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/Cumulative, batch-averaged KLs"
                    wandb.log({log_key: wandb.Image(plt)}, step=self.i)
                    plt.close(fig=fig)  # close the figure

                if self.H.vis_eval_state_norm_enc:
                    # TODO consider adding input of 0th block (s.t. starts at 0)
                    # TODO consider implementing the same plot for training
                    state_norm_enc_list_list = state_norm_enc_list_list[
                        : self.H.state_norm_n_batches
                    ]
                    state_norm_enc = [
                        torch.stack(state_norm_list, dim=1)
                        for state_norm_list in state_norm_enc_list_list
                    ]
                    state_norm_enc = torch.cat(state_norm_enc, dim=0)
                    state_norm_enc = state_norm_enc.detach().cpu().numpy()

                    res_to_n_layers = compute_blocks_per_res(
                        spec=self.H.enc_spec,
                        enc_or_dec="enc",
                        input_resolution=self.H.context_length
                        + self.H.forecast_length,
                        count_up_down=True,
                    )

                    fig = plot_state_norm(
                        state_norm=state_norm_enc,
                        enc_or_dec="enc",
                        res_to_n_layers=res_to_n_layers,
                    )
                    eval_type = (
                        "test_vis" if self.H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/State L2-norm (Encoder)"
                    wandb.log({log_key: wandb.Image(plt)}, step=self.i)
                    plt.close(fig=fig)  # close the figure

                if self.H.vis_eval_state_norm_dec:
                    # TODO consider adding input of 0th block (s.t. starts at 0)
                    # TODO consider implementing the same plot for training
                    state_norm_dec_list_list = state_norm_dec_list_list[
                        : self.H.state_norm_n_batches
                    ]
                    state_norm_dec = [
                        torch.stack(state_norm_list, dim=1)
                        for state_norm_list in state_norm_dec_list_list
                    ]
                    state_norm_dec = torch.cat(state_norm_dec, dim=0)
                    state_norm_dec = state_norm_dec.detach().cpu().numpy()

                    res_to_n_layers = compute_blocks_per_res(
                        spec=self.H.dec_spec,
                        enc_or_dec="dec",
                        input_resolution=self.H.context_length
                        + self.H.forecast_length,
                        count_up_down=True,
                    )

                    fig = plot_state_norm(
                        state_norm=state_norm_dec,
                        enc_or_dec="dec",
                        res_to_n_layers=res_to_n_layers,
                    )
                    eval_type = (
                        "test_vis" if self.H.test_id is not None else "val_vis"
                    )
                    log_key = eval_type + "/State L2-norm (Decoder)"
                    wandb.log({log_key: wandb.Image(plt)}, step=self.i)
                    plt.close(fig=fig)  # close the figure

            t1_eval = time.time()
            time_in_sec_training_step = t1_eval - t0_eval  # time in seconds

            # do logging of scalar metrics
            # Note: kl_list, n_batches_eval not logged
            if self.H.test_id is not None:
                eval_type = "test/"
            else:
                eval_type = "val/"  #
            log_dict = {
                eval_type + "epoch": self.i / self.H.n_train_iters_per_epoch,
                eval_type + "elbo": elbo,
                eval_type + "distortion": distortion,
                eval_type + "rate": rate,
                eval_type + "elbo_filtered": elbo_filtered,
                eval_type + "distortion_filtered": distortion_filtered,
                eval_type + "rate_filtered": rate_filtered,
                eval_type
                + "total_eval_time_seconds": time_in_sec_training_step,
            }

            self.log_dict(log_dict, on_epoch=True)
            # wandb.log(log_dict, step=self.i)  # "As long as you keep passing the same value for step, W&B will collect the keys and values from each call in one unified dictionary. As soon you call wandb.log() with a different value for step than the previous one, W&B will write all the collected keys and values to the history"  (see https://docs.wandb.ai/guides/track/log)

            print("Evaluating. Iter %d: " % (self.i), log_dict)

    def test_step(self, x_forecast, x_context, batch_idx):
        self.validation_step(x_forecast, x_context, batch_idx)

    def get_predictor(self, preprocess_fn):
        # TODO pass device as argument?
        pytorch_predictor = PyTorchPredictor(
            prediction_length=self.H.forecast_length,
            input_names=["x_context"],
            prediction_net=self,
            batch_size=self.H.batch_size,
            input_transform=preprocess_fn,
        )

        return pytorch_predictor

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            weight_decay=self.H.adam_weight_decay,
            lr=self.H.lr,
            betas=(self.H.adam_beta_1, self.H.adam_beta_2),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup(self.H.warmup_iters)
        )

        config_dict = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return config_dict
