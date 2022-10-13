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
"""
VDVAE [1] for time series data.
This source code is partially based on and where suitable modifies the VDVAE code from https://github.com/openai/vdvae, which is Copyright (c) 2020 OpenAI. See LICENSE_vdvae for the associated license.
[1] https://arxiv.org/abs/2011.10650
"""

import torch
import torch as T
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from torch.nn import Conv2d
import numpy as np
import math
from gluonts.torch.model.predictor import PyTorchPredictor

import duru.hvae as hvae
from duru.likelihood import (
    GaussianLikelihoodSigmaEstimated,
    GaussianLikelihoodSigmaHyperparam,
)
from utils import (
    compute_stochastic_depth,
    compute_bottleneck_res,
    insert_channel_dim,
    get_stable_scale,
)

from duru.vdvae_conv import DownPooling, Decoder, Encoder, Bias_xs

from utils import gaussian_analytical_kl


class VDVAEfc(hvae.HVAE):
    def __init__(self, H):
        super(VDVAEfc, self).__init__()
        self.H = H
        self.encoder_forecast = Encoder(H)
        if H.conditional:
            self.encoder_context = Encoder(H, conditional=True)
        self.decoder = Decoder(H)
        if H.likelihood_type == "GaussianSigmaHyperparam":
            self.likelihood_model = GaussianLikelihoodSigmaHyperparam(H)
        elif H.likelihood_type == "GaussianSigmaEstimated":
            self.likelihood_model = GaussianLikelihoodSigmaEstimated(H)

    def input_flatten(self, x):
        # flatten the input tensor, just keeping the batch dim separate (assuming just one measurement)
        assert self.H.n_meas == 1
        x = torch.flatten(x, start_dim=1)

        return x

    def forward_regular(self, x_forecast, x_context=None):
        # required s.t. 2D tensors inside of VDVAE
        x_forecast = self.input_flatten(x_forecast)
        if x_context is not None:
            x_context = self.input_flatten(x_context)

        # print("VDVAE forward inputs: ", x_forecast.shape, x_context.shape)
        res_to_bottom_up_activations, state_norm_enc_list = self.encoder_forecast(
            x_forecast
        )
        if self.H.conditional:
            (
                res_to_bottom_up_activations_context,
                state_norm_enc_context_list,
            ) = self.encoder_context(x_context)
        else:
            res_to_bottom_up_activations_context = None
            state_norm_enc_context_list = [
                None
            ]  # required, because we return it in any case
        state, z_sample_cond_list, kl_list, state_norm_dec_list = self.decoder(
            res_to_bottom_up_activations=res_to_bottom_up_activations,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )
        if self.H.likelihood_type == "GaussianSigmaHyperparam":
            # convert back to 3 dimensions for likelihood computation (for consistency with VDVAE_conv)
            p_x_z_mean = state.unsqueeze(1)
            x_forecast = x_forecast.unsqueeze(1)

            distortion, p_x_z = self.likelihood_model.compute_distortion(
                p_x_z_mean=p_x_z_mean, x=x_forecast
            )
        elif self.H.likelihood_type == "GaussianSigmaEstimated":
            _, padded_forecast_length = (
                self.H.pad_context + self.H.context_length,
                self.H.pad_forecast + self.H.forecast_length,
            )
            x_forecast = x_forecast.unsqueeze(
                1
            )  # convert back to 3 dimensions for likelihood computation (for consistency with VDVAE_conv)
            p_x_z_mean = state[:, : self.H.n_meas * padded_forecast_length].reshape(
                state.shape[0], self.H.n_meas, padded_forecast_length
            )
            p_x_z_log_std = state[:, self.H.n_meas * padded_forecast_length :].reshape(
                state.shape[0], self.H.n_meas, padded_forecast_length
            )
            distortion, p_x_z = self.likelihood_model.compute_distortion(
                p_x_z_mean=p_x_z_mean, p_x_z_log_std=p_x_z_log_std, x=x_forecast
            )
        # compute rate
        rate = torch.zeros_like(distortion)
        for kl in kl_list:
            rate -= kl  # .sum(dim=(1, 2, 3))  --> no longer required, as now done when computing the KL divergences already
        # take negative (as done in bits or nats)
        distortion = -distortion
        rate = -rate
        # compute elbo
        elbo = distortion + rate
        # average over batch dim
        elbo, distortion, rate = (
            torch.mean(elbo, 0),
            torch.mean(distortion, 0),
            torch.mean(rate, 0),
        )

        # convert to "bits"
        elbo *= math.log2(math.e)
        distortion *= math.log2(math.e)
        rate *= math.log2(math.e)
        kl_list = [kl * math.log2(math.e) for kl in kl_list]
        # convert to "per dim"
        n_dims_data = np.prod(
            x_forecast.shape[1:]
        )  # 784, not 32x32 (even though that's the dimension after padding)
        elbo /= n_dims_data
        distortion /= n_dims_data
        rate /= n_dims_data
        kl_list = [kl / n_dims_data for kl in kl_list]

        if self.H.conditional:
            return (
                elbo,
                distortion,
                rate,
                kl_list,
                state_norm_enc_list,
                state_norm_enc_context_list,
                state_norm_dec_list,
                p_x_z,
            )
        else:
            return (
                elbo,
                distortion,
                rate,
                kl_list,
                state_norm_enc_list,
                state_norm_dec_list,
                p_x_z,
            )

    def sample_p(self, n_samples, temp, x_context=None, set_z_sample=None):
        # required s.t. 2D tensors inside of VDVAE
        if x_context is not None:
            x_context = self.input_flatten(x_context)

        if self.H.conditional:
            res_to_bottom_up_activations_context, _ = self.encoder_context(x_context)
        else:
            res_to_bottom_up_activations_context = None
        state, _ = self.decoder.sample_p(
            n_samples=n_samples,
            temp=temp,
            set_z_sample=set_z_sample,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )
        if self.H.likelihood_type == "GaussianSigmaHyperparam":
            # convert back to 3 dimensions for likelihood computation (for consistency with VDVAE_conv)
            p_x_z_mean = state.unsqueeze(1)

            x_sample = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)
        elif self.H.likelihood_type == "GaussianSigmaEstimated":
            _, padded_forecast_length = (
                self.H.pad_context + self.H.context_length,
                self.H.pad_forecast + self.H.forecast_length,
            )
            p_x_z_mean = state[:, : self.H.n_meas * padded_forecast_length].reshape(
                state.shape[0], self.H.n_meas, padded_forecast_length
            )
            p_x_z_log_std = state[:, self.H.n_meas * padded_forecast_length :].reshape(
                state.shape[0], self.H.n_meas, padded_forecast_length
            )
            x_sample = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)

        return x_sample

    def get_recon(self, x_forecast, x_context=None):
        # required s.t. 2D tensors inside of VDVAE
        x_forecast = self.input_flatten(x_forecast)
        if x_context is not None:
            x_context = self.input_flatten(x_context)

        res_to_bottom_up_activations, _ = self.encoder_forecast(x_forecast)
        if self.H.conditional:
            res_to_bottom_up_activations_context, _ = self.encoder_context(x_context)
        else:
            res_to_bottom_up_activations_context = None
        state = self.decoder.get_recon(
            res_to_bottom_up_activations=res_to_bottom_up_activations,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )

        if self.H.likelihood_type == "GaussianSigmaHyperparam":
            # convert back to 3 dimensions for likelihood computation (for consistency with VDVAE_conv)
            p_x_z_mean = state.unsqueeze(1)

            x_hat = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)
            p_x_z = self.likelihood_model.get_p_x_z(
                p_x_z_mean=p_x_z_mean.clone()
            )  # clone, since sometimes doing in-place operations of the x_hat --> p_x_z shall remain unaffected
        elif self.H.likelihood_type == "GaussianSigmaEstimated":
            _, padded_forecast_length = (
                self.H.pad_context + self.H.context_length,
                self.H.pad_forecast + self.H.forecast_length,
            )
            p_x_z_mean = state[:, : self.H.n_meas * padded_forecast_length].reshape(
                state.shape[0], self.H.n_meas, padded_forecast_length
            )
            p_x_z_log_std = state[:, self.H.n_meas * padded_forecast_length :].reshape(
                state.shape[0], self.H.n_meas, padded_forecast_length
            )
            x_hat = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)
            p_x_z = self.likelihood_model.get_p_x_z(
                p_x_z_mean=p_x_z_mean.clone(), p_x_z_log_std=p_x_z_log_std.clone()
            )  # clone, since sometimes doing in-place operations of the x_hat --> p_x_z shall remain unaffected

        return x_hat, p_x_z

    def get_cond_latent_samples(self, x_forecast, x_context=None):
        # required s.t. 2D tensors inside of VDVAE
        x_forecast = self.input_flatten(x_forecast)
        if x_context is not None:
            x_context = self.input_flatten(x_context)

        if self.H.conditional:
            res_to_bottom_up_activations_context, _ = self.encoder_context(x_context)
        else:
            res_to_bottom_up_activations_context = None
        res_to_bottom_up_activations, _ = self.encoder_forecast(x_forecast)
        z_sample_cond_list = self.decoder.cond_latent_sample(
            res_to_bottom_up_activations=res_to_bottom_up_activations,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )

        return z_sample_cond_list

    def forward(self, x_context, x_item_id, temp=1.0):
        """
        This function is not the regular forward function, but the one just used for the gluonts predictor.
        """
        # Note: x_context is typically a batch with size of 1

        # required s.t. 2D tensors inside of VDVAE
        x_context = self.input_flatten(x_context)

        # preprocessing (as in main)
        x_context = insert_channel_dim(x_context)
        # convert x_item_id to strings # TODO why necessary
        x_item_id = [str(id) for id in x_item_id]
        x_context = self.normalize_fn(x=x_context, ids=x_item_id)

        x_sample = self.sample_p(
            n_samples=self.H.p_sample_n_samples,
            temp=temp,
            x_context=x_context,
            set_z_sample=None,
        )

        # repeat each item id H.p_sample_n_samples times --> "repeat_interleave"
        x_item_id_new = []
        for id in x_item_id:
            for _ in range(self.H.p_sample_n_samples):
                x_item_id_new.append(id)
        x_item_id = x_item_id_new
        # unnormalize
        x_sample = self.unnormalize_fn(x_sample, ids=x_item_id)

        # reshape to (batch_size, sampling_size, prediction_length)
        x_sample = torch.flatten(x_sample, start_dim=1)

        x_sample = x_sample.unsqueeze(0)  # new batch dimension
        x_sample = torch.split(
            x_sample, split_size_or_sections=self.H.p_sample_n_samples, dim=1
        )
        x_sample = torch.cat(x_sample, dim=0)

        # x_sample = x_sample.reshape((x_context.shape[0], self.H.p_sample_n_samples, x_sample.shape[1]))

        return x_sample  # (batch_size, n_samples, prediction_length)

    def get_predictor(self, input_transform):
        return PyTorchPredictor(
            prediction_length=self.H.forecast_length,
            input_names=["past_target", "item_id"],
            prediction_net=self,
            batch_size=self.H.batch_size,
            input_transform=input_transform,
            # forecast_generator=DistributionForecastGenerator(self.distr_output),  # TODO what to do here?
            device=self.H.device,
        )


class BlockFCWithBottleneck(nn.Module):
    """
    (Residual) block with convolutions and a channel bottleneck.
    """

    def __init__(
        self,
        in_dim,
        bottleneck_dim,
        out_dim,
        n_fc,
        residual,
        zero_init_weights_last=False,
    ):
        super().__init__()
        self.a1 = torch.nn.ELU()
        self.fc1 = nn.Linear(in_dim, bottleneck_dim)
        self.a2_list = torch.nn.ModuleList([torch.nn.ELU() for _ in range(n_fc)])
        self.fc2_list = torch.nn.ModuleList(
            [nn.Linear(bottleneck_dim, bottleneck_dim) for _ in range(n_fc)]
        )
        self.a3 = torch.nn.ELU()
        self.fc3 = nn.Linear(in_features=bottleneck_dim, out_features=out_dim)

        self.residual = residual

    def forward(self, x):
        h = self.fc1(self.a1(x))  # first activation, then conv
        for a2, c2 in zip(self.a2_list, self.fc2_list):
            h = c2(a2(h))
        h = self.fc3(self.a3(h))
        x = x + h if self.residual else h  # residual connection or not

        return x


class Encoder(nn.Module):
    def __init__(self, H, conditional=False):
        super(Encoder, self).__init__()
        self.H = H

        if conditional:
            res = (
                self.H.enc_context_state_dim_input
            )  # first self.H.context_length, then self.H.enc_context_state_dim after in_conv
            enc_spec_split = H.enc_context_spec.split(",")
        else:
            res = (
                self.H.enc_forecast_state_dim_input
            )  #  first self.H.forecast_length, then self.H.enc_forecast_state_dim_input after in_conv
            enc_spec_split = H.enc_spec.split(",")

        padded_context_length, padded_forecast_length = (
            self.H.pad_context + self.H.context_length,
            self.H.pad_forecast + self.H.forecast_length,
        )

        if conditional:
            self.in_fc = nn.Linear(H.n_meas * padded_context_length, res)
        else:
            self.in_fc = nn.Linear(H.n_meas * padded_forecast_length, res)
        self.blocks = torch.nn.ModuleList()

        self.layer_index_to_out_res = (
            {}
        )  # mapping from layer index to resolution of the output of that layer
        self.down_layer_indices = []
        self.conditional = conditional

        j = 0
        for s in enc_spec_split:
            if "d" in s:
                down_rate = int(s[1:])  # cut away 'd' and interpret as int
                res = int(res / down_rate)
                # difference to VDVAE implementation: 'd' spec is no longer introducing a whole block just for that, it's just the pooling operation now
                self.blocks.append(DownPooling(down_rate=down_rate))
                self.down_layer_indices.append(j)
                self.layer_index_to_out_res[j] = res
                j += 1
            elif "r" in s:
                s_split = s.split("r")
                n_blocks, n_reps_per_block = int(s_split[0]), int(s_split[1])
                for _ in range(n_blocks):
                    self.blocks.append(
                        BlockFCWithBottleneck(
                            in_dim=res,
                            bottleneck_dim=int(res * self.H.enc_bottleneck_dim_factor),
                            out_dim=res,
                            n_fc=self.H.enc_n_fc_bottleneck,
                            residual=True,
                        )
                    )
                    self.layer_index_to_out_res[j] = res
                    j += 1
                    for _2 in range(
                        n_reps_per_block - 1
                    ):  # already applied once, since layer itself added
                        self.blocks.append(self.blocks[-1])
                        self.layer_index_to_out_res[j] = res
                        j += 1
            else:  # just an integer ('x')
                for _ in range(int(s)):
                    self.blocks.append(
                        BlockFCWithBottleneck(
                            in_dim=res,
                            bottleneck_dim=int(res * self.H.enc_bottleneck_dim_factor),
                            out_dim=res,
                            n_fc=self.H.enc_n_fc_bottleneck,
                            residual=True,
                        )
                    )
                    self.layer_index_to_out_res[j] = res
                    j += 1

    def forward(self, x):
        x = self.in_fc(x)
        res_to_bottom_up_activations = {}

        # TODO 14/9 this res correct?
        res = self.layer_index_to_out_res[0]
        if self.conditional:
            res_to_bottom_up_activations[res] = x
        else:
            res_to_bottom_up_activations[res] = x
        state_norm_list = []
        for j, block in enumerate(self.blocks):
            x = block(x)
            res = self.layer_index_to_out_res[j]
            res_to_bottom_up_activations[res] = x
            # print("Encoder", self.conditional, j, res)

            state_norm = torch.linalg.norm(torch.flatten(x, start_dim=1), dim=1)
            state_norm_list.append(state_norm)

            # TODO magnitude constant
            # TODO plot to measure state size
            # TODO x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res]) --> required?

        return res_to_bottom_up_activations, state_norm_list


class DecoderBlock(hvae.DecoderBlock):
    def __init__(self, H, dec_state_dim, enc_forecast_state_dim, enc_context_state_dim):
        super().__init__()
        self.H = H

        bottleneck_dim = int(dec_state_dim * self.H.dec_bottleneck_dim_factor)
        dec_stochastic_depth = compute_stochastic_depth(self.H.dec_spec)

        # could also be done somewhere else
        if not H.conditional:
            enc_context_state_dim = 0

        self.z_dim = int(
            self.H.z_dim_factor * (dec_state_dim + enc_context_state_dim)
        )  # block-specific

        if self.H.conditional:
            self.p_z_block = BlockFCWithBottleneck(
                in_dim=dec_state_dim + enc_context_state_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=dec_state_dim + self.z_dim * 2,
                n_fc=self.H.dec_n_fc_bottleneck,
                residual=False,
                zero_init_weights_last=False,
            )
            self.q_z_block = BlockFCWithBottleneck(
                in_dim=enc_forecast_state_dim + enc_context_state_dim + dec_state_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=self.z_dim * 2,
                n_fc=self.H.dec_n_fc_bottleneck,
                residual=False,
                zero_init_weights_last=False,
            )
        else:
            self.p_z_block = BlockFCWithBottleneck(
                in_dim=dec_state_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=dec_state_dim + self.z_dim * 2,
                n_fc=self.H.dec_n_fc_bottleneck,
                residual=False,
                zero_init_weights_last=False,
            )
            self.q_z_block = BlockFCWithBottleneck(
                in_dim=enc_forecast_state_dim + dec_state_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=self.z_dim * 2,
                n_fc=self.H.dec_n_fc_bottleneck,
                residual=False,
                zero_init_weights_last=False,
            )
        self.mean_block = BlockFCWithBottleneck(
            in_dim=dec_state_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=dec_state_dim,
            n_fc=self.H.dec_n_fc_bottleneck,
            residual=True,
            zero_init_weights_last=False,
        )
        self.sample_proj_block = nn.Linear(
            in_features=self.z_dim, out_features=dec_state_dim
        )

        # special initializations
        self.mean_block.fc3.weight.data *= torch.sqrt(
            torch.tensor(1 / dec_stochastic_depth)
        )
        self.sample_proj_block.weight.data *= torch.sqrt(
            torch.tensor(1 / dec_stochastic_depth)
        )

    def forward(self, state, bottom_up_activation, bottom_up_activation_context=None):
        # q(z_l | z_>l, x)
        # Note: difference ot original VDVAE code: we here estimate log sigma square, not sigma square or sigma
        if self.H.conditional:
            # TODO output of q_z_block is 1d tensor, not 2d
            mu_q_z, log_sigma_q_z = self.q_z_block(
                torch.cat(
                    [state, bottom_up_activation, bottom_up_activation_context], dim=1
                )
            ).chunk(2, dim=1)
        else:
            mu_q_z, log_sigma_q_z = self.q_z_block(
                torch.cat([state, bottom_up_activation], dim=1)
            ).chunk(2, dim=1)
        q_z = D.Independent(
            D.Normal(loc=mu_q_z, scale=get_stable_scale(log_sigma_q_z)),
            reinterpreted_batch_ndims=1,
        )  # e.g. batch_shape=10, event_shape=[16,32,32]

        # p(z_l | z_>l)
        if self.H.conditional:
            p_z_activations = self.p_z_block(
                torch.cat([state, bottom_up_activation_context], dim=1)
            )
        else:
            p_z_activations = self.p_z_block(state)
        # Note: difference ot original VDVAE code: we here estimate log sigma square, not sigma square or sigma
        mu_p_z, log_sigma_p_z, z_iplus1_0_sigma = (
            p_z_activations[:, : self.z_dim],
            p_z_activations[:, self.z_dim : self.z_dim * 2],
            p_z_activations[:, self.z_dim * 2 :],
        )
        p_z = D.Independent(
            D.Normal(loc=mu_p_z, scale=get_stable_scale(log_sigma_p_z)),
            reinterpreted_batch_ndims=1,
        )  # e.g. batch_shape=10, event_shape=[16,32,32]

        # state
        state = state + z_iplus1_0_sigma
        z_sample_cond = q_z.rsample()
        state = state + self.sample_proj_block(z_sample_cond)
        state = self.mean_block(state)

        # Version 1:
        # kl = D.kl_divergence(q_z, p_z)  # for good introduction on shapes in torch distributions, see https://bochang.me/blog/posts/pytorch-distributions/#:~:text=Batch%20shape%20describes%20independent%2C%20not,modeled%20by%20its%20own%20distribution.
        # Version 2:
        kl = gaussian_analytical_kl(mu_q_z, mu_p_z, log_sigma_q_z, log_sigma_p_z)

        return state, z_sample_cond, kl

    def sample_p(
        self, state, temp, bottom_up_activations_context=None, set_z_sample=None
    ):
        # p(z_l | z_>l)
        if self.H.conditional:
            p_z_activations = self.p_z_block(
                torch.cat([state, bottom_up_activations_context], dim=1)
            )
        else:
            p_z_activations = self.p_z_block(state)
        # Note: difference ot original VDVAE code: we here estimate log sigma square, not sigma square or sigma
        mu_p_z, log_sigma_p_z, z_iplus1_0_sigma = (
            p_z_activations[:, : self.z_dim],
            p_z_activations[:, self.z_dim : self.z_dim * 2],
            p_z_activations[:, self.z_dim * 2 :],
        )
        scale = get_stable_scale(log_sigma_p_z)
        scale = scale * temp
        p_z = D.Independent(D.Normal(loc=mu_p_z, scale=scale), 1)

        # state
        state = state + z_iplus1_0_sigma
        if set_z_sample is not None and not self.H.conditional:
            z_sample_uncond = set_z_sample
        elif set_z_sample is None and not self.H.conditional:
            z_sample_uncond = p_z.rsample()  # properly taking a sample, not the mean
        elif set_z_sample is None and self.H.conditional:
            z_sample_uncond = p_z.rsample(
                torch.Size([])
            )  # note: only one sample is drawn, because "sample_shape is part of the batch_shape"
        state = state + self.sample_proj_block(z_sample_uncond)
        state = self.mean_block(state)

        return state, z_sample_uncond


class UpInterpolate(nn.Module):
    def __init__(self, up_rate):
        super().__init__()
        # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        self.up_rate = up_rate

    def forward(self, x):
        # squeeze and flatten required, since interpolate only implemented for 3D tensors
        x = torch.unsqueeze(x, 1)
        x = F.interpolate(x, scale_factor=self.up_rate)
        x = torch.flatten(x, start_dim=1)

        return x


# TODO
class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

        self.blocks = torch.nn.ModuleList()
        _, padded_forecast_length = (
            self.H.pad_context + self.H.context_length,
            self.H.pad_forecast + self.H.forecast_length,
        )
        if H.likelihood_type == "GaussianSigmaHyperparam":
            self.out_fc = nn.Linear(
                in_features=self.H.dec_state_dim_input,
                out_features=self.H.n_meas * padded_forecast_length,
            )
        elif H.likelihood_type == "GaussianSigmaEstimated":
            self.out_fc = nn.Linear(
                in_features=self.H.dec_state_dim_input,
                out_features=self.H.n_meas * padded_forecast_length * 2,
            )

        self.layer_index_to_out_res_dec = (
            {}
        )  # mapping from layer index to resolution of the output of that layer - decoder
        self.layer_index_to_out_res_enc_forecast = (
            {}
        )  # mapping from layer index to resolution of the output of that layer - encoder forecast
        self.layer_index_to_out_res_enc_context = (
            {}
        )  # mapping from  layer index to resolution of the output of that layer (in "context dimensions") - encoder context
        self.resolutions_dec = []  # unique set of all resolutions
        self.up_layer_indices = []  # layer indices which contain up-scaling operation

        # input_resolution is state dim here, since increased to that in first layer regardless of input size
        res_dec = int(
            compute_bottleneck_res(
                data_resolution=self.H.dec_state_dim_input, dec_spec=self.H.dec_spec
            )
        )
        res_enc_forecast = int(
            compute_bottleneck_res(
                data_resolution=self.H.enc_forecast_state_dim_input,
                enc_spec=self.H.enc_spec,
            )
        )
        res_enc_context = int(
            compute_bottleneck_res(
                data_resolution=self.H.enc_context_state_dim_input,
                enc_spec=self.H.enc_context_spec,
            )
        )
        dec_spec_split = H.dec_spec.split(",")
        # print("res_dec compute bottleneck", res_dec)

        j = 0
        for s in dec_spec_split:
            if "u" in s:
                up_rate = int(s[1:])  # cut away 'u' and interpret as int
                res_dec = int(res_dec * up_rate)
                res_enc_forecast = int(res_enc_forecast * up_rate)
                res_enc_context = int(res_enc_context * up_rate)
                # difference to VDVAE implementation: 'd' spec is no longer introducing a whole block just for that, it's just the pooling operation now
                self.blocks.append(
                    UpInterpolate(up_rate=up_rate)
                )  # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
                self.up_layer_indices.append(j)
                self.layer_index_to_out_res_dec[j] = res_dec
                self.layer_index_to_out_res_enc_forecast[j] = res_enc_forecast
                self.layer_index_to_out_res_enc_context[j] = res_enc_context
                j += 1
            elif "r" in s:
                s_split = s.split("r")
                n_blocks, n_reps_per_block = int(s_split[0]), int(s_split[1])
                for _ in range(n_blocks):
                    self.blocks.append(
                        DecoderBlock(
                            H,
                            dec_state_dim=res_dec,
                            enc_forecast_state_dim=res_enc_forecast,
                            enc_context_state_dim=res_enc_context,
                        )
                    )
                    self.layer_index_to_out_res_dec[j] = res_dec
                    self.layer_index_to_out_res_enc_forecast[j] = res_enc_forecast
                    self.layer_index_to_out_res_enc_context[j] = res_enc_context
                    j += 1
                    for _ in range(
                        n_reps_per_block - 1
                    ):  # already applied once, since layer itself added
                        self.blocks.append(self.blocks[-1])
                        self.layer_index_to_out_res_dec[j] = res_dec
                        self.layer_index_to_out_res_enc_forecast[j] = res_enc_forecast
                        self.layer_index_to_out_res_enc_context[j] = res_enc_context
                        j += 1
            else:  # just an integer ('x')
                for _ in range(int(s)):
                    self.blocks.append(
                        DecoderBlock(
                            H,
                            dec_state_dim=res_dec,
                            enc_forecast_state_dim=res_enc_forecast,
                            enc_context_state_dim=res_enc_context,
                        )
                    )
                    self.layer_index_to_out_res_dec[j] = res_dec
                    self.layer_index_to_out_res_enc_forecast[j] = res_enc_forecast
                    self.layer_index_to_out_res_enc_context[j] = res_enc_context
                    j += 1

        self.resolutions_dec = sorted(
            map(int, list(set(self.layer_index_to_out_res_dec.values())))
        )

        # cannot use nn.ParameterList with DataParallel (otherwise, self.bias_xs is empty), see https://github.com/pytorch/pytorch/issues/36035
        # workaround: use a module with parameters registered
        # this works when using DataParallel
        self.res_bias_object = Bias_xs()
        # print("self.resolutions_dec", self.resolutions_dec)
        for i, res_dec in enumerate(self.resolutions_dec):
            param_name = "p" + str(res_dec)
            setattr(
                self.res_bias_object, param_name, nn.Parameter(torch.zeros(1, res_dec))
            )  # string conversion required for passing dict to nn.ParameterDict
            self.res_bias_object.param_names.append(param_name)

        self.new_res_layer_indices = [0] + [int(i + 1) for i in self.up_layer_indices]

        self.last_scale = nn.Parameter(torch.ones(1, H.dec_state_dim_input))
        self.last_shift = nn.Parameter(torch.zeros(1, H.dec_state_dim_input))

    def forward(
        self, res_to_bottom_up_activations, res_to_bottom_up_activations_context=None
    ):
        _, padded_forecast_length = (
            self.H.pad_context + self.H.context_length,
            self.H.pad_forecast + self.H.forecast_length,
        )
        state = torch.zeros(
            (
                res_to_bottom_up_activations[self.H.enc_forecast_state_dim_input].shape[
                    0
                ],
                self.resolutions_dec[0],
            )
        ).to(
            "cuda" if "cuda" in self.H.device else "cpu"
        )  # 0th dimension: "batch size" is batch size of whatever input is fed --> assumed the input res always has an activatino from the encoder
        z_sample_cond_list, kl_list, state_norm_list = [], [], []
        for j, block in enumerate(self.blocks):
            res_dec = int(self.layer_index_to_out_res_dec[j])
            res_enc_forecast = int(self.layer_index_to_out_res_enc_forecast[j])
            res_enc_context = int(self.layer_index_to_out_res_enc_context[j])
            # print("Decoder: ", j, res_enc_forecast, res_enc_context)
            bottom_up_activation = res_to_bottom_up_activations[res_enc_forecast]
            if self.H.conditional:
                bottom_up_activation_context = res_to_bottom_up_activations_context[
                    res_enc_context
                ]
            else:
                bottom_up_activation_context = None

            if j in self.new_res_layer_indices:
                state = state + getattr(
                    self.res_bias_object, "p" + str(res_dec)
                )  # keys must be strings

            if j in self.up_layer_indices:
                state = block(state[:])
                # print("up", state.shape)
            else:
                state, z_sample_cond, kl = block(
                    state, bottom_up_activation, bottom_up_activation_context
                )
                # print(state.shape)
                z_sample_cond_list.append(z_sample_cond)
                kl_list.append(kl)

            state_norm = torch.linalg.norm(torch.flatten(state, start_dim=1), dim=1)
            state_norm_list.append(state_norm)

        state = state * self.last_scale + self.last_shift
        state = self.out_fc(state)

        return state, z_sample_cond_list, kl_list, state_norm_list

    def sample_p(
        self,
        n_samples,
        temp,
        res_to_bottom_up_activations_context=None,
        set_z_sample=None,
    ):
        if self.H.conditional:
            state = torch.zeros(
                (
                    res_to_bottom_up_activations_context[
                        self.layer_index_to_out_res_enc_context[0]
                    ].shape[0]
                    * n_samples,
                    self.resolutions_dec[0],
                )
            ).to("cuda" if "cuda" in self.H.device else "cpu")
        else:
            state = torch.zeros(
                (n_samples, self.H.dec_state_dim_input, self.resolutions_dec[0])
            ).to("cuda" if "cuda" in self.H.device else "cpu")
        z_sample_uncond_list = []
        if set_z_sample is None:
            set_z_sample = [None for _ in range(len(self.blocks))]

        k = 0  # index over non-up layers
        for j, block in enumerate(self.blocks):
            res_dec = self.layer_index_to_out_res_dec[j]
            res_enc_forecast = self.layer_index_to_out_res_enc_forecast[j]
            res_enc_context = self.layer_index_to_out_res_enc_context[j]

            if j in self.new_res_layer_indices:
                state = state + getattr(self.res_bias_object, "p" + str(res_dec))

            if j in self.up_layer_indices:
                state = block(state[:])
            else:
                # TODO slighlty inconsistent: j index is over blocks, while k index is over non-up blocks --> just do j index, but requires adjustemtnof set_z_sample passed
                if self.H.conditional:
                    bottom_up_activations_context = res_to_bottom_up_activations_context[
                        res_enc_context
                    ]
                    bottom_up_activations_context = torch.repeat_interleave(
                        bottom_up_activations_context, repeats=n_samples, dim=0
                    )
                    state, z_sample_uncond = block.sample_p(
                        state=state,
                        temp=temp,
                        set_z_sample=set_z_sample[k],
                        bottom_up_activations_context=bottom_up_activations_context,
                    )
                else:
                    state, z_sample_uncond = block.sample_p(
                        state=state,
                        temp=temp,
                        set_z_sample=set_z_sample[k],
                        bottom_up_activations_context=None,
                    )
                z_sample_uncond_list.append(z_sample_uncond)
                k += 1

        state = state * self.last_scale + self.last_shift
        state = self.out_fc(state)

        return state, z_sample_uncond_list

    def get_recon(
        self, res_to_bottom_up_activations, res_to_bottom_up_activations_context=None
    ):
        state, _, _, _ = self.forward(
            res_to_bottom_up_activations,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )

        return state

    def cond_latent_sample(
        self, res_to_bottom_up_activations, res_to_bottom_up_activations_context=None
    ):
        # TODO one could implement this with less operations than used in forward
        _, z_sample_cond_list, _, _ = self.forward(
            res_to_bottom_up_activations, res_to_bottom_up_activations_context
        )

        return z_sample_cond_list
