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
from gluonts.model.forecast_generator import DistributionForecastGenerator

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

from utils import gaussian_analytical_kl


class VDVAEConv(hvae.HVAE):
    def __init__(self, H):
        super(VDVAEConv, self).__init__()
        self.H = H
        self.encoder_forecast = Encoder(H)
        if H.conditional:
            self.encoder_context = Encoder(H, conditional=True)
        self.decoder = Decoder(H)
        if H.likelihood_type == "GaussianSigmaHyperparam":
            self.likelihood_model = GaussianLikelihoodSigmaHyperparam(H)
        elif H.likelihood_type == "GaussianSigmaEstimated":
            self.likelihood_model = GaussianLikelihoodSigmaEstimated(H)

    def forward_regular(self, x_forecast, x_context=None):
        (
            res_to_bottom_up_activations,
            state_norm_enc_forecast_list,
        ) = self.encoder_forecast(x_forecast)
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
            p_x_z_mean = state
            distortion, p_x_z = self.likelihood_model.compute_distortion(
                p_x_z_mean=p_x_z_mean, x=x_forecast
            )
        elif self.H.likelihood_type == "GaussianSigmaEstimated":
            p_x_z_mean = state[:, : self.H.n_meas, ...]
            p_x_z_log_std = state[:, self.H.n_meas :, ...]
            distortion, p_x_z = self.likelihood_model.compute_distortion(
                p_x_z_mean=p_x_z_mean,
                p_x_z_log_std=p_x_z_log_std,
                x=x_forecast,
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
                state_norm_enc_forecast_list,
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
                state_norm_enc_forecast_list,
                state_norm_dec_list,
                p_x_z,
            )

    def sample_p(self, n_samples, temp, x_context=None, set_z_sample=None):
        if self.H.conditional:
            res_to_bottom_up_activations_context, _ = self.encoder_context(
                x_context
            )
        else:
            res_to_bottom_up_activations_context = None
        state, _ = self.decoder.sample_p(
            n_samples=n_samples,
            temp=temp,
            set_z_sample=set_z_sample,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )
        if self.H.likelihood_type == "GaussianSigmaHyperparam":
            p_x_z_mean = state
            x_sample = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)
        elif self.H.likelihood_type == "GaussianSigmaEstimated":
            p_x_z_mean = state[:, : self.H.n_meas, ...]
            p_x_z_log_std = state[:, self.H.n_meas :, ...]
            x_sample = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)

        return x_sample

    def get_recon(self, x_forecast, x_context=None):
        res_to_bottom_up_activations, _ = self.encoder_forecast(x_forecast)
        if self.H.conditional:
            res_to_bottom_up_activations_context, _ = self.encoder_context(
                x_context
            )
        else:
            res_to_bottom_up_activations_context = None
        state = self.decoder.get_recon(
            res_to_bottom_up_activations=res_to_bottom_up_activations,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )

        if self.H.likelihood_type == "GaussianSigmaHyperparam":
            p_x_z_mean = state
            x_hat = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)
            p_x_z = self.likelihood_model.get_p_x_z(
                p_x_z_mean=p_x_z_mean.clone()
            )  # clone, since sometimes doing in-place operations of the x_hat --> p_x_z shall remain unaffected
        elif self.H.likelihood_type == "GaussianSigmaEstimated":
            p_x_z_mean = state[:, : self.H.n_meas, ...]
            p_x_z_log_std = state[:, self.H.n_meas :, ...]
            x_hat = self.likelihood_model.get_mean(p_x_z_mean=p_x_z_mean)
            p_x_z = self.likelihood_model.get_p_x_z(
                p_x_z_mean=p_x_z_mean.clone(),
                p_x_z_log_std=p_x_z_log_std.clone(),
            )  # clone, since sometimes doing in-place operations of the x_hat --> p_x_z shall remain unaffected

        return x_hat, p_x_z

    def get_cond_latent_samples(self, x_forecast, x_context=None):
        if self.H.conditional:
            res_to_bottom_up_activations_context, _ = self.encoder_context(
                x_context
            )
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


def get_conv_layer(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups=1,
    zero_init_bias=True,
    zero_init_weights=False,
):
    """
    Source code as modified from https://github.com/openai/vdvae.
    """
    layer = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    if zero_init_bias:
        layer.bias.data *= 0.0
    if zero_init_weights:
        layer.weights.data *= 0.0
    return layer


def get_3x3_conv_layer(
    in_channels, out_channels, zero_init_bias=True, zero_init_weights=False
):
    """
    Source code as modified from https://github.com/openai/vdvae.
    """
    return get_conv_layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        zero_init_bias=zero_init_bias,
        zero_init_weights=zero_init_weights,
    )


def get_1x1_conv_layer(
    in_channels, out_channels, zero_init_bias=True, zero_init_weights=False
):
    """
    Source code as modified from https://github.com/openai/vdvae.
    """
    return get_conv_layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        zero_init_bias=zero_init_bias,
        zero_init_weights=zero_init_weights,
    )


class DownPooling(nn.Module):
    def __init__(self, down_rate):
        super().__init__()
        self.down_rate = down_rate

    def forward(self, x):
        x = F.avg_pool1d(x, kernel_size=self.down_rate, stride=self.down_rate)

        return x


class BlockConvWithBottleneck(nn.Module):
    """
    (Residual) block with convolutions and a channel bottleneck.
    """

    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        n_conv_3x3,
        residual,
        zero_init_weights_last=False,
    ):
        super().__init__()
        self.a1 = torch.nn.ELU()
        self.c1 = get_1x1_conv_layer(in_channels, bottleneck_channels)
        self.a2_list = torch.nn.ModuleList(
            [torch.nn.ELU() for _ in range(n_conv_3x3)]
        )
        self.c2_list = torch.nn.ModuleList(
            [
                get_3x3_conv_layer(bottleneck_channels, bottleneck_channels)
                for _ in range(n_conv_3x3)
            ]
        )
        self.a3 = torch.nn.ELU()
        self.c3 = get_1x1_conv_layer(
            bottleneck_channels,
            out_channels,
            zero_init_weights=zero_init_weights_last,
        )

        self.residual = residual

    def forward(self, x):
        h = self.c1(self.a1(x))  # first activation, then conv
        for a2, c2 in zip(self.a2_list, self.c2_list):
            h = c2(a2(h))
        h = self.c3(self.a3(h))
        x = x + h if self.residual else h  # residual connection or not

        return x


class Encoder(nn.Module):
    def __init__(self, H, conditional=False):
        super(Encoder, self).__init__()
        self.H = H

        if conditional:
            res = self.H.context_length + self.H.pad_context
            enc_spec_split = H.enc_context_spec.split(",")
            enc_state_channels = self.H.vdvae_enc_context_state_channels
        else:
            res = self.H.forecast_length + self.H.pad_forecast
            enc_spec_split = H.enc_spec.split(",")
            enc_state_channels = self.H.vdvae_enc_state_channels

        self.in_conv = get_3x3_conv_layer(H.n_meas, enc_state_channels)
        self.blocks = torch.nn.ModuleList()

        self.layer_index_to_out_res = (
            {}
        )  # mapping from layer index to resolution of the output of that layer
        self.down_layer_indices = []
        self.conditional = conditional

        j = 0
        for s in enc_spec_split:
            # print("Encoder", self.conditional, res)
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
                        BlockConvWithBottleneck(
                            in_channels=enc_state_channels,
                            bottleneck_channels=int(
                                enc_state_channels
                                * self.H.vdvae_enc_bottleneck_channels_factor
                            ),
                            out_channels=enc_state_channels,
                            n_conv_3x3=H.vdvae_enc_n_conv_3x3,
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
                        BlockConvWithBottleneck(
                            in_channels=enc_state_channels,
                            bottleneck_channels=int(
                                enc_state_channels
                                * self.H.vdvae_enc_bottleneck_channels_factor
                            ),
                            out_channels=enc_state_channels,
                            n_conv_3x3=H.vdvae_enc_n_conv_3x3,
                            residual=True,
                        )
                    )
                    self.layer_index_to_out_res[j] = res
                    j += 1

    def forward(self, x):
        x = self.in_conv(x)
        res_to_bottom_up_activations = {}
        if self.conditional:
            res_to_bottom_up_activations[self.H.context_length] = x
        else:
            res_to_bottom_up_activations[self.H.forecast_length] = x
        state_norm_list = []
        for j, block in enumerate(self.blocks):
            x = block(x)
            res = self.layer_index_to_out_res[j]
            res_to_bottom_up_activations[res] = x

            state_norm = torch.linalg.norm(
                torch.flatten(x, start_dim=1), dim=1
            )
            state_norm_list.append(state_norm)

            # TODO magnitude constant
            # TODO plot to measure state size
            # TODO x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res]) --> required?

        return res_to_bottom_up_activations, state_norm_list


class DecoderBlock(hvae.DecoderBlock):
    def __init__(self, H):
        super().__init__()
        self.H = H

        bottleneck_channels = int(
            self.H.vdvae_dec_state_channels
            * self.H.vdvae_dec_bottleneck_channels_factor
        )
        dec_stochastic_depth = compute_stochastic_depth(self.H.dec_spec)

        if H.conditional:
            enc_context_state_channels = (
                self.H.vdvae_enc_context_state_channels
            )
        else:
            enc_context_state_channels = 0

        if self.H.conditional:
            self.p_z_block = BlockConvWithBottleneck(
                in_channels=self.H.vdvae_dec_state_channels
                + enc_context_state_channels,
                bottleneck_channels=bottleneck_channels,
                out_channels=self.H.vdvae_dec_state_channels
                + self.H.z_channels * 2,
                n_conv_3x3=self.H.vdvae_dec_n_conv_3x3,
                residual=False,
                zero_init_weights_last=False,
            )
            self.q_z_block = BlockConvWithBottleneck(
                in_channels=self.H.vdvae_enc_state_channels
                + enc_context_state_channels
                + self.H.vdvae_dec_state_channels,
                bottleneck_channels=bottleneck_channels,
                out_channels=self.H.z_channels * 2,
                n_conv_3x3=self.H.vdvae_dec_n_conv_3x3,
                residual=False,
                zero_init_weights_last=False,
            )
        else:
            self.p_z_block = BlockConvWithBottleneck(
                in_channels=self.H.vdvae_dec_state_channels,
                bottleneck_channels=bottleneck_channels,
                out_channels=self.H.vdvae_dec_state_channels
                + self.H.z_channels * 2,
                n_conv_3x3=self.H.vdvae_dec_n_conv_3x3,
                residual=False,
                zero_init_weights_last=False,
            )
            self.q_z_block = BlockConvWithBottleneck(
                in_channels=self.H.vdvae_enc_state_channels
                + self.H.vdvae_dec_state_channels,
                bottleneck_channels=bottleneck_channels,
                out_channels=self.H.z_channels * 2,
                n_conv_3x3=self.H.vdvae_dec_n_conv_3x3,
                residual=False,
                zero_init_weights_last=False,
            )
        self.mean_block = BlockConvWithBottleneck(
            in_channels=self.H.vdvae_dec_state_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=self.H.vdvae_dec_state_channels,
            n_conv_3x3=self.H.vdvae_dec_n_conv_3x3,
            residual=True,
            zero_init_weights_last=False,
        )
        self.sample_proj_block = get_1x1_conv_layer(
            in_channels=self.H.z_channels,
            out_channels=self.H.vdvae_dec_state_channels,
        )

        # special initializations
        self.mean_block.c3.weight.data *= torch.sqrt(
            torch.tensor(1 / dec_stochastic_depth)
        )
        self.sample_proj_block.weight.data *= torch.sqrt(
            torch.tensor(1 / dec_stochastic_depth)
        )

    def forward(
        self, state, bottom_up_activation, bottom_up_activation_context=None
    ):
        # q(z_l | z_>l, x)
        # Note: difference ot original VDVAE code: we here estimate log sigma square, not sigma square or sigma
        if self.H.conditional:
            mu_q_z, log_sigma_q_z = self.q_z_block(
                torch.cat(
                    [
                        state,
                        bottom_up_activation,
                        bottom_up_activation_context,
                    ],
                    dim=1,
                )
            ).chunk(2, dim=1)
        else:
            mu_q_z, log_sigma_q_z = self.q_z_block(
                torch.cat([state, bottom_up_activation], dim=1)
            ).chunk(2, dim=1)
        q_z = D.Independent(
            D.Normal(loc=mu_q_z, scale=get_stable_scale(log_sigma_q_z)),
            reinterpreted_batch_ndims=2,
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
            p_z_activations[:, : self.H.z_channels, ...],
            p_z_activations[:, self.H.z_channels : self.H.z_channels * 2, ...],
            p_z_activations[:, self.H.z_channels * 2 :, ...],
        )
        p_z = D.Independent(
            D.Normal(loc=mu_p_z, scale=get_stable_scale(log_sigma_p_z)),
            reinterpreted_batch_ndims=2,
        )  # e.g. batch_shape=10, event_shape=[16,32,32]

        # state
        state = state + z_iplus1_0_sigma
        z_sample_cond = q_z.rsample()
        state = state + self.sample_proj_block(z_sample_cond)
        state = self.mean_block(state)

        # Version 1
        # kl = D.kl_divergence(q_z, p_z)  # for good introduction on shapes in torch distributions, see https://bochang.me/blog/posts/pytorch-distributions/#:~:text=Batch%20shape%20describes%20independent%2C%20not,modeled%20by%20its%20own%20distribution.
        # Version 2:
        kl = gaussian_analytical_kl(
            mu_q_z, mu_p_z, log_sigma_q_z, log_sigma_p_z
        )

        return state, z_sample_cond, kl

    def sample_p(
        self,
        state,
        temp,
        bottom_up_activations_context=None,
        set_z_sample=None,
    ):
        # p(z_l | z_>l)
        if self.H.conditional:
            p_z_activations = self.p_z_block(
                torch.cat([state, bottom_up_activations_context], dim=1)
            )
            # in conditional mode, require:
            # sample_shape=n_samples, event_shape=whatever it was before
            # batch_shape=batch_size (as before)
        else:
            p_z_activations = self.p_z_block(state)
        # Note: difference ot original VDVAE code: we here estimate log sigma square, not sigma square or sigma
        mu_p_z, log_sigma_p_z, z_iplus1_0_sigma = (
            p_z_activations[:, : self.H.z_channels, ...],
            p_z_activations[:, self.H.z_channels : self.H.z_channels * 2, ...],
            p_z_activations[:, self.H.z_channels * 2 :, ...],
        )
        scale = get_stable_scale(log_sigma_p_z)
        scale = scale * temp
        p_z = D.Independent(D.Normal(loc=mu_p_z, scale=scale), 2)

        # state
        state = state + z_iplus1_0_sigma
        if set_z_sample is not None and not self.H.conditional:
            z_sample_uncond = set_z_sample
        elif set_z_sample is None and not self.H.conditional:
            z_sample_uncond = (
                p_z.rsample()
            )  # properly taking a sample, not the mean
        elif set_z_sample is None and self.H.conditional:
            z_sample_uncond = p_z.rsample(
                torch.Size([])
            )  # note: only one sample is drawn, because "sample_shape is part of the batch_shape"
            # z_sample_uncond = torch.permute(z_sample_uncond, (1, 0, 2, 3))  # reorder dimensions s.t. (batch_size, sample_shape, event_shape_1, event_shape_2)
            # z_sample_uncond = torch.flatten(z_sample_uncond, start_dim=0, end_dim=1)   # dimensions: (batch_size * sample_shape, event_shape_1, event_shape_2), s.t. in 0th dimension, blocks of sample_shape each correponding to one datapoint in x_context (in Decoder)
        state = state + self.sample_proj_block(z_sample_uncond)
        state = self.mean_block(state)

        return state, z_sample_uncond


class UpInterpolate(nn.Module):
    def __init__(self, up_rate):
        super().__init__()
        # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        self.up_rate = up_rate

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.up_rate)

        return x


class Bias_xs(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_names = []


class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

        self.blocks = torch.nn.ModuleList()
        if H.likelihood_type == "GaussianSigmaHyperparam":
            self.out_conv = get_conv_layer(
                in_channels=self.H.vdvae_dec_state_channels,
                out_channels=self.H.n_meas,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        elif H.likelihood_type == "GaussianSigmaEstimated":
            self.out_conv = get_conv_layer(
                in_channels=self.H.vdvae_dec_state_channels,
                out_channels=self.H.n_meas * 2,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        # mapping from layer index to resolution of the output of that layer
        self.layer_index_to_out_res = {}
        # mapping from  layer index to resolution of the output of that layer (in "context dimensions")
        self.layer_index_to_out_res_context = {}
        # unique set of all resolutions
        self.resolutions = []
        # layer indices which contain up-scaling operation
        self.up_layer_indices = []

        res = int(
            compute_bottleneck_res(
                data_resolution=self.H.forecast_length + self.H.pad_forecast,
                enc_spec=self.H.enc_spec,
            )
        )
        res_context = int(
            compute_bottleneck_res(
                data_resolution=self.H.context_length + self.H.pad_context,
                enc_spec=self.H.enc_context_spec,
            )
        )
        dec_spec_split = H.dec_spec.split(",")

        j = 0
        for s in dec_spec_split:
            if "u" in s:
                up_rate = int(s[1:])  # cut away 'u' and interpret as int
                res = int(res * up_rate)
                res_context = int(res_context * up_rate)
                # difference to VDVAE implementation: 'd' spec is no longer introducing a whole block just for that, it's just the pooling operation now
                self.blocks.append(
                    UpInterpolate(up_rate=up_rate)
                )  # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
                self.up_layer_indices.append(j)
                self.layer_index_to_out_res[j] = res
                self.layer_index_to_out_res_context[j] = res_context
                j += 1
            elif "r" in s:
                s_split = s.split("r")
                n_blocks, n_reps_per_block = int(s_split[0]), int(s_split[1])
                for _ in range(n_blocks):
                    self.blocks.append(DecoderBlock(H))
                    self.layer_index_to_out_res[j] = res
                    self.layer_index_to_out_res_context[j] = res_context
                    j += 1
                    for _ in range(
                        n_reps_per_block - 1
                    ):  # already applied once, since layer itself added
                        self.blocks.append(self.blocks[-1])
                        self.layer_index_to_out_res[j] = res
                        self.layer_index_to_out_res_context[j] = res_context
                        j += 1
            else:  # just an integer ('x')
                for _ in range(int(s)):
                    self.blocks.append(DecoderBlock(H))
                    self.layer_index_to_out_res[j] = res
                    self.layer_index_to_out_res_context[j] = res_context
                    j += 1

        self.resolutions = sorted(
            map(int, list(set(self.layer_index_to_out_res.values())))
        )

        # cannot use nn.ParameterList with DataParallel (otherwise, self.bias_xs is empty), see https://github.com/pytorch/pytorch/issues/36035
        # workaround: use a module with parameters registered
        # this works when using DataParallel
        self.res_bias_object = Bias_xs()
        for i, res in enumerate(self.resolutions):
            param_name = "p" + str(res)
            setattr(
                self.res_bias_object,
                param_name,
                nn.Parameter(
                    torch.zeros(1, self.H.vdvae_dec_state_channels, res)
                ),
            )  # string conversion required for passing dict to nn.ParameterDict
            self.res_bias_object.param_names.append(param_name)

        self.new_res_layer_indices = [0] + [
            int(i + 1) for i in self.up_layer_indices
        ]

        self.last_scale = nn.Parameter(
            torch.ones(1, H.vdvae_dec_state_channels, 1)
        )
        self.last_shift = nn.Parameter(
            torch.zeros(1, H.vdvae_dec_state_channels, 1)
        )

    def forward(
        self,
        res_to_bottom_up_activations,
        res_to_bottom_up_activations_context=None,
    ):
        state = torch.zeros(
            (
                res_to_bottom_up_activations[self.H.forecast_length].shape[0],
                self.H.vdvae_dec_state_channels,
                self.resolutions[0],
            )
        ).to(
            "cuda" if "cuda" in self.H.device else "cpu"
        )  # 0th dimension: "batch size" is batch size of whatever input is fed --> assumed the input res always has an activatino from the encoder
        z_sample_cond_list, kl_list, state_norm_list = [], [], []
        for j, block in enumerate(self.blocks):
            res = int(self.layer_index_to_out_res[j])
            res_context = int(self.layer_index_to_out_res_context[j])
            # print("Decoder", j, res, res_context)
            bottom_up_activation = res_to_bottom_up_activations[res]
            if self.H.conditional:
                bottom_up_activation_context = (
                    res_to_bottom_up_activations_context[res_context]
                )
            else:
                bottom_up_activation_context = None

            if j in self.new_res_layer_indices:
                state = state + getattr(
                    self.res_bias_object, "p" + str(res)
                )  # keys must be strings  # TODO double-check!

            if j in self.up_layer_indices:
                # TODO why this slicing? Correct?
                # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
                state = block(
                    state[:, : state.shape[1], ...]
                )  # TODO check!!!!
            else:
                state, z_sample_cond, kl = block(
                    state, bottom_up_activation, bottom_up_activation_context
                )
                z_sample_cond_list.append(z_sample_cond)
                kl_list.append(kl)

            state_norm = torch.linalg.norm(
                torch.flatten(state, start_dim=1), dim=1
            )
            state_norm_list.append(state_norm)

        state = state * self.last_scale + self.last_shift
        state = self.out_conv(state)

        return state, z_sample_cond_list, kl_list, state_norm_list

    def sample_p(
        self,
        n_samples,
        temp,
        res_to_bottom_up_activations_context=None,
        set_z_sample=None,
    ):
        if self.H.conditional:
            # in conditional mode: n_samples samples per data point in the x_context batch are produced in DecoderBlock
            # hence, here shape (batch_size * n_samples, n_channels, res)
            state = torch.zeros(
                (
                    res_to_bottom_up_activations_context[
                        self.layer_index_to_out_res_context[0]
                    ].shape[0]
                    * n_samples,
                    self.H.vdvae_dec_state_channels,
                    self.resolutions[0],
                )
            ).to("cuda" if "cuda" in self.H.device else "cpu")
        else:
            state = torch.zeros(
                (
                    res_to_bottom_up_activations_context[
                        self.layer_index_to_out_res_context[0]
                    ].shape[0],
                    self.H.vdvae_dec_state_channels,
                    self.resolutions[0],
                )
            ).to("cuda" if "cuda" in self.H.device else "cpu")
        z_sample_uncond_list = []
        if set_z_sample is None:
            set_z_sample = [None for _ in range(len(self.blocks))]

        k = 0  # index over non-up layers
        for j, block in enumerate(self.blocks):
            res = self.layer_index_to_out_res[j]
            res_context = self.layer_index_to_out_res_context[j]

            if j in self.new_res_layer_indices:
                state = state + getattr(self.res_bias_object, "p" + str(res))

            if j in self.up_layer_indices:
                # TODO why this slicing?
                # TODO correct?
                # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
                state = block(state[:, : state.shape[1], ...])
            else:
                # TODO slighlty inconsistent: j index is over blocks, while k index is over non-up blocks --> just do j index, but requires adjustemtnof set_z_sample passed
                if self.H.conditional:
                    bottom_up_activations_context = (
                        res_to_bottom_up_activations_context[res_context]
                    )
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
        state = self.out_conv(state)

        return state, z_sample_uncond_list

    def get_recon(
        self,
        res_to_bottom_up_activations,
        res_to_bottom_up_activations_context=None,
    ):
        state, _, _, _ = self.forward(
            res_to_bottom_up_activations,
            res_to_bottom_up_activations_context=res_to_bottom_up_activations_context,
        )

        return state

    def cond_latent_sample(
        self,
        res_to_bottom_up_activations,
        res_to_bottom_up_activations_context=None,
    ):
        # TODO one could implement this with less operations than used in forward
        _, z_sample_cond_list, _, _ = self.forward(
            res_to_bottom_up_activations, res_to_bottom_up_activations_context
        )

        return z_sample_cond_list
