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
import torch.distributions as D
from utils import get_stable_scale


class GaussianLikelihoodSigmaHyperparam:
    def __init__(self, H):
        self.H = H

    def compute_distortion(self, p_x_z_mean, x):
        # if padding is used: exclude from loss computation
        if self.H.pad_forecast > 0:
            # padding was added on the left, in the second channel
            x = x[:, :, self.H.pad_forecast :]
            p_x_z_mean = p_x_z_mean[:, :, self.H.pad_forecast :]

        p_x_z = D.Independent(
            D.Normal(
                loc=p_x_z_mean, scale=torch.ones_like(p_x_z_mean) * self.H.sigma_p_x_z
            ),
            reinterpreted_batch_ndims=2,
        )  # TODO check reinterpreted_batch_ndims argument
        distortion = p_x_z.log_prob(x)

        return distortion, p_x_z

    def get_mean(self, p_x_z_mean):
        # if padding is used: exclude from loss computation
        if self.H.pad_forecast > 0:
            # padding was added on the left, in the second channel
            p_x_z_mean = p_x_z_mean[:, :, self.H.pad_forecast :]

        return p_x_z_mean

    def get_p_x_z(self, p_x_z_mean):
        # if padding is used: exclude from loss computation
        if self.H.pad_forecast > 0:
            # padding was added on the left, in the second channel
            p_x_z_mean = p_x_z_mean[:, :, self.H.pad_forecast :]

        p_x_z = D.Independent(
            D.Normal(
                loc=p_x_z_mean, scale=torch.ones_like(p_x_z_mean) * self.H.sigma_p_x_z
            ),
            reinterpreted_batch_ndims=2,
        )  # TODO check reinterpreted_batch_ndims argument

        return p_x_z


class GaussianLikelihoodSigmaEstimated:
    def __init__(self, H):
        self.H = H

    def compute_distortion(self, p_x_z_mean, p_x_z_log_std, x):
        # if padding is used: exclude from loss computation
        if self.H.pad_forecast > 0:
            # padding was added on the left, in the second channel
            x = x[:, :, self.H.pad_forecast :]  # TODO assumes just one measurement
            p_x_z_mean = p_x_z_mean[:, :, self.H.pad_forecast :]
            p_x_z_log_std = p_x_z_log_std[:, :, self.H.pad_forecast :]

        p_x_z = D.Independent(
            D.Normal(loc=p_x_z_mean, scale=get_stable_scale(p_x_z_log_std)),
            reinterpreted_batch_ndims=2,
        )
        distortion = p_x_z.log_prob(x)

        return distortion, p_x_z

    def get_mean(self, p_x_z_mean):
        # if padding is used: exclude from loss computation
        if self.H.pad_forecast > 0:
            # padding was added on the left, in the second channel
            p_x_z_mean = p_x_z_mean[:, :, self.H.pad_forecast :]

        return p_x_z_mean

    def get_p_x_z(self, p_x_z_mean, p_x_z_log_std):
        # if padding is used: exclude from loss computation
        if self.H.pad_forecast > 0:
            # padding was added on the left, in the second channel
            p_x_z_mean = p_x_z_mean[:, :, self.H.pad_forecast :]
            p_x_z_log_std = p_x_z_log_std[:, :, self.H.pad_forecast :]

        p_x_z = D.Independent(
            D.Normal(loc=p_x_z_mean, scale=get_stable_scale(p_x_z_log_std)),
            reinterpreted_batch_ndims=2,
        )

        return p_x_z
