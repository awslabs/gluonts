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
import torch as T
import torch.nn as nn


class HVAE(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x_input, x_target):
        return NotImplementedError

    def sample_p(self, n_samples, temp, set_z_sample=None):
        return NotImplementedError

    def get_recon(self, x):
        return NotImplementedError

    def get_cond_latent_samples(self, x):
        return NotImplementedError


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return NotImplementedError

    def sample_p(
        self, state, temp, set_z_sample=None
    ):  # Note: could make the arguments more generic in this class
        return NotImplementedError


class HVAE_dummy(HVAE):
    """
    Model only works with MNIST.
    """

    def __init__(self, H):
        super().__init__()
        self.H = H
        self.encoder = nn.Conv1d(H.n_meas, 10, kernel_size=1, stride=1, bias=False)
        self.decoder = nn.Conv1d(10, H.n_meas, kernel_size=1, stride=1, bias=False)

    def forward(self, *args):
        x_t = self.encoder(args[0])
        p_x_z = self.decoder(x_t)
        loss = nn.MSELoss()
        elbo = loss(p_x_z, args[0])

        # return elbo, distortion, rate, kl_list
        return (
            elbo,
            T.Tensor([1.0]),
            T.Tensor([1.0]),
            [T.Tensor([1.0]) for i in range(9)],
            [T.Tensor([1.0]) for i in range(9)],
            [T.Tensor([1.0]) for i in range(9)],
        )

    def sample_p(self, n_samples, temp, set_z_sample=None):
        """
        set_z_samples enables to partly conditionally sample by setting some of the latent samples.
        """
        return T.zeros(
            (n_samples, self.H.n_meas, self.H.context_length + self.H.forecast_length)
        )

    def get_recon(self, x):
        return T.zeros(
            (x.shape[0], self.H.n_meas, self.H.context_length + self.H.forecast_length)
        )

    def get_cond_latent_samples(self, x):
        return [
            T.zeros(
                (
                    x.shape[0],
                    self.H.z_channels,
                    self.H.context_length + self.H.forecast_length,
                )
            )[i]
            for i in range(10)
        ]
