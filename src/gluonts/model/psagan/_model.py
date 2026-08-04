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
from math import log2, sqrt
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from gluonts.core.component import validated

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

# from icecream import ic
# from datetime import datetime


# def time_format():
#     return f"{datetime.now()}|> "


# ic.configureOutput(prefix=time_format, includeContext=True)


class SelfAttention(nn.Module):
    """Implements a self attention layer with spectral normalisation.

    Attributes:
        nb_in_features:
            number of features in the input sequence.
        key_features:
            size of the queries and the keys vectors
        value_features:
            size of the values vectors
        ks_value:
            size of the kernel of the convolutional layer that produces the
            values vectors
        ks_query:
            size of the kernel of the convolutional layer that produces the
            queries vectors
        ks_key:
            size of the kernel of the convolutional layer that produces the
            keys vectors
    """

    @validated()
    def __init__(
        self,
        nb_in_features: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
    ):
        super(SelfAttention, self).__init__()
        assert ks_value % 2 == 1, "ks_value should be an odd number"
        assert ks_query % 2 == 1, "ks_query should be an odd number"
        assert ks_key % 2 == 1, "ks_key should be an odd number"

        d_query, p_query = self._set_params(ks_query)

        self.conv_Q = nn.Conv1d(
            in_channels=nb_in_features,
            out_channels=key_features,
            kernel_size=ks_query,
            dilation=d_query,
            padding=p_query,
        )

        d_key, p_key = self._set_params(ks_key)
        self.conv_K = nn.Conv1d(
            in_channels=nb_in_features,
            out_channels=key_features,
            kernel_size=ks_key,
            dilation=d_key,
            padding=p_key,
        )

        d_value, p_value = self._set_params(ks_value)
        self.conv_V = nn.Conv1d(
            in_channels=nb_in_features,
            out_channels=value_features,
            kernel_size=ks_value,
            dilation=d_value,
            padding=p_value,
        )

        self.key_features = key_features

    def _set_params(self, kernel_size):
        """Computes dilation and padding parameter given the kernel size

        The dilation and padding parameter are computed such as
        an input sequence to a 1d convolution does not change in length.

        Returns:
            Two integer for dilation and padding.
        """

        if kernel_size % 2 == 1:  # If kernel size is an odd number
            dilation = 1
            padding = int((kernel_size - 1) / 2)
        else:  # If kernel size is an even number
            dilation = 2
            padding = int(kernel_size - 1)

        return dilation, padding

    def forward(self, x):
        """Computes self-attention

        Arguments:
            x: torch.tensor object of shape (batch_size, nb_in_features, L_in)

        Returns:
            torch.tensor object of shape (batch_size, value_features, L_in)
        """

        Q = self.conv_Q(x)  # Shape (batch_size, key_features, L_in)
        K = self.conv_K(x)  # Shape (batch_size, key_features, L_in)
        V = self.conv_V(x)  # Shape (batch_size, value_features, L_in)
        A = (
            torch.matmul(Q.permute(0, 2, 1), K) / sqrt(self.key_features)
        ).softmax(
            2
        )  # Shape (batch_size, L_in, L_in)
        H = torch.matmul(A, V.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # Shape (batch_size, value_features, L_in)

        return H


class ResidualSelfAttention(nn.Module):
    """Implement self attention module as described in [1].

    It consists of a self attention layer with spectral normalisation,
    followed by 1x1 convolution with spectral normalisation and a
    parametrized skip connection.

    Attributes:
        nb_in_features:
            number of features in the input sequence. It is also the number
            of output features.
        key_features:
            size of the queries and the keys vectors
        value_features:
            size of the values vectors
        ks_value:
            size of the kernel of the convolutional layer that produces the
            values vectors
        ks_query:
            size of the kernel of the convolutional layer that produces the
            queries vectors
        ks_key:
            size of the kernel of the convolutional layer that produces the
            keys vectors
        skip_param:
            float value, initial value of the parametrised skip connection.

    [1]: https://arxiv.org/pdf/1805.08318.pdf
    """

    @validated()
    def __init__(
        self,
        nb_in_features: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        skip_param: float = 0.0,
    ):
        super(ResidualSelfAttention, self).__init__()
        self.self_attention = SelfAttention(
            nb_in_features=nb_in_features,
            key_features=key_features,
            value_features=value_features,
            ks_value=ks_value,
            ks_query=ks_query,
            ks_key=ks_key,
        )
        self.conv = nn.Conv1d(
            in_channels=value_features,
            out_channels=nb_in_features,
            kernel_size=1,
        )

        self.gamma = nn.Parameter(torch.tensor(skip_param))

    def forward(self, x):
        """Comptues the forward

        Arguments:
            torch.tensor of shape (batch size, nb_in_features, length)
        Returns
            torch.tensor of shape (batch size, nb_in_features, length)
        """
        y = self.self_attention(x)
        y = self.conv(y)
        return self.gamma * y + x


class ConvResdiualSelfAttention(nn.Module):
    """Implement Convolution with Residual self attention

    It consists of a convolution layer with spectral normalisation
    and LeakyReLU then a residual self attention described above.

    Attributes:
        nb_in_features:
            number of features in the input sequence.
        nb_out_features:
            number of features in the output sequence.
        ks_conv:
            kernel size of the conv layer
        key_features:
            size of the queries and the keys vectors
        value_features:
            size of the values vectors
        ks_value:
            size of the kernel of the convolutional layer that produces the
            values vectors
        ks_query:
            size of the kernel of the convolutional layer that produces the
            queries vectors
        ks_key:
            size of the kernel of the convolutional layer that produces the
            keys vectors
        skip_param:
            float value, initial value of the parametrised skip connection.

    """

    @validated()
    def __init__(
        self,
        nb_in_features: int,
        nb_out_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        skip_param: float = 0.0,
        self_attention: bool = True,
    ):
        super(ConvResdiualSelfAttention, self).__init__()
        dilation, padding = self._set_params(ks_conv)
        self.spectral_conv = spectral_norm(
            nn.Conv1d(
                in_channels=nb_in_features,
                out_channels=nb_out_features,
                kernel_size=ks_conv,
                dilation=dilation,
                padding=padding,
            )
        )
        self.leakyrelu = nn.LeakyReLU()
        if self_attention:
            self.res_selfattention = ResidualSelfAttention(
                nb_in_features=nb_out_features,
                key_features=key_features,
                value_features=value_features,
                ks_value=ks_value,
                ks_query=ks_query,
                ks_key=ks_key,
                skip_param=skip_param,
            )
        self.self_attention = self_attention

    def _set_params(self, kernel_size):
        """Computes dilation and padding parameter given the kernel size

        The dilation and padding parameter are computed such as
        an input sequence to a 1d convolution does not change in length.

        Returns:
            Two integer for dilation and padding.
        """

        if kernel_size % 2 == 1:  # If kernel size is an odd number
            dilation = 1
            padding = int((kernel_size - 1) / 2)
        else:  # If kernel size is an even number
            dilation = 2
            padding = int(kernel_size - 1)

        return dilation, padding

    def forward(self, x):
        """Computes the forward

        Arguments:
            torch.tensor of shape (batch size, nb_in_features, length)
        Returns:
            torch.tensor of shape (batch size, nb_out_features, length)
        """
        x = self.spectral_conv(x)
        x = self.leakyrelu(x)
        if self.self_attention:
            x = self.res_selfattention(x)

        return x


class customActivation(nn.Module):
    def __init__(self):
        super(customActivation, self).__init__()

    def forward(self, x):
        y = torch.div(x, 2 * (1 + torch.abs(x))) + 0.5
        return y


class ProGenerator(nn.Module):
    """Implementation of the progressive generator.

    The generator will take as input a univariate noise vector and time features
    of length 8. It will then gradually generate univariate time series of length
    target_len by doubling the size of the input vector at each time.

    Attributes:
        target_len:
            Integer that specifies the output length of the generated time series
        nb_features:
            Number of features passed with as input with the noise vector
        ks_conb:
            kernel size of the conv layer before the self attention module
        key_features:
            size of the key vectors in the self attention module
        value_features:
            size of the value and query vectors in the self attention module
        ks_value:
            kernel size of the conv layer computing the value vectors
        ks_query:
            kernel size of the conv layer computing the query vectors
        ks_key:
            kernel size of the conv layer computing the key vectors
    """

    @validated()
    def __init__(
        self,
        target_len: int,
        nb_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        device: str,
        residual_factor: float = 0.0,
        scaling: str = "local",
        cardinality: Optional[List[int]] = None,
        embedding_dim: int = 10,
        self_attention: bool = True,
        channel_nb: int = 32,
        context_length: int = 0,
    ):
        super(ProGenerator, self).__init__()
        assert device == "cpu" or device == "cuda"
        self.device = torch.device(device)
        assert (
            scaling == "local" or scaling == "global" or scaling == "NoScale"
        ), "scaling has to be \
            local or global. If it is global, then the whole time series will be\
            mix-max scaled. If it is local, then each subseries of length \
            target_len will be min-max scaled independenty. If is is NoScale\
            then no scaling is applied to the dataset."
        self.scaling = scaling
        assert (
            log2(target_len) % 1 == 0
        ), "target len must be an integer that is a power of 2."
        assert target_len >= 8, "target len should be at least of value 8."
        assert (
            0 <= residual_factor <= 1
        ), "residual factor must be included in [0,1]"
        self.target_len = target_len
        self.nb_step = int(log2(target_len)) - 2
        self.nb_features = nb_features
        self.channel_nb = channel_nb
        self.residual_factor = residual_factor
        self.initial_block = ConvResdiualSelfAttention(
            nb_in_features=nb_features + 1,
            nb_out_features=channel_nb,
            ks_conv=ks_conv,
            key_features=key_features,
            value_features=value_features,
            ks_value=ks_value,
            ks_query=ks_query,
            ks_key=ks_key,
            skip_param=0.0,
            self_attention=self_attention,
        )

        self.last_block = spectral_norm(
            nn.Conv1d(
                in_channels=channel_nb,
                out_channels=1,
                kernel_size=1,
            )
        )
        if cardinality is not None:
            assert (
                len(cardinality) == 1
            ), "MyGAN only supports a single static cat feature for now"
            self.embedding = nn.Embedding(cardinality[0], embedding_dim)
        self.embedding_dim = embedding_dim
        self.cardinality = cardinality

        self.block_list = nn.ModuleList([])
        for stage in range(1, self.nb_step):
            self.block_list.append(
                ConvResdiualSelfAttention(
                    nb_in_features=channel_nb + self.nb_features,
                    nb_out_features=channel_nb,
                    ks_conv=ks_conv,
                    key_features=key_features,
                    value_features=value_features,
                    ks_value=ks_value,
                    ks_query=ks_query,
                    ks_key=ks_key,
                    skip_param=0.0,
                    self_attention=self_attention,
                )
            )
        self.skip_block_list = nn.ModuleList([])
        for stage in range(1, self.nb_step):
            self.skip_block_list.append(
                nn.Conv1d(
                    in_channels=channel_nb,
                    out_channels=channel_nb,
                    kernel_size=1,
                )
            )
        # self.sigmoid = nn.Sigmoid()
        # self.activation = customActivation()
        self.softmax = nn.Softmax(dim=1)
        self.context_length = context_length
        if self.context_length > 0:
            self.mlp_block_list = nn.ModuleList([])
            for stage in range(1, self.nb_step):
                self.mlp_block_list.append(
                    nn.Sequential(
                        nn.Linear(
                            in_features=self.context_length + channel_nb,
                            out_features=self.context_length // 2 + channel_nb,
                        ),
                        nn.LeakyReLU(),
                        nn.Linear(
                            in_features=self.context_length // 2 + channel_nb,
                            out_features=channel_nb,
                        ),
                    )
                )

    def _softmax_min_max_scaling(self, target: torch.Tensor, alpha: int = 100):
        _min = torch.sum(
            target * self.softmax(-alpha * target), dim=1, keepdim=True
        )
        _max = torch.sum(
            target * self.softmax(alpha * target), dim=1, keepdim=True
        )

        return (target - _min) / (_max - _min)

    # def _get_noise(self, time_features: torch.Tensor):
    #     bs = time_features.size(0)
    #     target_len = time_features.size(2)
    #     noise = torch.randn((bs, 1, target_len), device=self.device)
    #     noise = torch.cat((time_features, noise), dim=1)
    #     return noise

    def forward(
        self,
        x: torch.Tensor,
        depth: int = None,
        residual: bool = False,
        feat_static_cat: torch.Tensor = None,
        context: torch.Tensor = None,
    ):
        """Computes the forward

        Arguments:
            x:
                torch.tensor of shape (batch size, 1 + nb_features, target_len)
            depth:
                the depth at which the the tensor should flow.
            feat_static_cat:
                static features of shape (batch size, 1)
        Returns
            torch.tensor of shape (batch_size, 1, length_out).
            length_out will depend on the current stage we are in. It is included between
            8 and target_len
        """
        if depth is None:
            depth = self.nb_step - 1

        assert x.dim() == 3, "input must be three dimensional"
        assert (
            x.size(2) == self.target_len
        ), "third dimension of input must be equal to target_len"
        assert depth <= self.nb_step - 1, "depth is too high"
        if self.cardinality is not None:
            em = (
                self.embedding(feat_static_cat.long())
                .permute(0, 2, 1)
                .expand(x.size(0), self.embedding_dim, x.size(2))
            )
            x = torch.cat((em, x), dim=1)

        # if self.scaling=="global" or self.scaling=="NoScale":
        #     x  = self._get_noise(x)
        reduced_x = F.avg_pool1d(
            x, kernel_size=2 ** (self.nb_step - 1)
        )  # Reduce x to length 8
        y = self.initial_block(reduced_x)
        for idx, l in enumerate(self.block_list[:depth]):
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            previous_y = y
            tf = F.avg_pool1d(
                x[:, :-1, :], kernel_size=2 ** (self.nb_step - 1 - (idx + 1))
            )  # time features reduced
            if self.context_length > 0:
                context_reshaped = (
                    context[:, -self.context_length :]
                    .unsqueeze(-1)
                    .expand(y.shape[0], self.context_length, y.shape[2])
                )
                y_mlp = torch.cat((context_reshaped, y), dim=1)
                y_mlp = self.mlp_block_list[idx](
                    y_mlp.permute(0, 2, 1)
                ).permute(0, 2, 1)
                y = y + y_mlp
            y = torch.cat((tf, y), dim=1)
            y = l(y)
            last_idx = idx

        # if depth > 0:
        #     l_skip = self.last_block[depth-1](self.skip_block_list[last_idx])

        if residual and depth > 0:
            l_skip = self.skip_block_list[last_idx]
            # y = self.residual_factor * self.last_block[depth](y).squeeze(1) + (1 - self.residual_factor) * self.last_block[depth-1](l_skip(
            #     previous_y
            # )).squeeze(1)
            y = self.residual_factor * self.last_block(y).squeeze(1) + (
                1 - self.residual_factor
            ) * self.last_block(l_skip(previous_y)).squeeze(1)

        else:
            # y = self.last_block[depth](y).squeeze(1)
            y = self.last_block(y).squeeze(1)

        # y = self.activation(y).unsqueeze(1)
        if self.scaling == "local":
            y = self._softmax_min_max_scaling(y).unsqueeze(1)
        else:
            y = y.unsqueeze(1)
        return y


class ProGeneratorInference(nn.Module):
    @validated()
    def __init__(self, trained_generator: nn.Module):
        super(ProGeneratorInference, self).__init__()
        self.trained_generator = trained_generator

    @staticmethod
    def _get_noise(time_features: torch.Tensor):
        time_features = time_features.permute(0, 2, 1)
        bs = time_features.size(0)
        target_len = time_features.size(2)
        noise = torch.randn((bs, 1, target_len))
        noise = torch.cat((time_features, noise), dim=1)
        return noise

    def forward(
        self,
        past_target: torch.Tensor,
        time_features: torch.Tensor,
        feat_static_cat: torch.Tensor,
    ):
        noise = self._get_noise(time_features)
        return self.trained_generator(
            x=noise, feat_static_cat=feat_static_cat, context=past_target
        )


class ProDiscriminator(nn.Module):
    """
    Attributes:
        target_len:
            length of the longest time series that can be discriminated
    """

    @validated()
    def __init__(
        self,
        target_len: int,
        nb_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        residual_factor: float = 0.0,
        cardinality: Optional[List[int]] = None,
        embedding_dim: int = 10,
        self_attention: bool = True,
        channel_nb: int = 32,
    ):
        super(ProDiscriminator, self).__init__()
        assert target_len >= 8, "target length should be at least of value 8"
        assert (
            log2(target_len) % 1 == 0
        ), "input length must be an integer that is a power of 2."
        assert (
            0 <= residual_factor <= 1
        ), "residual factor must be included in [0,1]"
        self.target_len = target_len
        self.nb_step = (
            int(log2(target_len)) - 2
        )  # nb of step to go from 8 to target_len
        self.residual_factor = residual_factor
        self.channel_nb = channel_nb
        self.initial_block = nn.Sequential(
            spectral_norm(
                nn.Conv1d(
                    in_channels=1 + nb_features,
                    out_channels=channel_nb,
                    kernel_size=1,
                )
            ),
            nn.LeakyReLU(),
        )
        self.cardinality = cardinality
        if self.cardinality is not None:
            assert (
                len(cardinality) == 1
            ), "MyGAN only supports a single static cat feature for now"
            self.embedding = nn.Embedding(cardinality[0], embedding_dim)
        self.embedding_dim = embedding_dim
        self.cardinality = cardinality

        self.last_block = nn.Sequential(
            ConvResdiualSelfAttention(
                nb_in_features=channel_nb,
                nb_out_features=channel_nb,
                ks_conv=ks_conv,
                key_features=key_features,
                value_features=value_features,
                ks_value=ks_value,
                ks_query=ks_query,
                ks_key=ks_key,
                skip_param=0.0,
                self_attention=self_attention,
            ),
            spectral_norm(
                nn.Conv1d(
                    in_channels=channel_nb, out_channels=1, kernel_size=1
                )
            ),
            nn.LeakyReLU(),
        )
        self.fc = spectral_norm(nn.Linear(8, 1))
        self.block_list = nn.ModuleList([])
        for stage in range(self.nb_step - 1, 0, -1):
            self.block_list.append(
                ConvResdiualSelfAttention(
                    nb_in_features=channel_nb,
                    nb_out_features=channel_nb,
                    ks_conv=ks_conv,
                    key_features=key_features,
                    value_features=value_features,
                    ks_value=ks_value,
                    ks_query=ks_query,
                    ks_key=ks_key,
                    skip_param=0.0,
                    self_attention=self_attention,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        tf: torch.Tensor,
        feat_static_cat: torch.Tensor,
        depth: int,
        residual: bool = False,
    ):
        """Computes the forward pass

        Arguments:
            x:
                tensor of shape (batch size, 1, input_length)
            tf:
                time features of shape (batch_size, nb_features, target_len)
            depth:
                the depth at which the the tensor should flow.
        """
        assert x.dim() == 3, "input must be three dimensional"
        assert (
            x.size(2) >= 8
        ), "third dimension of input must be greater or equal than 8"
        assert (
            log2(x.size(2)) % 1 == 0
        ), "input length must be an integer that is a power of 2."
        assert (
            tf.size(2) == self.target_len
        ), "length of features should be equal to target len"
        if depth is None:
            depth = self.nb_step - 1
        reduce_factor = int(log2(self.target_len)) - int(log2(x.size(2)))
        if self.cardinality is not None:
            em = (
                self.embedding(feat_static_cat.long())
                .permute(0, 2, 1)
                .expand(tf.size(0), self.embedding_dim, tf.size(2))
            )
            tf = torch.cat((em, tf), dim=1)
        reduced_tf = F.avg_pool1d(tf, kernel_size=2 ** reduce_factor)

        if residual:
            pre_reduce_tf = F.avg_pool1d(
                tf, kernel_size=2 ** (reduce_factor + 1)
            )
            pre_x = F.avg_pool1d(x, kernel_size=2)
            # pre_x = self.initial_block[depth-1](
            #     torch.cat((pre_reduce_tf, pre_x), dim=1)
            # )
            pre_x = self.initial_block(
                torch.cat((pre_reduce_tf, pre_x), dim=1)
            )

        x = torch.cat((reduced_tf, x), dim=1)
        # x = self.initial_block[depth](x)
        x = self.initial_block(x)

        for idx, l in enumerate(self.block_list[reduce_factor:]):
            x = l(x)
            x = F.avg_pool1d(x, kernel_size=2)
            if idx == 0:
                if residual:
                    x = (
                        self.residual_factor * x
                        + (1 - self.residual_factor) * pre_x
                    )

        x = self.last_block(x)
        x = self.fc(x.squeeze(1))
        return x


class ProGenDiscrim(nn.Module):
    """Class to encapsulate in a module the generator and discriminator"""

    @validated()
    def __init__(self, discriminator: nn.Module, generator: nn.Module):
        super(ProGenDiscrim, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def forward(
        self,
        x: torch.Tensor,
        tf: torch.Tensor = None,
        depth: int = None,
        residual: bool = False,
        net: str = "generator",
        feat_static_cat: torch.Tensor = None,
        context: torch.Tensor = None,
    ):
        """Computes the forward pass

        If the argument net is discriminator, then it computes the forward pass of the discriminator,
        If the argument net is generator, then it computes the forward pass of the generator

        Arguments:
            x:
                torch.tensor of (batch size, 1, input_length) if net == discriminator
                or (batch size, 1 + nb_features, target_len) is net == generator
            tf:
                time features concatenated to the time series in the case of the
                discriminator being used. If net == discriminator then it is
                a tensor of shape  (batch size,nb_features, target_len)
                otherwise it is None
            depth:
                integer passed for the forward pass of the generator, specifies the depth
                at which the tensor will flow. If net == discriminator, then it is None
            residual:
                Boolean value that specifies if the network, after creating a new layer, should use
                a skip connection in addition, so that new layer fades in smoothly. Technique implemented
                in [1], Page 4, Figure 2.
            net:
                string that is either generator or discriminator. Specifies which network
                will be used to compute the forward pass.

        [1]: https://arxiv.org/pdf/1710.10196.pdf
        """
        assert (
            net == "generator" or net == "discriminator"
        ), "net must be generator or discriminator"
        if net == "generator":
            return self.generator(x, depth, residual, feat_static_cat, context)
        else:
            return self.discriminator(x, tf, feat_static_cat, depth, residual)
