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

from typing import List, Tuple, Union

import torch
from torch import nn, Tensor

from gluonts.core.component import validated
from gluonts.torch.modules.lambda_layer import LambdaLayer


def _get_int(a: Union[int, List[int], Tuple[int]]) -> int:
    if isinstance(a, (list, tuple)):
        assert len(a) == 1
        return a[0]
    return a


class CausalConv1D(nn.Module):
    """
    1D causal temporal convolution, where the term causal means that output[t]
    does not depend on input[t+1:]. Notice that Conv1D is not implemented in
    Gluon.

    This is the basic structure used in Wavenet [ODZ+16]_ and Temporal
    Convolution Network [BKK18]_.

    The output has the same shape as the input, while we always left-pad zeros.

    Parameters
    ----------

    layer_no
        layer number
    init_dim
        input dimension into the layer
    channels
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size
        Specifies the dimensions of the convolution window.
    dilation
        Specifies the dilation rate to use for dilated convolution.
    """

    def __init__(
        self,
        layer_no: int,
        init_dim: int,
        channels: int,
        kernel_size: Union[int, Tuple[int], List[int]],
        dilation: Union[int, Tuple[int], List[int]] = 1,
        **kwargs,
    ):
        super().__init__()

        self.dilation = _get_int(dilation)
        self.kernel_size = _get_int(kernel_size)
        self.padding = self.dilation * (self.kernel_size - 1)
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=init_dim if layer_no == 0 else channels,
                out_channels=channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.ReLU(),
        )

    def forward(self, data: Tensor) -> Tensor:
        """
        In Gluon's conv1D implementation, input has dimension NCW where N is
        batch_size, C is channel, and W is time (sequence_length).

        Parameters
        ----------
        data
            Shape (batch_size, num_features, sequence_length)

        Returns
        -------
        Tensor
            causal conv1d output. Shape (batch_size, num_features,
            sequence_length)
        """
        ct = self.conv1d(data)
        if self.kernel_size > 0:
            ct = ct[:, :, : ct.shape[2] - self.padding]
        return ct


class HierarchicalCausalConv1DEncoder(nn.Module):
    """
    Defines a stack of dilated convolutions as the encoder.
    See the following paper for details:
    1. Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner,
    N., Senior, A.W. and Kavukcuoglu, K., 2016, September. WaveNet: A generative model for raw audio. In SSW (p. 125).
    Parameters
    ----------
    cnn_init_dim
        input dimension into CNN encoder
    dilation_seq
        dilation for each convolution in the stack.
    kernel_size_seq
        kernel size for each convolution in the stack.
    channels_seq
        number of channels for each convolution in the stack.
    joint_embedding_dimension (int):
        final dimension to embed all static features
    hidden_dimension_sequence (List[int], optional):
        list of hidden dimensions for the MLP used to embed static features. Defaults to [].
    use_residual
        flag to toggle using residual connections.
    use_static_feat
        flag to toggle whether to use use_static_feat as input to the encoder
    use_dynamic_feat
        flag to toggle whether to use use_dynamic_feat as input to the encoder
    """

    @validated()
    def __init__(
        self,
        cnn_init_dim: int,
        dilation_seq: List[int],
        kernel_size_seq: List[int],
        channels_seq: List[int],
        joint_embedding_dimension: int,
        mlp_init_dim: int,
        hidden_dimension_sequence: List[int] = [],
        use_residual: bool = False,
        use_static_feat: bool = False,
        use_dynamic_feat: bool = False,
        **kwargs,
    ) -> None:

        assert all(
            [x > 0 for x in dilation_seq]
        ), "`dilation_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in kernel_size_seq]
        ), "`kernel_size_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in channels_seq]
        ), "`channel_dim_seq` values must be greater than zero"

        super().__init__()

        self.use_residual = use_residual
        self.use_static_feat = use_static_feat
        self.use_dynamic_feat = use_dynamic_feat

        # CNN for dynamic features (and/or target)
        self.cnn = nn.Sequential()

        # swap axes because Conv1D expects NCT
        self.cnn.append(LambdaLayer(lambda x: torch.transpose(x, 2, 1)))

        it = zip(channels_seq, kernel_size_seq, dilation_seq)
        for layer_no, (channels, kernel_size, dilation) in enumerate(it):

            convolution = CausalConv1D(
                layer_no=layer_no,
                init_dim=cnn_init_dim,
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.cnn.append(convolution)

        # swap axes to get back to NTC
        self.cnn.append(LambdaLayer(lambda x: torch.transpose(x, 2, 1)))

        # MLP for static features
        modules: List[nn.Module] = []
        mlp_dimension_sequence = (
            [mlp_init_dim]
            + hidden_dimension_sequence
            + [joint_embedding_dimension]
        )
        if use_static_feat:

            for in_features, out_features in zip(
                mlp_dimension_sequence[:-1], mlp_dimension_sequence[1:]
            ):
                layer = nn.Linear(
                    in_features,
                    out_features,
                )
                modules += [layer, nn.ReLU()]

            self.static = nn.Sequential(*modules)

    def forward(
        self,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        target
            target time series,
            shape (batch_size, sequence_length, 1)
        static_features
            static features,
            shape (batch_size, num_feat_static)
        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_feat_dynamic)
        Returns
        -------
        Tensor
            static code,
            shape (batch_size, channel_seqs + (1) if use_residual)
        Tensor
            dynamic code,
            shape (batch_size, sequence_length, channel_seqs + (1) if use_residual)
        """

        if self.use_dynamic_feat:
            dynamic_inputs = torch.cat(
                (target, dynamic_features), dim=2
            )  # (N, T, C)
        else:
            dynamic_inputs = target

        dynamic_encoded = self.cnn(dynamic_inputs.float())

        if self.use_residual:
            dynamic_encoded = torch.cat((dynamic_encoded, target), dim=2)

        if self.use_static_feat:
            static_encoded = self.static(static_features)
        else:
            static_encoded = None

        # we return them separately so that static features can be replicated
        return static_encoded, dynamic_encoded


class Enc2Dec(nn.Module):
    """
    Integrates the encoder_output_static, encoder_output_dynamic and future_features_dynamic
    and passes them through as the dynamic input to the decoder.

    Parameters:
    ------------
    num_forking [int]:
            number of forks
    """

    @validated()
    def __init__(
        self,
        num_forking: int,
        **kwargs,
    ) -> None:

        super().__init__()
        self.num_forking = num_forking

    def forward(
        self,
        encoder_output_static: torch.Tensor,
        encoder_output_dynamic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        encoder_output_static
            shape (batch_size, num_feat_static) or (N, C)
        encoder_output_dynamic
            shape (batch_size, sequence_length, channels_seq[-1] + 1) or (N, T, C)
        Returns
        -------
        Tensor
            shape (batch_size, sequence_length, channels_seq[-1] + 1 + num_feat_static) or (N, C)
        """

        encoder_output_static = torch.unsqueeze(encoder_output_static, dim=1)
        encoder_output_static_expanded = torch.repeat_interleave(
            encoder_output_static, repeats=self.num_forking, dim=1
        )

        # concatenate static and dynamic output of the encoder
        # => (batch_size, sequence_length, num_enc_output_dynamic + num_enc_output_static)
        encoder_output = torch.cat(
            (encoder_output_dynamic, encoder_output_static_expanded), dim=2
        )

        return encoder_output


class ForkingMLPDecoder(nn.Module):
    """
    Multilayer perceptron decoder for sequence-to-sequence models.
    See [WTN+17]_ for details.
    Parameters
    ----------
    dec_len
        length of the decoder (usually the number of forecasted time steps).
    encoded_input_dim (int):
        input dimension out of encoder
    local_mlp_final_dim (int):
        final dimension of the local mlp (output of the decoder before quantile output layer)
    global_mlp_final_dim (int):
        final dimension of the horizon agnostic part of the global mlp. Note that horizon specific part will use //2 dimension.
    future_feat_embedding_dim (int):
        dimension of the embedding layer for global encoding of future features.
    local_mlp_hidden_dim_sequence (List[int], optional):
        dimensions of local mlp hidden layers. Defaults to [].
    """

    @validated()
    def __init__(
        self,
        dec_len: int,
        encoded_input_dim: int,
        local_mlp_final_dim: int,
        global_mlp_final_dim: int,
        future_feat_init_dim: int,
        future_feat_embedding_dim: int,
        local_mlp_hidden_dim_sequence: List[int] = [],
        **kwargs,
    ) -> None:

        super().__init__()

        self.dec_len = dec_len

        # Global embeddings for future dynamic features [N, T, C],
        # where N - batch_size, T - number of forks,
        # C - joint embedding dimension for future features
        self.global_future_layer = self._get_global_future_layer(
            future_feat_init_dim, future_feat_embedding_dim
        )

        # Local embeddings for future dynamic features [N, T, K, C]
        # where N - batch_size, T - number of forks, K - number of horizons (dec_len),
        # C - number of future dynamic features per horizon (same dimensions as input to decoder)
        self.local_future_layer = self._get_local_future_layer()

        # Horizon specific global MLP outputs [N, T, K, C],
        # C - number of outputs per horizon (global mlp final dimension // 2)
        input_dim = encoded_input_dim + future_feat_embedding_dim
        horizon_specific_dim = global_mlp_final_dim // 2
        self.horizon_specific = self._get_horizon_specific(
            input_dim, horizon_specific_dim
        )

        # Horizon agnostic global MLP outputs [N, T, K, C],
        # C - number of identical outputs per horizon (global mlp final dimension)
        self.horizon_agnostic = self._get_horizon_agnostic(
            input_dim, global_mlp_final_dim
        )

        # Local MLP outputs [N, T, K, C],
        # C - number of outputs per horizon (local mlp final dimension)
        local_mlp_init_dim = (
            horizon_specific_dim + global_mlp_final_dim + future_feat_init_dim
        )
        self.local_mlp = self._get_local_mlp(
            local_mlp_init_dim,
            local_mlp_final_dim,
            local_mlp_hidden_dim_sequence,
        )

    def _get_global_future_layer(self, input_size, embedding_dim):
        layer = nn.Sequential()
        layer.append(
            LambdaLayer(
                lambda x: torch.reshape(x, (x.shape[0], x.shape[1], -1))
            )
        )  ## [N, T, K, C] where T is number of forks, K number of horizons (dec_len)

        layer.append(nn.Linear(input_size * self.dec_len, embedding_dim))
        layer.append(nn.Tanh())  # [N, T, embedding_dim]
        return layer

    def _get_local_future_layer(self):
        layer = nn.Sequential()
        layer.append(nn.Tanh())  ##[N, T, K, C]
        return layer

    def _get_horizon_specific(self, input_size, units_per_horizon):
        mlp = nn.Sequential()
        mlp.append(nn.Linear(input_size, self.dec_len * units_per_horizon))
        mlp.append(nn.ReLU())
        mlp.append(
            LambdaLayer(
                lambda x: torch.reshape(
                    x, (x.shape[0], x.shape[1], self.dec_len, -1)
                )
            )
        )
        return mlp

    def _get_horizon_agnostic(self, input_size, hidden_size):
        mlp = nn.Sequential()
        mlp.append(nn.Linear(input_size, hidden_size))
        mlp.append(nn.ReLU())
        mlp.append(LambdaLayer(lambda x: torch.unsqueeze(x, dim=2)))
        mlp.append(
            LambdaLayer(
                lambda x: torch.repeat_interleave(
                    x, repeats=self.dec_len, dim=2
                )
            )
        )
        return mlp

    def _get_local_mlp(self, init_dim, final_dim, hidden_dimension_seq):
        modules: List[nn.Module] = []
        dimensions = [init_dim] + hidden_dimension_seq

        for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
            layer = nn.Linear(
                in_features,
                out_features,
            )
            modules += [layer, nn.ReLU()]

        modules += [nn.Linear(dimensions[-1], final_dim), nn.Softplus()]

        local_mlp = nn.Sequential(*modules)
        return local_mlp

    def forward(self, encoded_input: Tensor, future_input: Tensor) -> Tensor:
        """Forward pass for MQCNN decoder

        Args:
            encoded_input (Tensor):
        decoder input from the output of the MQCNN encoder, including static and past dynamic feature encoding
            future_input (Tensor):
        decoder input from future dynamic features

        Returns:
            output of the decoder (MQCNN)
        """

        # Embed future features globally
        global_future_embedded = self.global_future_layer(future_input.float())

        # Encode future features locally for each timestep/feature
        local_future_encoded = self.local_future_layer(future_input)

        # Combine encoded historical dynamic and static features and globally embedded future features
        encoded_input_and_future = torch.cat(
            (encoded_input, global_future_embedded), dim=-1
        )

        # Produce horizon specific encoding (c_t for K horizons in the paper)
        horizon_specific_encoded = self.horizon_specific(
            encoded_input_and_future
        )

        # Produce horizon agnostic encoding (c_a in the paper)
        horizon_agnostic_encoded = self.horizon_agnostic(
            encoded_input_and_future
        )

        # Combine horizon agnostic, horizon specific and future local encodings
        encoded = torch.cat(
            (
                horizon_specific_encoded,
                horizon_agnostic_encoded,
                local_future_encoded,
            ),
            dim=-1,
        )

        # Train local mlp on each fork and horizon (weights are shared)
        output = self.local_mlp(encoded.float())

        return output
