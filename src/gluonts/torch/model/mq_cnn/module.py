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

from typing import Optional, Tuple, List

import torch
from torch import nn
from gluonts.core.component import validated
from gluonts.torch.distributions import Output
from gluonts.model import Input, InputSpec

from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler
from gluonts.torch.distributions.quantile_output import QuantileOutput
from gluonts.torch.util import weighted_average

from .layers import (
    HierarchicalCausalConv1DEncoder,
    Enc2Dec,
    ForkingMLPDecoder,
)


class MQCNNModel(nn.Module):
    """
    Base network for the :class:`MQCnnEstimator`.
    Parameters
    ----------
    context_length (int):
        length of the encoding sequence.
    prediction_length (int):
        prediction length
    num_forking (int):
        number of forks to do in the decoder.
    past_feat_dynamic_real_dim (int):
        dimension of past real dynamic features
    feat_static_real_dim (int):
        dimenstion of real static features
    feat_dynamic_real_dim (int):
        dimension of real dynamic features
    cardinality_dynamic (List[int]):
        cardinalities of dynamic categorical features
    embedding_dimension_dynamic (List[int]):
        embedding dymensions of dynamic categorical features
    cardinality_static (List[int]):
        cardinalities of static categorical features
    embedding_dimension_static (List[int]):
        embedding dimensions of static categorical features
    scaling (bool):
        if True, scale the target values
    scaling_decoder_dynamic_feature (bool):
        if True, scale the dynamic features for the decoder
    encoder_cnn_init_dim (int):
        input dimensions of encoder CNN
    dilation_seq (List[int]):
        dilation sequence of encoder CNN
    kernel_size_seq (List[int]):
        kernel sizes of encoder CNN
    channels_seq (List[int]):
        numbers of cannels of encoder CNN
    joint_embedding_dimension (int):
        joint embedding dimension of the encoder
    encoder_mlp_init_dim (int):
        input dimension of static features encoder MLP
    encoder_mlp_dim_seq (List[int]):
        sequence of hidden layer dimentions of encoder MLP
    use_residual (bool):
        if True, target is added to encoder CNN output
    decoder_mlp_dim_seq (List[int]):
        sequence of layer dimensions of decoder MLP
    decoder_hidden_dim (int):
        decoder MLP hidden dimension
    decoder_future_init_dim (int):
        decoder init dimension for embedding future dynamic features
    decoder_future_embedding_dim (int):
        decoder embedding dimension for future dynamic features
    distr_output (Optional[Output]):
        distribution output block. Defaults to None,
    kwargs: dict
        dictionary of parameters
    """

    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_forking: int,
        past_feat_dynamic_real_dim: int,
        feat_static_real_dim: int,
        feat_dynamic_real_dim: int,
        cardinality_dynamic: List[int],
        embedding_dimension_dynamic: List[int],
        cardinality_static: List[int],
        embedding_dimension_static: List[int],
        scaling: bool,
        scaling_decoder_dynamic_feature: bool,
        encoder_cnn_init_dim: int,
        dilation_seq: List[int],
        kernel_size_seq: List[int],
        channels_seq: List[int],
        joint_embedding_dimension: int,
        encoder_mlp_init_dim: int,
        encoder_mlp_dim_seq: List[int],
        use_residual: bool,
        decoder_mlp_dim_seq: List[int],
        decoder_hidden_dim: int,
        decoder_future_init_dim: int,
        decoder_future_embedding_dim: int,
        distr_output: Optional[Output] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_forking = num_forking

        self.feat_static_real_dim = feat_static_real_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.cardinality_dynamic = cardinality_dynamic
        self.cardinality_static = cardinality_static

        self.embedder_dynamic = (
            FeatureEmbedder(
                cardinalities=cardinality_dynamic,
                embedding_dims=embedding_dimension_dynamic,
            )
            if len(cardinality_dynamic) > 0
            else None
        )

        self.embedder_dynamic_future = (
            FeatureEmbedder(
                cardinalities=cardinality_dynamic,
                embedding_dims=embedding_dimension_dynamic,
            )
            if len(cardinality_dynamic) > 0
            else None
        )

        self.embedder_static = (
            FeatureEmbedder(
                cardinalities=cardinality_static,
                embedding_dims=embedding_dimension_static,
            )
            if len(cardinality_static) > 0
            else None
        )

        self.scaling = scaling
        self.scaling_decoder_dynamic_feature = scaling_decoder_dynamic_feature

        if self.scaling:
            self.scaler = MeanScaler(dim=1)
        else:
            self.scaler = NOPScaler(dim=1)

        if self.scaling_decoder_dynamic_feature:
            self.scaler_decoder_dynamic_feature = MeanScaler(dim=1)
        else:
            self.scaler_decoder_dynamic_feature = NOPScaler(dim=1)

        self.encoder = HierarchicalCausalConv1DEncoder(
            cnn_init_dim=encoder_cnn_init_dim,
            dilation_seq=dilation_seq,
            kernel_size_seq=kernel_size_seq,
            channels_seq=channels_seq,
            joint_embedding_dimension=joint_embedding_dimension,
            mlp_init_dim=encoder_mlp_init_dim,
            hidden_dimension_sequence=encoder_mlp_dim_seq,
            use_residual=use_residual,
            use_static_feat=True,
            use_dynamic_feat=True,
        )

        self.enc2dec = Enc2Dec(num_forking=num_forking)

        self.decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            encoded_input_dim=channels_seq[-1] + joint_embedding_dimension + 1,
            local_mlp_final_dim=decoder_mlp_dim_seq[-1],
            global_mlp_final_dim=decoder_hidden_dim,
            future_feat_init_dim=decoder_future_init_dim,
            future_feat_embedding_dim=decoder_future_embedding_dim,
            local_mlp_hidden_dim_sequence=decoder_mlp_dim_seq[:-1],
        )

        if distr_output is None:
            distr_output = QuantileOutput(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
        self.distr_output = distr_output
        self.distr_proj = self.distr_output.get_args_proj(
            in_features=decoder_mlp_dim_seq[-1]
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length, 1),
                    dtype=torch.float,
                ),
                "past_feat_dynamic": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        self.past_feat_dynamic_real_dim,
                    ),
                    dtype=torch.float,
                ),
                "future_feat_dynamic": Input(
                    shape=(
                        batch_size,
                        self.num_forking,
                        self.prediction_length,
                        self.feat_dynamic_real_dim,
                    ),
                    dtype=torch.float,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.feat_static_real_dim),
                    dtype=torch.float,
                ),
                "feat_static_cat": Input(
                    shape=(batch_size, len(self.cardinality_static)),
                    dtype=torch.long,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length, 1),
                    dtype=torch.float,
                ),
                "past_feat_dynamic_cat": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        len(self.cardinality_dynamic),
                    ),
                    dtype=torch.long,
                ),
                "future_feat_dynamic_cat": Input(
                    shape=(
                        batch_size,
                        self.num_forking,
                        self.prediction_length,
                        len(self.cardinality_dynamic),
                    ),
                    dtype=torch.long,
                ),
            },
            torch.zeros,
        )

    # this method connects the sub-networks and returns the decoder output
    def get_decoder_network_output(
        self,
        past_target: torch.Tensor,
        past_feat_dynamic: torch.Tensor,
        future_feat_dynamic: torch.Tensor,
        feat_static: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        past_target: torch.Tensor
            shape (batch_size, encoder_length, 1)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        feat_static
            shape (batch_size, num_feat_static_real)
        past_observed_values: torch.Tensor
            shape (batch_size, encoder_length, 1)
        Returns
        -------
        decoder output, loc and scale
        """

        # scale shape: (batch_size, 1, 1)
        scaled_past_target, loc, scale = self.scaler(
            past_target, past_observed_values
        )

        # in addition to embedding features, use the log scale as it can help prediction too
        # (batch_size, num_feat_static = sum(embedding_dimension) + 1)
        feat_static = torch.cat((feat_static, torch.log(scale)), dim=1)

        # Passing past_observed_values as a feature would allow the network to
        # make that distinction and possibly ignore the masked values.
        past_feat_dynamic_extended = torch.cat(
            (past_feat_dynamic, past_observed_values), dim=-1
        )

        # arguments: target, static_features, dynamic_features
        # enc_output_static shape: (batch_size, channels_seq[-1] + 1)
        # enc_output_dynamic shape: (batch_size, encoder_length, channels_seq[-1] + 1)
        enc_output_static, enc_output_dynamic = self.encoder(
            scaled_past_target, feat_static, past_feat_dynamic_extended
        )

        # arguments: encoder_output_static, encoder_output_dynamic, future_features
        # dec_input_static shape: (batch_size, channels_seq[-1] + 1)
        # dec_input_dynamic shape:(batch_size, num_forking, channels_seq[-1] + 1 + decoder_length * num_feat_dynamic)
        dec_input_encoded = self.enc2dec(
            enc_output_static,
            # slice axis 1 from encoder_length = context_length to num_forking
            enc_output_dynamic[
                :, -self.num_forking : enc_output_dynamic.shape[1], ...
            ],
        )

        scaled_future_feat_dynamic, _, _ = self.scaler_decoder_dynamic_feature(
            future_feat_dynamic, torch.ones_like(future_feat_dynamic)
        )

        # arguments: dynamic_input, static_input
        dec_output = self.decoder(
            dec_input_encoded, scaled_future_feat_dynamic
        )

        # the output shape should be: (batch_size, num_forking, dec_len, decoder_mlp_dim_seq[0])
        return dec_output, loc, scale

    # noinspection PyMethodOverriding
    def forward(
        self,
        past_target: torch.Tensor,
        past_feat_dynamic: torch.Tensor,
        future_feat_dynamic: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        future_feat_dynamic_cat: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        past_target: torch.Tensor
            shape (batch_size, encoder_length, 1)
        past_feat_dynamic
            shape (batch_size, encoder_length, past_feat_dynamic_dim)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, feat_dynamic_dim)
        feat_static_real
            shape (batch_size, feat_static_real_dim)
        feat_static_cat
            shape (batch_size, feat_static_cat_dim)
        past_observed_values: torch.Tensor
            shape (batch_size, encoder_length, 1)
        future_observed_values: torch.Tensor
            shape (batch_size, num_forking, decoder_length)
        past_feat_dynamic_cat: torch.Tensor,
            shape (batch_size, encoder_length, feature_dynamic_cat_dim)
        future_feat_dynamic_cat: torch.Tensor,
            shape (batch_size, num_forking, decoder_length, feature_dynamic_cat_dim)
        Returns
        -------
        distr_args, loc, scale
        """

        if self.embedder_static is not None:
            embedded_cat = self.embedder_static(feat_static_cat)
            feat_static = torch.cat((embedded_cat, feat_static_real), dim=1)
        else:
            feat_static = torch.add(feat_static_real, feat_static_cat)

        if self.embedder_dynamic is not None:

            # Embed dynamic categorical features
            embedded_past_feature_dynamic_cat = self.embedder_dynamic(
                past_feat_dynamic_cat
            )
            embedded_future_feature_dynamic_cat = self.embedder_dynamic_future(
                future_feat_dynamic_cat
            )
            # Combine all dynamic features
            past_feat_dynamic = torch.cat(
                (past_feat_dynamic, embedded_past_feature_dynamic_cat), dim=-1
            )
            future_feat_dynamic = torch.cat(
                (future_feat_dynamic, embedded_future_feature_dynamic_cat),
                dim=-1,
            )

        else:
            # Make sure that future_feat_dynamic_cat also has [N, T, H, C] dimensions
            future_feat_dynamic_cat = torch.reshape(
                future_feat_dynamic_cat, [0, 0, 0, -1]
            )

            past_feat_dynamic = torch.add(
                past_feat_dynamic, past_feat_dynamic_cat
            )
            future_feat_dynamic = torch.add(
                future_feat_dynamic, future_feat_dynamic_cat
            )

        # shape: (batch_size, num_forking, decoder_length, decoder_mlp_dim_seq[0])
        dec_output, loc, scale = self.get_decoder_network_output(
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static,
            past_observed_values,
        )

        # shape: (batch_size, num_forking, decoder_length, len(quantiles))
        distr_args = self.distr_proj(dec_output)
        return distr_args, loc, scale

    # noinspection PyMethodOverriding
    def loss(
        self,
        past_target: torch.Tensor,
        future_target: torch.Tensor,
        past_feat_dynamic: torch.Tensor,
        future_feat_dynamic: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        future_feat_dynamic_cat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        past_target: torch.Tensor
            shape (batch_size, encoder_length, 1)
        future_target: torch.Tensor
            shape (batch_size, num_forking, decoder_length)
        past_feat_dynamic
            shape (batch_size, encoder_length, past_feat_dynamic_dim)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, feat_dynamic_dim)
        feat_static_real
            shape (batch_size, feat_static_real_dim)
        feat_static_cat
            shape (batch_size, feat_static_cat_dim)
        past_observed_values: torch.Tensor
            shape (batch_size, encoder_length, 1)
        future_observed_values: torch.Tensor
            shape (batch_size, num_forking, decoder_length)
        past_feat_dynamic_cat: torch.Tensor,
            shape (batch_size, encoder_length, feature_dynamic_cat_dim)
        future_feat_dynamic_cat: torch.Tensor,
            shape (batch_size, num_forking, decoder_length, feature_dynamic_cat_dim)
        Returns
        -------
        loss with shape (batch_size, prediction_length)
        """

        distr_args, loc, scale = self(
            past_target,
            past_feat_dynamic,
            future_feat_dynamic,
            feat_static_real,
            feat_static_cat,
            past_observed_values,
            past_feat_dynamic_cat,
            future_feat_dynamic_cat,
        )

        # shape: (batch_size, num_forking, decoder_length = prediction_length)
        loss = self.distr_output.loss(
            target=future_target,
            distr_args=distr_args,
            loc=loc.unsqueeze(-1),
            scale=scale.unsqueeze(-1),
        )

        # mask the loss based on observed indicator
        # shape: (batch_size, decoder_length)
        return weighted_average(x=loss, weights=future_observed_values, dim=1)
