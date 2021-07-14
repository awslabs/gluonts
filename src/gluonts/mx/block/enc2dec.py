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

from typing import Tuple

from mxnet.gluon import nn

from gluonts.core.component import validated
from gluonts.mx import Tensor


class Seq2SeqEnc2Dec(nn.HybridBlock):
    """
    Abstract class for any module that pass encoder to decoder, such as
    attention network.
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        encoder_output_static: Tensor,
        encoder_output_dynamic: Tensor,
        future_features_dynamic: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        encoder_output_static
            shape (batch_size, channels_seq[-1] + 1) or (N, C)

        encoder_output_dynamic
            shape (batch_size, sequence_length, channels_seq[-1] + 1) or (N, T, C)

        future_features_dynamic
            shape (batch_size, sequence_length, prediction_length=decoder_length, num_feat_dynamic) or (N, T, P, C`)
        """
        pass


class PassThroughEnc2Dec(Seq2SeqEnc2Dec):
    """
    Simplest class for passing encoder tensors do decoder. Passes through
    tensors, except that future_features_dynamic is dropped.
    """

    def hybrid_forward(
        self,
        F,
        encoder_output_static: Tensor,
        encoder_output_dynamic: Tensor,
        future_features_dynamic: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        encoder_output_static
            shape (batch_size, channels_seq[-1] + 1) or (N, C)

        encoder_output_dynamic
            shape (batch_size, sequence_length, channels_seq[-1] + 1) or (N, T, C)

        future_features_dynamic
            shape (batch_size, sequence_length, prediction_length=decoder_length,  num_feat_dynamic) or (N, T, P, C`)


        Returns
        -------
        Tensor
            shape (batch_size, channels_seq[-1] + 1) or (N, C)

        Tensor
            shape (batch_size, sequence_length, channels_seq[-1] + 1) or (N, T, C)
        """
        return encoder_output_static, encoder_output_dynamic


class FutureFeatIntegratorEnc2Dec(Seq2SeqEnc2Dec):
    """
    Integrates the encoder_output_dynamic and future_features_dynamic into one
    and passes them through as the dynamic input to the decoder.
    """

    def hybrid_forward(
        self,
        F,
        encoder_output_static: Tensor,
        encoder_output_dynamic: Tensor,
        future_features_dynamic: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        encoder_output_static
            shape (batch_size, channels_seq[-1] + 1) or (N, C)

        encoder_output_dynamic
            shape (batch_size, sequence_length, channels_seq[-1] + 1) or (N, T, C)

        future_features_dynamic
            shape (batch_size, sequence_length, prediction_length=decoder_length, num_feat_dynamic) or (N, T, P, C`)


        Returns
        -------
        Tensor
            shape (batch_size, channels_seq[-1] + 1) or (N, C)

        Tensor
            shape (batch_size, prediction_length=decoder_length, channels_seq[-1] + 1 + decoder_length * num_feat_dynamic) or (N, T, C)

        """

        # flatten the last two dimensions:
        # => (batch_size, sequence_length, decoder_length * num_feat_dynamic), where
        # num_future_feat_dynamic = decoder_length * num_feat_dynamic
        future_features_dynamic = F.reshape(
            future_features_dynamic, shape=(0, 0, -1)
        )

        # concatenate output of decoder and future_feat_dynamic covariates:
        # => (batch_size, sequence_length, num_dec_input_dynamic + num_future_feat_dynamic)
        total_dec_input_dynamic = F.concat(
            encoder_output_dynamic, future_features_dynamic, dim=2
        )

        return encoder_output_static, total_dec_input_dynamic
