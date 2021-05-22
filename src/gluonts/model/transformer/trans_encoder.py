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

from typing import Dict

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.model.transformer.layers import (
    InputLayer,
    MultiHeadSelfAttention,
    TransformerFeedForward,
    TransformerProcessBlock,
)
from gluonts.mx import Tensor


class TransformerEncoder(HybridBlock):
    @validated()
    def __init__(self, encoder_length: int, config: Dict, **kwargs) -> None:

        super().__init__(**kwargs)

        self.encoder_length = encoder_length

        with self.name_scope():
            self.enc_input_layer = InputLayer(model_size=config["model_dim"])

            self.enc_pre_self_att = TransformerProcessBlock(
                sequence=config["pre_seq"],
                dropout=config["dropout_rate"],
                prefix="pretransformerprocessblock_",
            )
            self.enc_self_att = MultiHeadSelfAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="multiheadselfattention_",
            )
            self.enc_post_self_att = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postselfatttransformerprocessblock_",
            )
            self.enc_ff = TransformerFeedForward(
                inner_dim=config["model_dim"] * config["inner_ff_dim_scale"],
                out_dim=config["model_dim"],
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
                prefix="transformerfeedforward_",
            )
            self.enc_post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postfftransformerprocessblock_",
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, data: Tensor) -> Tensor:

        """
        A transformer encoder block consists of a self-attention and a feed-forward layer with pre/post process blocks
        in between.
        """

        # input layer
        inputs = self.enc_input_layer(data)

        # self-attention
        data_self_att, _ = self.enc_self_att(
            self.enc_pre_self_att(inputs, None)
        )
        data = self.enc_post_self_att(data_self_att, inputs)

        # feed-forward
        data_ff = self.enc_ff(data)
        data = self.enc_post_ff(data_ff, data)

        return data
