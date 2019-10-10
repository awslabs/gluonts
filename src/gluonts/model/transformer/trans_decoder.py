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

# Standard library imports
from typing import Dict, Optional

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.model.transformer.layers import (
    TransformerProcessBlock,
    TransformerFeedForward,
    MultiHeadSelfAttention,
    MultiHeadAttention,
    InputLayer,
)


class TransformerDecoder(HybridBlock):
    @validated()
    def __init__(self, decoder_length: int, config: Dict, **kwargs) -> None:

        super().__init__(**kwargs)

        self.decoder_length = decoder_length
        self.cache = {}

        with self.name_scope():
            self.enc_input_layer = InputLayer(model_size=config["model_dim"])

            self.dec_pre_self_att = TransformerProcessBlock(
                sequence=config["pre_seq"],
                dropout=config["dropout_rate"],
                prefix="pretransformerprocessblock_",
            )
            self.dec_self_att = MultiHeadSelfAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="multiheadselfattention_",
            )
            self.dec_post_self_att = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postselfatttransformerprocessblock_",
            )
            self.dec_enc_att = MultiHeadAttention(
                att_dim_in=config["model_dim"],
                heads=config["num_heads"],
                att_dim_out=config["model_dim"],
                dropout=config["dropout_rate"],
                prefix="multiheadattention_",
            )
            self.dec_post_att = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postatttransformerprocessblock_",
            )
            self.dec_ff = TransformerFeedForward(
                inner_dim=config["model_dim"] * config["inner_ff_dim_scale"],
                out_dim=config["model_dim"],
                act_type=config["act_type"],
                dropout=config["dropout_rate"],
                prefix="transformerfeedforward_",
            )
            self.dec_post_ff = TransformerProcessBlock(
                sequence=config["post_seq"],
                dropout=config["dropout_rate"],
                prefix="postffransformerprocessblock_",
            )

    def cache_reset(self):
        self.cache = {}

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        enc_out: Tensor,
        mask: Optional[Tensor] = None,
        is_train: bool = True,
    ) -> Tensor:

        """
        A transformer encoder block consists of a self-attention and a feed-forward layer with pre/post process blocks
        in between.
        """

        # embedding
        inputs = self.enc_input_layer(data)

        # self-attention
        data_att, cache = self.dec_self_att(
            self.dec_pre_self_att(inputs, None),
            mask,
            self.cache.copy() if not is_train else None,
        )
        data = self.dec_post_self_att(data_att, inputs)

        # encoder attention
        data_att = self.dec_enc_att(data, enc_out)
        data = self.dec_post_att(data_att, data)

        # feed-forward
        data_ff = self.dec_ff(data)
        data = self.dec_post_ff(data_ff, data)

        if not is_train:
            self.cache = cache.copy()

        return data
