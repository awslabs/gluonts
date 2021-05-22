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

import math
from typing import List, Optional, Tuple

import mxnet as mx
import numpy as np
from mxnet import gluon, init
from mxnet.gluon import HybridBlock, nn, rnn

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder


class GatedLinearUnit(HybridBlock):
    @validated()
    def __init__(self, axis: int = -1, nonlinear: bool = True, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.axis = axis
        self.nonlinear = nonlinear

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        val, gate = F.split(x, axis=self.axis, num_outputs=2)
        if self.nonlinear:
            val = F.tanh(val)
        gate = F.sigmoid(gate)
        return F.broadcast_mul(gate, val)


class GatedResidualNetwork(HybridBlock):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        d_input: Optional[int] = None,
        d_output: Optional[int] = None,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.d_hidden = d_hidden
        self.d_input = d_input or d_hidden
        self.d_static = d_static or 0
        if d_output is None:
            self.d_output = self.d_input
            self.add_skip = False
        else:
            self.d_output = d_output
            if d_output != self.d_input:
                self.add_skip = True
                with self.name_scope():
                    self.skip_proj = nn.Dense(
                        units=self.d_output,
                        in_units=self.d_input,
                        flatten=False,
                        weight_initializer=init.Xavier(),
                    )
            else:
                self.add_skip = False

        with self.name_scope():
            self.mlp = nn.HybridSequential(prefix="mlp_")
            self.mlp.add(
                nn.Dense(
                    units=self.d_hidden,
                    in_units=self.d_input + self.d_static,
                    flatten=False,
                    weight_initializer=init.Xavier(),
                )
            )
            self.mlp.add(nn.ELU())
            self.mlp.add(
                nn.Dense(
                    units=self.d_hidden,
                    in_units=self.d_hidden,
                    flatten=False,
                    weight_initializer=init.Xavier(),
                )
            )
            self.mlp.add(nn.Dropout(dropout)),
            self.mlp.add(
                nn.Dense(
                    units=self.d_output * 2,
                    in_units=self.d_hidden,
                    flatten=False,
                    weight_initializer=init.Xavier(),
                )
            )
            self.mlp.add(
                GatedLinearUnit(
                    axis=-1,
                    nonlinear=False,
                )
            )
            self.lnorm = nn.LayerNorm(axis=-1, in_channels=self.d_output)

    def hybrid_forward(
        self,
        F,
        x: Tensor,
        c: Optional[Tensor] = None,
    ) -> Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x
        if self.d_static > 0 and c is None:
            raise ValueError("static variable is expected.")
        if self.d_static == 0 and c is not None:
            raise ValueError("static variable is not accpeted.")
        if c is not None:
            x = F.concat(x, c, dim=-1)
        x = self.mlp(x)
        x = self.lnorm(F.broadcast_add(x, skip))
        return x


class VariableSelectionNetwork(HybridBlock):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        n_vars: int,
        dropout: float = 0.0,
        add_static: bool = False,
        **kwargs,
    ) -> None:
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.d_hidden = d_hidden
        self.n_vars = n_vars
        self.add_static = add_static

        with self.name_scope():
            self.weight_network = GatedResidualNetwork(
                d_hidden=self.d_hidden,
                d_input=self.d_hidden * self.n_vars,
                d_output=self.n_vars,
                d_static=self.d_hidden if add_static else None,
                dropout=dropout,
            )
            self.variable_network = []
            for n in range(self.n_vars):
                var_net = GatedResidualNetwork(
                    d_hidden=self.d_hidden,
                    dropout=dropout,
                )
                self.register_child(var_net, name=f"var_{n+1}")
                self.variable_network.append(var_net)

    def hybrid_forward(
        self,
        F,
        variables: List[Tensor],
        static: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(variables) != self.n_vars:
            raise ValueError(
                f"expect {self.n_vars} variables, {len(variables)} given."
            )
        if self.add_static and static is None:
            raise ValueError("static variable is expected.")
        if not self.add_static and static is not None:
            raise ValueError("static variable is not accpeted.")
        flatten = F.concat(*variables, dim=-1)
        if static is not None:
            static = F.broadcast_like(static, variables[0])
        weight = self.weight_network(flatten, static)
        weight = F.expand_dims(weight, axis=-2)
        weight = F.softmax(weight, axis=-1)
        var_encodings = []
        for var, net in zip(variables, self.variable_network):
            var_encodings.append(net(var))
        var_encodings = F.stack(*var_encodings, axis=-1)
        var_encodings = F.sum(F.broadcast_mul(var_encodings, weight), axis=-1)
        return var_encodings, weight


class SelfAttention(HybridBlock):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        n_head: int = 1,
        bias: bool = True,
        share_values: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        super(SelfAttention, self).__init__(**kwargs)
        if d_hidden % n_head != 0:
            raise ValueError(
                f"hidden dim {d_hidden} cannot be split into {n_head} heads."
            )
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_head = d_hidden // n_head
        self.bias = bias
        self.share_values = share_values
        self.temperature = temperature

        with self.name_scope():
            self.dropout = nn.Dropout(dropout)
            self.q_proj = nn.Dense(
                units=self.d_hidden,
                in_units=self.d_hidden,
                use_bias=self.bias,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="q_proj_",
            )
            self.k_proj = nn.Dense(
                units=self.d_hidden,
                in_units=self.d_hidden,
                use_bias=self.bias,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="k_proj_",
            )
            self.v_proj = nn.Dense(
                units=self.d_head if self.share_values else self.d_hidden,
                in_units=self.d_hidden,
                use_bias=self.bias,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="v_proj_",
            )
            self.out_proj = nn.Dense(
                units=self.d_hidden,
                in_units=self.d_hidden,
                use_bias=self.bias,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="out_proj_",
            )

    def _split_head(self, F, x: Tensor) -> Tensor:
        x = F.reshape(data=x, shape=(0, 0, -4, self.n_head, self.d_head))
        x = F.swapaxes(data=x, dim1=1, dim2=2)
        return x

    def _merge_head(self, F, x: Tensor) -> Tensor:
        x = F.swapaxes(data=x, dim1=1, dim2=2)
        x = F.reshape(data=x, shape=(0, 0, self.d_hidden))
        return x

    def _compute_qkv(self, F, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cx = F.slice_axis(x, axis=1, begin=-self.prediction_length, end=None)
        q = self.q_proj(cx)
        k = self.k_proj(x)
        q = self._split_head(F, q)
        k = self._split_head(F, k)
        v = self.v_proj(x)
        if self.share_values:
            v = F.broadcast_like(v.expand_dims(axis=1), k)
        else:
            v = self._split_head(F, v)
        return q, k, v

    def _apply_mask(
        self, F, score: Tensor, key_mask: Optional[Tensor]
    ) -> Tensor:
        k_idx = F.contrib.arange_like(score, axis=-1)
        k_idx = (
            k_idx.expand_dims(axis=0).expand_dims(axis=0).expand_dims(axis=0)
        )
        q_idx = F.contrib.arange_like(score, axis=-2) + self.context_length
        q_idx = (
            q_idx.expand_dims(axis=-1).expand_dims(axis=0).expand_dims(axis=0)
        )
        unidir_mask = F.broadcast_lesser_equal(k_idx, q_idx)
        unidir_mask = F.broadcast_like(unidir_mask, score)
        score = F.where(unidir_mask, score, F.ones_like(score) * -1e9)
        if key_mask is not None:
            key_mask = key_mask.expand_dims(axis=1)  # head
            key_mask = key_mask.expand_dims(axis=2)  # query
            key_mask = F.broadcast_like(key_mask, score)
            score = F.where(key_mask, score, F.ones_like(score) * -1e9)
        return score

    def _compute_attn_score(
        self,
        F,
        q: Tensor,
        k: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        score = F.batch_dot(lhs=q, rhs=k, transpose_b=True)
        score = self._apply_mask(F, score, mask)
        score = score / (math.sqrt(self.d_head) * self.temperature)
        score = F.softmax(score, axis=-1)
        score = self.dropout(score)
        return score

    def _compute_attn_output(self, F, score: Tensor, v: Tensor) -> Tensor:
        v = F.batch_dot(score, v)
        v = self._merge_head(F, v)
        v = self.out_proj(v)
        return v

    def hybrid_forward(self, F, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        q, k, v = self._compute_qkv(F, x)
        score = self._compute_attn_score(F, q, k, mask)
        v = self._compute_attn_output(F, score, v)
        return v


class TemporalFusionEncoder(HybridBlock):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_input: int,
        d_hidden: int,
        **kwargs,
    ) -> None:
        super(TemporalFusionEncoder, self).__init__(**kwargs)
        self.context_length = context_length
        self.prediction_length = prediction_length
        with self.name_scope():
            self.encoder_lstm = rnn.HybridSequentialRNNCell(prefix="encoder_")
            self.encoder_lstm.add(
                rnn.LSTMCell(
                    hidden_size=d_hidden,
                    input_size=d_input,
                )
            )
            self.decoder_lstm = rnn.HybridSequentialRNNCell(prefix="decoder_")
            self.decoder_lstm.add(
                rnn.LSTMCell(
                    hidden_size=d_hidden,
                    input_size=d_input,
                )
            )
            self.gate = nn.HybridSequential()
            self.gate.add(
                nn.Dense(units=d_hidden * 2, in_units=d_hidden, flatten=False)
            )
            self.gate.add(GatedLinearUnit(axis=-1, nonlinear=False))
            if d_input != d_hidden:
                self.skip_proj = nn.Dense(
                    units=d_hidden, in_units=d_input, flatten=False
                )
                self.add_skip = True
            else:
                self.add_skip = False
            self.lnorm = nn.LayerNorm(axis=-1, in_channels=d_hidden)

    def hybrid_forward(
        self,
        F,
        ctx_input: Tensor,
        tgt_input: Tensor,
        states: List[Tensor],
    ) -> Tensor:
        ctx_encodings, states = self.encoder_lstm.unroll(
            length=self.context_length,
            inputs=ctx_input,
            begin_state=states,
            merge_outputs=True,
        )
        tgt_encodings, _ = self.decoder_lstm.unroll(
            length=self.prediction_length,
            inputs=tgt_input,
            begin_state=states,
            merge_outputs=True,
        )
        encodings = F.concat(ctx_encodings, tgt_encodings, dim=1)
        skip = F.concat(ctx_input, tgt_input, dim=1)
        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(F.broadcast_add(skip, encodings))
        return encodings


class TemporalFusionDecoder(HybridBlock):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        d_var: int,
        n_head: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(TemporalFusionDecoder, self).__init__(**kwargs)
        self.context_length = context_length
        self.prediction_length = prediction_length

        with self.name_scope():
            self.enrich = GatedResidualNetwork(
                d_hidden=d_hidden,
                d_static=d_var,
                dropout=dropout,
            )
            self.attention = SelfAttention(
                context_length=context_length,
                prediction_length=prediction_length,
                d_hidden=d_hidden,
                n_head=n_head,
                share_values=True,
                dropout=dropout,
            )
            self.att_net = nn.HybridSequential(prefix="attention_")
            self.att_net.add(nn.Dropout(dropout))
            self.att_net.add(
                nn.Dense(
                    units=d_hidden * 2,
                    in_units=d_hidden,
                    flatten=False,
                    weight_initializer=init.Xavier(),
                )
            )
            self.att_net.add(
                GatedLinearUnit(
                    axis=-1,
                    nonlinear=False,
                )
            )
            self.att_lnorm = nn.LayerNorm(
                axis=-1,
                in_channels=d_hidden,
            )
            self.ff_net = nn.HybridSequential()
            self.ff_net.add(
                GatedResidualNetwork(
                    d_hidden,
                    dropout=dropout,
                )
            )
            self.ff_net.add(
                nn.Dense(
                    units=d_hidden * 2,
                    in_units=d_hidden,
                    flatten=False,
                    weight_initializer=init.Xavier(),
                )
            )
            self.ff_net.add(
                GatedLinearUnit(
                    axis=-1,
                    nonlinear=False,
                )
            )
            self.ff_lnorm = nn.LayerNorm(axis=-1, in_channels=d_hidden)

    def hybrid_forward(
        self, F, x: Tensor, static: Tensor, mask: Tensor
    ) -> Tensor:
        static = F.tile(
            static, reps=(1, self.context_length + self.prediction_length, 1)
        )
        skip = F.slice_axis(x, axis=1, begin=self.context_length, end=None)
        x = self.enrich(x, static)
        mask_pad = F.slice_axis(F.ones_like(mask), axis=1, begin=0, end=1)
        mask_pad = F.tile(mask_pad, reps=(1, self.prediction_length))
        mask = F.concat(mask, mask_pad, dim=1)
        att = self.attention(x, mask)
        att = self.att_net(att)
        x = F.slice_axis(x, axis=1, begin=self.context_length, end=None)
        x = self.att_lnorm(F.broadcast_add(x, att))
        x = self.ff_net(x)
        x = self.ff_lnorm(F.broadcast_add(x, skip))
        return x
