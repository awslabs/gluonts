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

from typing import Dict, Optional, Tuple

import mxnet as mx
from mxnet.gluon import HybridBlock

from gluonts.mx import Tensor


def split_heads(F, x: Tensor, dim_per_head: int, heads: int) -> Tensor:
    r"""
    Returns a tensor with head dimension folded into batch and last dimension divided by the number of heads.

    Parameters
    ----------
    x
        Tensor of shape (batch_size, time_length, dim).
    dim_per_head
        Dimension per head
    heads
        Number of heads

    Returns
    -------
    Tensor of shape (batch_size * heads, time_length, dim_per_head).
    """

    # (batch_size, time_length, heads, dim_per_head)
    x = F.reshape(data=x, shape=(0, -1, heads, dim_per_head))
    # (batch_size, heads, time_length, dim/heads)
    x = F.transpose(data=x, axes=(0, 2, 1, 3))
    # (batch_size * heads, time_length, dim/heads)
    return F.reshape(data=x, shape=(-3, -1, dim_per_head))


def dot_attention(
    F,
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
) -> Tensor:
    r"""

    Parameters
    ----------
    queries
        Attention queries of shape (n, lq, d)
    keys
        Attention keys of shape (n, lk, d)
    values
        Attention values of shape (n, lk, dv)
    mask
        Optional mask tensor
    dropout
        Dropout rate

    Returns
    -------
    'Context' vectors for each query of shape (n, lq, dv)
    """

    # (n, lq, lk)
    logits = F.batch_dot(lhs=queries, rhs=keys, transpose_b=True)

    if mask is not None:
        logits = F.broadcast_add(logits, mask)

    probs = F.softmax(logits, axis=-1)
    probs = F.Dropout(probs, p=dropout) if dropout > 0.0 else probs

    # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
    return F.batch_dot(lhs=probs, rhs=values)


def combine_heads(F, x: Tensor, dim_per_head: int, heads: int) -> Tensor:
    r"""

    Parameters
    ----------
    x
        Tensor of shape (batch_size * heads, time_length, dim_per_head)
    dim_per_head
        Dimension per head
    heads
        Number of heads

    Returns
    -------
    Tensor of shape (batch_size, time_length, dim)
    """

    # (batch_size, heads, time_length, dim_per_head)
    x = F.reshape(data=x, shape=(-4, -1, heads, 0, dim_per_head))
    # (batch_size, time_length, heads, dim_per_head)
    x = F.transpose(x, axes=(0, 2, 1, 3))
    # (batch_size, time_length, dim)
    return F.reshape(x, shape=(-1, 0, dim_per_head * heads))


class LayerNormalization(HybridBlock):
    """
    Implements layer normalization as proposed in [BKH16]_.
    """

    def __init__(
        self,
        scale_init: str = "ones",
        shift_init: str = "zeros",
        eps: float = 1e-06,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.scale_init = scale_init
        self.shift_init = shift_init

        with self.name_scope():
            self.lnorm = mx.gluon.nn.LayerNorm(
                axis=-1,
                gamma_initializer=self.scale_init,
                beta_initializer=self.shift_init,
                epsilon=eps,
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, data: Tensor) -> Tensor:
        r"""
        Normalizes hidden units of data as follows:

        data = scale * (data - mean) / sqrt(var + eps) + shift

        Normalization is performed over the last dimension of the input data.

        Parameters
        ----------
        data
            Data to normalize of shape (d0, ..., dn, num_hidden)

        Returns
        -------
        Normalized inputs of shape: (d0, ..., dn, num_hidden)
        """

        return self.lnorm(data)


class InputLayer(HybridBlock):
    r"""
    Transforms the input vector to model_size with an one-layer MPL, i.e.,
    (batch_size, time_length, input_dim) -> (batch_size, time_length, model_size)
    """

    def __init__(self, model_size: int = 64, **kwargs) -> None:

        super().__init__(**kwargs)

        self.model_size = model_size
        with self.name_scope():
            self.net = mx.gluon.nn.Dense(units=self.model_size, flatten=False)

    def hybrid_forward(self, F, data: Tensor, *args):
        return self.net(data)


class MultiHeadAttentionBase(HybridBlock):
    """
    Base class for Multi-head attention.

    Parameters
    ----------
    att_dim_in
        Attention dimension (number of hidden units)
    heads
        Number of attention heads
    att_dim_out
        Output dimension (number of output units)
    dropout
        Dropout rate on attention scores
    """

    def __init__(
        self,
        att_dim_in: int = 32,
        heads: int = 8,
        att_dim_out: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        assert (
            att_dim_in % heads == 0
        ), "Number of heads {} must divide attention att_dim_in {}".format(
            heads, att_dim_in
        )

        self.att_dim_in = att_dim_in
        self.heads = heads
        self.att_dim_out = att_dim_out
        self.dropout = dropout
        self.dim_per_head = self.att_dim_in // self.heads

        with self.name_scope():
            self.dense_att = mx.gluon.nn.Dense(
                units=self.att_dim_out, flatten=False
            )

    def _attend(
        self,
        F,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Returns context vectors of multi-head dot attention.

        Parameters
        ----------
        queries
            Queries tensor of shape (batch_size, query_max_length, dim)
        keys
            Keys tensor of shape (batch_size, memory_max_length, dim)
        values
            Values tensor of shape (batch_size, memory_max_length, dim)
        mask

        Returns
        -------
        Context vectors of shape (batch_size, query_max_length, att_dim_out)
        """

        # scale by 1/sqrt(dim_per_head)
        queries = queries * (self.dim_per_head ** -0.5)

        # (batch_size * heads, length, dim/heads)
        queries = split_heads(F, queries, self.dim_per_head, self.heads)
        keys = split_heads(F, keys, self.dim_per_head, self.heads)
        values = split_heads(F, values, self.dim_per_head, self.heads)

        # (batch_size * heads, query_max_length, dim_per_head)
        contexts = dot_attention(
            F, queries, keys, values, mask=mask, dropout=self.dropout
        )

        # (batch_size, query_max_length, input_dim)
        contexts = combine_heads(F, contexts, self.dim_per_head, self.heads)

        # contexts: (batch_size, query_max_length, output_dim)
        contexts = self.dense_att(contexts)

        return contexts


class MultiHeadSelfAttention(MultiHeadAttentionBase):
    r"""
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    Parameters
    ----------
    att_dim_in
        Attention dimension (number of hidden units)
    heads
        Number of attention heads
    att_dim_out
        Output dimension (number of output units)
    dropout
        Dropout rate on attention scores
    """

    def __init__(
        self,
        att_dim_in: int = 32,
        heads: int = 8,
        att_dim_out: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(att_dim_in, heads, att_dim_out, dropout, **kwargs)

        with self.name_scope():
            self.dense_pre_satt = mx.gluon.nn.Dense(
                units=self.att_dim_in * 3, flatten=False
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[Dict[str, Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Dict]]:
        r"""
        Computes multi-head attention on a set of inputs, serving as queries,
        keys, and values. If sequence lengths are provided, they will be used
        to mask the attention scores. May also use a cache of previously
        computed inputs.

        Parameters
        ----------
        inputs
            Input data of shape (batch_size, max_length, att_dim_in)
        mask
            Optional tensor to mask attention scores
        cache
            Optional dictionary of previously computed keys and values

        Returns
        -------
        Tensor
            A tensor of shape (batch_size, max_length, att_dim_out)
        """

        # Q = K = V -> Q * W_q, K * W_k, V * W_v

        # combined: (batch_size, max_length, att_dim_in * 3)
        combined = self.dense_pre_satt(inputs)

        # split into queries, keys and values
        # (batch_size, max_length, att_dim_in)
        queries, keys, values = F.split(data=combined, num_outputs=3, axis=2)

        if cache is not None:
            # append new keys and values to cache, update the cache
            keys = cache["k"] = (
                keys
                if "k" not in cache.keys()
                else F.concat(cache["k"], keys, dim=1)
            )
            values = cache["v"] = (
                values
                if "v" not in cache.keys()
                else F.concat(cache["v"], values, dim=1)
            )

        return self._attend(F, queries, keys, values, mask), cache


class MultiHeadAttention(MultiHeadAttentionBase):
    r"""
    Multi-head attention layer for queries independent from keys/values.

    Parameters
    ----------
    att_dim_in
        Attention dimension (number of hidden units)
    heads
        Number of attention heads
    att_dim_out
        Output dimension (number of output units)
    dropout
        Dropout rate on attention scores
    """

    def __init__(
        self,
        att_dim_in: int = 32,
        heads: int = 8,
        att_dim_out: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(att_dim_in, heads, att_dim_out, dropout, **kwargs)

        with self.name_scope():
            self.dense_pre_att_q = mx.gluon.nn.Dense(
                units=self.att_dim_in, flatten=False
            )
            self.dense_pre_att_k = mx.gluon.nn.Dense(
                units=self.att_dim_in, flatten=False
            )
            self.dense_pre_att_v = mx.gluon.nn.Dense(
                units=self.att_dim_in, flatten=False
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, queries: Tensor, memory: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A mask tensor may also be used to mask the attention scores.
        Returns a tensor of shape (batch_size, max_length, att_dim_out).

        Parameters
        ----------
        queries
            Queries tensor of shape (batch_size, query_max_length, att_dim_in)
        memory
            Memory tensor to attend to of shape (batch_size, memory_max_length, att_dim_in)
        mask
            Optional tensor to mask attention scores

        Returns
        -------
        Tensor of shape (batch_size, query_seq_len, att_dim_out)
        """

        # Q -> Q * W_q
        # K = V -> K * W_k, V * W_v

        # (batch, query_max_length, att_dim_in)
        queries = self.dense_pre_att_q(queries)

        # (batch, memory_max_length, att_dim_in)
        keys = self.dense_pre_att_k(memory)

        # (batch, memory_max_length, att_dim_in)
        values = self.dense_pre_att_v(memory)

        return self._attend(F, queries, keys, values, mask=mask)


class TransformerFeedForward(HybridBlock):
    r"""
    Position-wise feed-forward network with activation.

    .. math::

        activation(XW_1 + b_1)W_2 + b_2

    :math:`W_1`: (batch_size, d, inner_dim)
    :math:`W_2`: (batch_size, inner_dim, out_dim)
    """

    def __init__(
        self,
        inner_dim: int = 32,  # W1: (batch_size, d, inner_dim)
        out_dim: int = 32,  # W2: (batch_size, inner_dim, out_dim)
        act_type: str = "softrelu",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act_type = act_type

        with self.name_scope():
            self.mlp = mx.gluon.nn.HybridSequential()
            self.mlp.add(
                mx.gluon.nn.Dense(
                    units=self.inner_dim,
                    use_bias=True,
                    activation=self.act_type,
                    flatten=False,
                )
            )
            if self.dropout > 0.0:
                self.mlp.add(mx.gluon.nn.Dropout(self.dropout))
            self.mlp.add(
                mx.gluon.nn.Dense(units=out_dim, use_bias=True, flatten=False)
            )  # no activation

    def hybrid_forward(self, F, x: Tensor, *args) -> Tensor:
        r"""
        Position-wise feed-forward network with activation.

        Parameters
        ----------
        x
            Tensor of shape (batch_size, d, in_dim)

        Returns
        -------
        Tensor of shape (batch_size, d1, out_dim)
        """

        return self.mlp(x)


class TransformerProcessBlock(HybridBlock):
    r"""
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self, sequence: str, dropout: float, **kwargs) -> None:

        super().__init__(**kwargs)

        self.sequence = sequence
        self.dropout = dropout
        self.layer_norm = None
        if "n" in sequence:
            self.layer_norm = LayerNormalization()

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, data: Tensor, prev: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Apply processing sequence to data with optional previous input.

        Parameters
        ----------
        data
            Input data of shape: (batch_size, length, num_hidden)
        prev
            Previous data of shape (batch_size, length, num_hidden)

        Returns
        -------
        Processed data of shape (batch_size, length, num_hidden).
        """
        if not self.sequence:
            return data

        if prev is None:
            assert (
                "r" not in self.sequence
            ), "Residual connection not allowed if no previous value given."

        for step in self.sequence:

            if step == "r":
                data = F.broadcast_add(data, prev)

            elif step == "n":
                assert self.layer_norm is not None
                data = self.layer_norm(data)

            elif step == "d":
                if self.dropout > 0.0:
                    data = F.Dropout(data, p=self.dropout)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        return data
