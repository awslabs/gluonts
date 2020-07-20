from typing import Optional, Tuple, List
import math

import mxnet as mx
from mxnet import init
from mxnet.gluon import nn, Parameter, HybridBlock
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.mx.block.feature import FeatureEmbedder


def _torch_gather(F, data: Tensor, idx: Tensor, axis: int):
    """
    Pytorch-style gather_nd
    """
    if axis < 0:
        axis = data.ndim + axis
    mx_idx = []
    for dim in range(data.ndim):
        if dim == axis:
            d_idx = F.broadcast_like(idx, data)
        else:
            d_idx = F.contrib.arange_like(data, axis=dim)
            for _ in range(dim):
                d_idx = F.expand_dims(data=d_idx, axis=0)
            for _ in range(data.ndim - dim - 1):
                d_idx = F.expand_dims(data=d_idx, axis=-1)
            d_idx = F.broadcast_like(d_idx, data)
        mx_idx.append(d_idx)
    mx_idx = F.stack(*mx_idx, axis=0)
    return F.gather_nd(data, mx_idx)


class SinusoidalPositionalEmbedding(HybridBlock):
    @validated()
    def __init__(self, d_embed: int, **kwargs):
        super(SinusoidalPositionalEmbedding, self).__init__(**kwargs)
        if d_embed % 2 != 0:
            raise ValueError(
                "sinusoidal embedding must have an even dimension"
            )
        self.d_embed = d_embed

    def hybrid_forward(self, F, pos_seq: Tensor) -> Tensor:
        inv_freq = F.arange(0, self.d_embed, 2)
        inv_freq = F.exp((inv_freq / self.d_embed) * -math.log(1e4))
        pos_seq = F.reshape(data=pos_seq, shape=(0, 0, 1))
        pos_seq = F.broadcast_mul(pos_seq, inv_freq)
        return F.concat(F.sin(pos_seq), F.cos(pos_seq), dim=-1)


class Attention(HybridBlock):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        n_head: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        super(Attention, self).__init__(**kwargs)
        if d_hidden % n_head != 0:
            raise ValueError(
                f"hidden dim {d_hidden} cannot be split into {n_head} heads."
            )
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_head = d_hidden // n_head
        self.bias = bias
        self.temperature = temperature

        with self.name_scope():
            self.dropout = nn.Dropout(dropout)

    def _split_head(self, F, x: Tensor) -> Tensor:
        """
        Split hidden state into multi-heads
        
        Args
        ----------
            x : Tensor [batch, length, d_hidden]
        
        Returns
        -------
            Tensor [batch, n_head, length, d_head]
        """
        x = F.reshape(data=x, shape=(0, 0, -4, self.n_head, self.d_head))
        x = F.swapaxes(data=x, dim1=1, dim2=2)
        return x

    def _merge_head(self, F, x: Tensor) -> Tensor:
        """
        Merge multi-heads into one hidden state
        
        Args
        ----------
            x : Tensor [batch, n_head, length, d_head]
        
        Returns
        -------
            Tensor [batch, length, d_hidden]
        """
        x = F.swapaxes(data=x, dim1=1, dim2=2)
        x = F.reshape(data=x, shape=(0, 0, self.d_hidden))
        return x

    def _compute_qkv(self, F, *args, **kwargs):
        raise NotImplementedError

    def _compute_attn_score(self, F, q: Tensor, k: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def _compute_attn_output(
        self, F, score: Tensor, v: Tensor, **kwargs
    ) -> Tensor:
        raise NotImplementedError


class GroupSelfAttention(Attention):
    """
    Self-attention module with q,k,v from the same input

    Parameters
    ----------
    d_hidden : int
        hidden dimension
    n_groups: int
        number of groups in given shape repr
    n_head : int, optional
        number of attention heads, by default 1
    bias : bool, optional
        add bias term in input and output projections, by default True
    bidirectional : bool, optional
        if False, add a mask to avoid backward attention, by default False
    dist_encoding : Optional[str], optional
        add relative distance embeddings to dot-product attention, can be 
            'add' (linearly combine key and dist),
            'dot' (dot product between key and dist), 
            or None (disabled),
        by default None
    share_values : bool, optional
        if True, a value reprensentation is shared by all attention heads, by default False
        ref. https://arxiv.org/abs/1912.09363
    dropout : float, optional
        dropout rate, by default 0.0
    temperature : float, optional
        softmax temperature, by default 1.0
    """

    @validated()
    def __init__(
        self,
        d_hidden: int,
        n_groups: int,
        n_head: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        dist_enc: Optional[str] = None,
        share_values: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        try:
            assert d_hidden % n_groups == 0
            assert n_head % n_groups == 0
        except AssertionError:
            raise ValueError(
                "Both d_hidden and n_head must be divisible by n_groups"
            )
        super(GroupSelfAttention, self).__init__(
            d_hidden=d_hidden,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            temperature=temperature,
            **kwargs,
        )
        self.n_groups = n_groups
        self.bidirectional = bidirectional
        self.dist_enc = dist_enc
        self.share_values = share_values

        with self.name_scope():
            self.qk_proj = nn.Conv1D(
                channels=d_hidden * 2,
                kernel_size=1,
                groups=n_groups,
                use_bias=bias,
                weight_initializer=init.Xavier(),
                prefix="qk_proj_",
            )
            self.v_proj = nn.Dense(
                units=self.d_head if self.share_values else d_hidden,
                use_bias=bias,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="v_proj_",
            )
            self.out_proj = nn.Dense(
                units=d_hidden,
                use_bias=bias,
                flatten=False,
                weight_initializer=init.Xavier(),
                prefix="out_proj_",
            )

            if self.dist_enc is not None:
                assert self.dist_enc in [
                    "dot",
                    "add",
                ], f"distance encoding type {self.dist_enc} is not supported"
                self.posemb = SinusoidalPositionalEmbedding(d_hidden)
                self.pos_proj = nn.Dense(
                    units=d_hidden,
                    use_bias=bias,
                    flatten=False,
                    weight_initializer=init.Xavier(),
                    prefix="pos_proj_",
                )
                if self.dist_enc == "add":
                    self._ctt_bias_weight = Parameter(
                        "_ctt_bias_weight",
                        shape=(1, n_head, 1, self.d_head),
                        init=init.Xavier(),
                    )
                    self._pos_bias_weight = Parameter(
                        "_pos_bias_weight",
                        shape=(1, n_head, 1, self.d_head),
                        init=init.Xavier(),
                    )

    def _compute_qkv(
        self, F, value: Tensor, shape: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        qk = F.swapaxes(shape, dim1=1, dim2=2)
        qk = self.qk_proj(qk)
        qk = F.swapaxes(qk, dim1=1, dim2=2)
        qk = F.split(qk, num_outputs=2 * self.n_groups, axis=-1)
        q = F.concat(*qk[0::2], dim=-1)
        k = F.concat(*qk[1::2], dim=-1)
        q = self._split_head(F, q)
        k = self._split_head(F, k)
        v = self.v_proj(value)
        if self.share_values:
            v = F.broadcast_like(v.expand_dims(axis=1), k)
        else:
            v = self._split_head(F, v)
        return q, k, v

    def _apply_mask(
        self, F, score: Tensor, key_mask: Optional[Tensor]
    ) -> Tensor:
        if not self.bidirectional:
            k_idx = F.contrib.arange_like(score, axis=-1).reshape(1, 1, 1, -1)
            q_idx = F.contrib.arange_like(score, axis=-2).reshape(1, 1, -1, 1)
            unidir_mask = F.broadcast_greater(k_idx, q_idx)
            unidir_mask = F.broadcast_like(unidir_mask, score)
            score = F.where(
                unidir_mask, F.ones_like(score) * float("-inf"), score
            )
        if key_mask is not None:
            mem_mask = key_mask.squeeze(axis=-1)
            mem_mask = mem_mask.expand_dims(axis=1)  # head
            mem_mask = mem_mask.expand_dims(axis=2)  # query
            mem_mask = F.broadcast_like(mem_mask, score)
            score = F.where(
                mem_mask, F.ones_like(score) * float("-inf"), score
            )
        return score

    def _compute_attn_score(
        self,
        F,
        q: Tensor,
        k: Tensor,
        mask: Optional[Tensor],
        _ctt_bias_weight: Optional[Tensor],
        _pos_bias_weight: Optional[Tensor],
    ) -> Tensor:
        score = F.batch_dot(lhs=q, rhs=k, transpose_b=True)
        if self.dist_enc is not None:
            # score_{ij} = <q_i, k_j> + s_{ij}
            # idx.shape = [klen, klen]
            # idx[i][j] = i-j
            idx = F.contrib.arange_like(k, axis=2)
            idx = F.broadcast_sub(
                idx.expand_dims(axis=1), idx.expand_dims(axis=0)
            )
            # idx[i][j] = |i-j|
            idx = idx.abs()
            # idx.shape = [1, 1, klen, klen]
            idx = idx.expand_dims(axis=0).expand_dims(axis=0)
            # dist representation r for attention
            # r.shape = [1, klen, d_hidden]
            r = F.contrib.arange_like(k, axis=2).expand_dims(axis=0)
            r = self.posemb(r)
            r = self.pos_proj(r)
            # r.shape = [1, n_head, klen, d_head]
            r = self._split_head(F, r)
            # r.shape = [batch, n_head, klen, d_head]
            r = r.broadcast_like(k)
            if self.dist_enc == "add":
                # transformer-xl style: https://arxiv.org/abs/1901.02860
                # s_{ij} = <q_i, r_{|i-j|}> + <u, k_j> + <v, r_{|i-j|}>
                #      u = _content_bias_weight
                #      v = _position_bias_weight

                # qr_{ij} = <q_i, r_j>
                # qr'_{ij} = qr_{i,idx[i][j]} = qr_{i,|i-j|}
                qr = F.batch_dot(lhs=q, rhs=r, transpose_b=True)
                qr = _torch_gather(F, data=qr, idx=idx, axis=-1)
                # rk_{ij} = <v, r_i> + <u, k_j>
                # rk'_{ij} = rk_{idx[i][j], j} = rk_{|i-j|, j}
                u = F.broadcast_to(_ctt_bias_weight, k)
                v = F.broadcast_to(_pos_bias_weight, r)
                rk = F.batch_dot(u, k, transpose_b=True) + F.batch_dot(
                    v, r, transpose_b=True
                )
                rk = _torch_gather(F, data=rk, idx=idx, axis=-2)
                # s_{ij} = qr_{i,|i-j|} + rk_{|i-j|, j}
                s = qr + rk
            else:
                # s_{ij} = <r_{|i-j|}, (q_i+k_j)>
                #        = <q_i, r_{|i-j|}> + <r_{|i-j|}, k_j>

                # qr_{ij} = <q_i, r_j>
                # qr'_{ij} = qr_{i,idx[i][j]} = qr_{i,|i-j|}
                qr = F.batch_dot(lhs=q, rhs=r, transpose_b=True)
                qr = _torch_gather(F, data=qr, idx=idx, axis=-1)
                # rk_{ij} = <r_i, k_j>
                # rk'_{ij} = rk_{idx[i][j], j} = rk_{|i-j|, j}
                rk = F.batch_dot(lhs=r, rhs=k, transpose_b=True)
                rk = _torch_gather(F, data=rk, idx=idx, axis=-2)
                # s_{ij} = qr_{i,|i-j|} + rk_{|i-j|,j}
                s = qr + rk
            # add relative positional bias to content-based attention score
            score = score + s
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

    def hybrid_forward(
        self,
        F,
        value: Tensor,
        shape: Tensor,
        mask: Tensor,
        _ctt_bias_weight: Optional[Tensor] = None,
        _pos_bias_weight: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self._compute_qkv(F, value, shape)
        score = self._compute_attn_score(
            F, q, k, mask, _ctt_bias_weight, _pos_bias_weight
        )
        v = self._compute_attn_output(F, score, v)
        return v


class DualSelfAttention(GroupSelfAttention):
    @validated()
    def __init__(self, **kwargs):
        super(DualSelfAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.pat_proj = nn.Conv1D(
                channels=self.d_hidden,
                kernel_size=1,
                groups=self.n_groups,
                use_bias=self.bias,
                weight_initializer=init.Xavier(),
                prefix="pat_proj_",
            )

    def _compute_attn_output(
        self, F, score: Tensor, v: Tensor, k: Tensor
    ) -> Tuple[Tensor, Tensor]:
        v = super(DualSelfAttention, self)._compute_attn_output(F, score, v)
        s = F.batch_dot(score, k)
        s = self._merge_head(F, s)
        s = F.swapaxes(s, dim1=1, dim2=2)
        s = self.pat_proj(s)
        s = F.swapaxes(s, dim1=1, dim2=2)
        return v, s

    def hybrid_forward(
        self,
        F,
        value: Tensor,
        shape: Tensor,
        mask: Optional[Tensor],
        _ctt_bias_weight: Optional[Tensor] = None,
        _pos_bias_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        q, k, v = self._compute_qkv(F, value, shape)
        score = self._compute_attn_score(
            F, q, k, mask, _ctt_bias_weight, _pos_bias_weight
        )
        v, s = self._compute_attn_output(F, score, v, k)
        return v, s


class PosFFN(HybridBlock):
    @validated()
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        activation: str = "softrelu",
        pre_ln: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(PosFFN, self).__init__(**kwargs)
        self.pre_ln = pre_ln
        with self.name_scope():
            self.linear1 = nn.Dense(
                units=d_hidden,
                use_bias=True,
                flatten=False,
                activation=activation,
                weight_initializer=init.Xavier(),
            )
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Dense(
                units=d_model,
                use_bias=True,
                flatten=False,
                weight_initializer=init.Xavier(),
            )
            self.lnorm = nn.LayerNorm(axis=-1)

    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        if self.pre_ln:
            y = self.lnorm(x)
        else:
            y = x
        y = self.linear1(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = y + x
        if not self.pre_ln:
            y = self.lnorm(y)
        return y


class CausalConv1D(HybridBlock):
    @validated()
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: str = "tanh",
        **kwargs,
    ):
        super(CausalConv1D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.channels = channels
        with self.name_scope():
            self.net = nn.Conv1D(
                channels,
                kernel_size,
                use_bias=False,
                activation="tanh",
                weight_initializer=init.Xavier(),
            )

    def hybrid_forward(self, F, x: Tensor, *args) -> Tensor:
        pad = (
            F.zeros_like(x)
            .slice_axis(axis=1, begin=0, end=1)
            .tile(reps=(1, self.kernel_size - 1, 1))
        )
        x = F.concat(pad, x, dim=1)
        x = F.swapaxes(x, dim1=1, dim2=2)
        x = self.net(x)
        x = F.swapaxes(x, dim1=1, dim2=2)
        return x
