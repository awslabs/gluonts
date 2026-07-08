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

"""Functional-attention token mixers for SMT.

These are drop-in replacements for the self-attention sub-layer of a
Transformer block: each maps a token sequence ``(batch, length, d_model)`` to
one of the same shape. They come from the "functional" view of attention, in
which tokens are treated as samples of an underlying function and mixing is a
low-rank operator between learned function spaces rather than a dense pairwise
affinity (Xu et al., 2026, "Functional Attention"; Katharopoulos et al., 2020,
linear attention; the IntentionNet ridge-regression read-out).

All three pool information *globally* over the sequence, so they are used only
where SMT's attention is bidirectional (the encoder and the recurrent memory
cell). The causal decoder keeps standard softmax attention.
"""

import math

import torch
import torch.nn as nn


class LinearAttentionMixer(nn.Module):
    """Kernelized linear self-attention (Katharopoulos et al., 2020).

    Softmax feature maps on the queries and keys let the ``K^T V``
    summary be formed once and reused for every query, giving
    ``O(length)`` mixing.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dim_head = d_model // nhead
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        heads, dim_head = self.nhead, self.dim_head

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, heads, dim_head).transpose(1, 2)

        q = split(self.to_q(x)).softmax(dim=-1)
        k = split(self.to_k(x)).softmax(dim=-1)
        v = split(self.to_v(x))
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        out = torch.einsum("bhnd,bhde->bhne", q, kv) / math.sqrt(dim_head)
        out = out.transpose(1, 2).reshape(batch, length, heads * dim_head)
        return self.dropout(self.to_out(out))


class IntentionMixer(nn.Module):
    """IntentionNet-style attention: keys and queries share an encoder, the
    values are the tokens themselves, and the read-out is a ridge-regression
    solve (the FuncAttn paper shows functional attention recovers this).

    The solve uses the primal ``d x d`` form (Xu et al., 2026, Eq. 46), which
    via the Woodbury identity is identical to the dual ``n x n`` form but is
    linear -- not cubic -- in the sequence length.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ridge: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dim_head = d_model // nhead
        self.ridge = ridge
        self.enc_k = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        heads, dim_head = self.nhead, self.dim_head

        def split(t: torch.Tensor) -> torch.Tensor:
            return (
                t.view(batch, length, heads, dim_head)
                .transpose(1, 2)
                .reshape(batch * heads, length, dim_head)
            )

        # shared encoder for keys and queries (self-attention: q and k tokens
        # are the same sequence)
        kq = self.enc_k(x)
        key = split(kq)
        query = split(kq)
        value = split(self.to_v(x))

        # Q (K^T K + lambda I_d)^-1 K^T V: the "primal" ridge solve over the
        # d x d Gram matrix (Xu et al., 2026, Eq. 46). Via the Woodbury identity
        # this is identical to the dual Q K^T (K K^T + lambda I_n)^-1 V but only
        # ever inverts a d x d matrix, so the cost is linear -- not cubic -- in
        # the sequence length n.
        key_t = key.transpose(1, 2)
        gram = torch.bmm(key_t, key)
        key_value = torch.bmm(key_t, value)
        identity = torch.eye(dim_head, device=x.device, dtype=x.dtype)
        operator = torch.linalg.solve(gram + self.ridge * identity, key_value)
        out = torch.bmm(query, operator)

        out = (
            out.view(batch, heads, length, dim_head)
            .transpose(1, 2)
            .reshape(batch, length, heads * dim_head)
        )
        return self.dropout(self.to_out(out))


class FuncAttnMixer(nn.Module):
    """Functional Attention (Xu et al., 2026).

    Tokens are softly assigned to a small set of adaptive basis
    "slices"; the optimal linear operator between the slice bases is
    estimated with a ridge least-squares solve, and the result is de-
    sliced back to tokens. Cost scales with the number of slices, not
    the (squared) sequence length.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_slices: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dim_head = d_model // nhead
        self.num_slices = num_slices
        self.in_x = nn.Linear(d_model, d_model)
        self.in_fx = nn.Linear(d_model, d_model)
        self.slice = nn.Linear(self.dim_head, num_slices)
        nn.init.orthogonal_(self.slice.weight)
        self.temperature = nn.Parameter(torch.ones(1, nhead, 1, 1) * 0.5)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        heads, dim_head = self.nhead, self.dim_head
        num_slices = min(self.num_slices, length)

        def heads_first(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, heads, dim_head).permute(0, 2, 1, 3)

        fx = heads_first(self.in_fx(x))
        xm = heads_first(self.in_x(x))

        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        logits = self.slice(xm)[..., :num_slices]
        weights = torch.softmax(logits / temp, dim=-1)  # (b, h, n, g)
        norm = weights.sum(dim=2)  # (b, h, g)
        tokens = torch.einsum("bhnd,bhng->bhgd", fx, weights)
        tokens = tokens / (norm[..., None] + 1e-5)

        q = self.to_q(tokens)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        k_t = k.transpose(-1, -2)
        gram = torch.matmul(k_t, k)  # (b, h, dim_head, dim_head)
        alpha = torch.sigmoid(self.alpha)
        identity = torch.eye(dim_head, device=x.device, dtype=x.dtype)
        reg = (1 - alpha) * gram + alpha * identity
        z = torch.linalg.solve(reg, k_t)
        operator = torch.matmul(q, z)  # (b, h, g, g)
        out_tokens = torch.matmul(operator, v)

        out = torch.einsum("bhgd,bhng->bhnd", out_tokens, weights)
        out = out.permute(0, 2, 1, 3).reshape(batch, length, heads * dim_head)
        return self.dropout(self.to_out(out))


def make_mixer(
    attn_type: str,
    d_model: int,
    nhead: int,
    num_slices: int,
    dropout: float,
) -> nn.Module:
    if attn_type == "funcattn":
        return FuncAttnMixer(d_model, nhead, num_slices, dropout)
    if attn_type == "intention":
        return IntentionMixer(d_model, nhead, dropout=dropout)
    if attn_type == "linear":
        return LinearAttentionMixer(d_model, nhead, dropout)
    raise ValueError(f"unknown functional attn_type: {attn_type}")


class FunctionalEncoderLayer(nn.Module):
    """Pre-norm Transformer block whose token mixer is a functional-attention
    module, mirroring ``nn.TransformerEncoderLayer(norm_first=True)``."""

    def __init__(
        self,
        mixer: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mixer = mixer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class FunctionalEncoder(nn.Module):
    """A stack of ``FunctionalEncoderLayer`` blocks."""

    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def make_functional_encoder(
    attn_type: str,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    num_slices: int,
) -> FunctionalEncoder:
    layers = [
        FunctionalEncoderLayer(
            make_mixer(attn_type, d_model, nhead, num_slices, dropout),
            d_model,
            dim_feedforward,
            dropout,
        )
        for _ in range(num_layers)
    ]
    return FunctionalEncoder(layers)
