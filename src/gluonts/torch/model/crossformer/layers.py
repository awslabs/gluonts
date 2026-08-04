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

from math import ceil

import torch
from torch import nn


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSWEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SegMerging(nn.Module):
    def __init__(self, d_model: int, win_size: int) -> None:
        super().__init__()
        self.win_size = win_size
        self.norm = nn.LayerNorm(win_size * d_model)
        self.proj = nn.Linear(win_size * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, target_dim, seg_num, d_model = x.shape
        remainder = seg_num % self.win_size
        if remainder:
            pad_num = self.win_size - remainder
            x = torch.cat((x, x[:, :, -1:, :].expand(-1, -1, pad_num, -1)), dim=2)
            seg_num += pad_num

        x = x.reshape(
            batch_size,
            target_dim,
            seg_num // self.win_size,
            self.win_size * d_model,
        )
        return self.proj(self.norm(x))


class TwoStageAttentionLayer(nn.Module):
    def __init__(
        self,
        seg_num: int,
        factor: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seg_num = seg_num
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dim_sender = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dim_receiver = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn_time = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.ffn_dim = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, target_dim, seg_num, d_model = x.shape

        time_in = x.reshape(batch_size * target_dim, seg_num, d_model)
        time_enc, _ = self.time_attention(time_in, time_in, time_in)
        time_out = self.norm1(time_in + self.dropout(time_enc))
        time_out = self.norm2(time_out + self.ffn_time(time_out))

        dim_in = time_out.reshape(batch_size, target_dim, seg_num, d_model)
        dim_in = dim_in.permute(0, 2, 1, 3).reshape(batch_size * seg_num, target_dim, d_model)

        routers = self.router.unsqueeze(0).expand(batch_size, -1, -1, -1)
        routers = routers.reshape(batch_size * seg_num, self.router.shape[1], d_model)

        dim_buffer, _ = self.dim_sender(routers, dim_in, dim_in)
        dim_enc, _ = self.dim_receiver(dim_in, dim_buffer, dim_buffer)
        dim_out = self.norm3(dim_in + self.dropout(dim_enc))
        dim_out = self.norm4(dim_out + self.ffn_dim(dim_out))

        return (
            dim_out.reshape(batch_size, seg_num, target_dim, d_model)
            .permute(0, 2, 1, 3)
            .contiguous()
        )


class ScaleBlock(nn.Module):
    def __init__(
        self,
        win_size: int,
        seg_num: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        self.merge = SegMerging(d_model=d_model, win_size=win_size) if win_size > 1 else None
        self.layers = nn.ModuleList(
            [
                TwoStageAttentionLayer(
                    seg_num=seg_num,
                    factor=factor,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge is not None:
            x = self.merge(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        win_size: int,
        in_seg_num: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        block_depth: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ScaleBlock(
                    win_size=1 if i == 0 else win_size,
                    seg_num=ceil(in_seg_num / (win_size**i)),
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    depth=block_depth,
                    dropout=dropout,
                    factor=factor,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = [x]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs


class DecoderLayer(nn.Module):
    def __init__(
        self,
        seg_len: int,
        out_seg_num: int,
        d_model: int,
        latent_dim: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.self_attention = TwoStageAttentionLayer(
            seg_num=out_seg_num,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.latent_proj = nn.Linear(d_model, seg_len * latent_dim)

    def forward(
        self, x: torch.Tensor, cross: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, target_dim, out_seg_num, d_model = x.shape
        x = self.self_attention(x)

        query = x.reshape(batch_size * target_dim, out_seg_num, d_model)
        cross = cross.reshape(batch_size * target_dim, cross.shape[2], d_model)
        attn_out, _ = self.cross_attention(query, cross, cross)
        query = self.norm1(query + self.dropout(attn_out))
        query = self.norm2(query + self.ffn(query))

        x = query.reshape(batch_size, target_dim, out_seg_num, d_model)
        latent = self.latent_proj(x)
        latent = latent.reshape(
            batch_size,
            target_dim,
            out_seg_num,
            self.seg_len,
            -1,
        )
        latent = latent.permute(0, 2, 3, 1, 4).reshape(
            batch_size,
            out_seg_num * self.seg_len,
            target_dim,
            -1,
        )
        return x, latent


class Decoder(nn.Module):
    def __init__(
        self,
        seg_len: int,
        out_seg_num: int,
        num_layers: int,
        d_model: int,
        latent_dim: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    seg_len=seg_len,
                    out_seg_num=out_seg_num,
                    d_model=d_model,
                    latent_dim=latent_dim,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    factor=factor,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cross: list[torch.Tensor]) -> torch.Tensor:
        latent = None
        for layer, cross_state in zip(self.layers, cross):
            x, layer_latent = layer(x, cross_state)
            latent = layer_latent if latent is None else latent + layer_latent
        assert latent is not None
        return latent
