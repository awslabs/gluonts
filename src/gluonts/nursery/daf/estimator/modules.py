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


from typing import Optional, List, Tuple, Union, Iterator, Callable
from itertools import product, chain
from copy import deepcopy
import warnings
import math

import torch as pt
from torch import Tensor, LongTensor, BoolTensor
from torch import nn
from torch.nn import Parameter, functional as F

from tslib.nn.activations import PositiveSoftplus
from ..network import (
    AttentionKernel,
    ExpKernel,
    RBFKernel,
    EncoderModule,
    DecoderModule,
    AttentionBlock,
    AdversarialBlock,
    SimpleDiscriminator,
)
from .hooks import (
    ZScoreNormalizer,
    LossFunction,
    MSELoss,
    MAELoss,
)

MIN_TRADEOFF = 0.5


class AttentionEstimator(nn.Module):
    def __init__(
        self,
        *lists_of_modules: List[nn.Module],
        horizon: Union[int, float],
        loss_func: LossFunction = MAELoss(),
        normalize_input: bool = True,
        rescale_output: bool = True,
        layerwise_loss: bool = True,
    ) -> None:
        super(AttentionEstimator, self).__init__()
        self._create_modules(*lists_of_modules)

        if isinstance(horizon, int):
            assert horizon >= 0, "Positive horizon is required."
        else:
            assert 0 <= horizon <= 1, "horizon must be from [0,1]."
        self._softplus = PositiveSoftplus(margin=MIN_TRADEOFF)
        self._tradeoff = Parameter(
            pt.tensor(math.log(math.exp(1 - MIN_TRADEOFF) - 1) + MIN_TRADEOFF),
            requires_grad=False,
        )
        self.horizon = horizon
        self.normalize_input = normalize_input
        self.rescale_output = rescale_output
        self.layerwise_loss = layerwise_loss

        if self.normalize_input:
            self._normalizer = ZScoreNormalizer(
                rescale_loss=self.rescale_output
            )
            self.register_forward_pre_hook(self._normalizer.forward_pre_hook)
            self.register_forward_hook(self._normalizer.forward_hook)
        else:
            self._normalizer = None
        self.register_loss_func(loss_func)

        self.register_buffer("bc_loss", None, persistent=False)
        self.register_buffer("fc_loss", None, persistent=False)
        self.register_buffer("residue", None, persistent=False)
        self.register_buffer("forecast", None, persistent=False)
        self.register_buffer("denominator", None, persistent=False)

    def _create_modules(self, *modules):
        self.blocks = nn.ModuleList(
            [AttentionBlock(*ms) for ms in zip(*modules)]
        )

    @property
    def tradeoff(self) -> Tensor:
        return self._softplus(self._tradeoff)

    @property
    def n_layer(self) -> int:
        return len(self.blocks)

    @property
    def tie_layers(self) -> bool:
        return (self.n_layer == 1) or (
            all(
                [
                    (a.encoder is b.encoder) and (a.decoder is b.decoder)
                    for a, b in product(self.blocks[:1], self.blocks[1:])
                ]
            )
        )

    def register_loss_func(self, func: LossFunction) -> None:
        if not isinstance(func, LossFunction):
            raise ValueError(
                "Expect a `LossFunction` object, "
                f"but receive a `{type(func).__class__.__name__}` object."
            )
        if hasattr(self, "_loss_hook_handle"):
            self._loss_hook_handle.remove()
            del self._loss_hook_handle
            del self._loss_func
        self._loss_func = func
        self._loss_hook_handle = self.register_forward_hook(
            self._loss_func.loss_hook
        )

    def _step(
        self,
        data: Tensor,
        feats: Optional[Tensor],
        mask: Optional[BoolTensor],
    ) -> Tuple[Tensor, Tensor]:
        cumsum = pt.tensor(0, dtype=data.dtype, device=data.device)
        residue = []
        forecast = []
        for block in self.blocks:
            interp, extrap = block(data, feats, mask)
            data = data - interp
            cumsum = cumsum + extrap
            residue.append(data)
            forecast.append(cumsum)
        residue = pt.stack(residue, dim=2)
        forecast = pt.stack(forecast, dim=2)
        return residue, forecast

    def _rollout(
        self,
        data: Tensor,
        feats: Optional[Tensor],
        mask: Optional[BoolTensor],
        total_steps: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        if feats is not None:
            prefix_feats = feats[:, : data.size(1)]
            n_steps = feats.size(1) - data.size(1) + 1
            if total_steps is None:
                total_steps = n_steps
            else:
                total_steps = min(total_steps, n_steps)
        else:
            if (total_steps is None) and (not self.training):
                raise ValueError("Total decoding steps must be provided.")
            prefix_feats = None

        residue, forecast = self._step(data, prefix_feats, mask)
        if not self.training:
            preds = forecast
            forecast = pt.narrow(forecast, dim=1, start=0, length=0)
            for step in range(total_steps):
                prev = pt.narrow(preds, dim=1, start=-1, length=1)
                forecast = pt.cat([forecast, prev], dim=1)
                prev = prev[..., -1, :]
                data = pt.cat([data, prev.detach()], dim=1)
                if feats is not None:
                    prefix_feats = pt.narrow(
                        feats, dim=1, start=0, length=data.size(1)
                    )
                if mask is not None:
                    mask = pt.cat(
                        [
                            mask,
                            pt.zeros_like(
                                pt.narrow(mask, dim=1, start=0, length=1)
                            ),
                        ],
                        dim=1,
                    )
                _, preds = self._step(data, prefix_feats, mask)

        return residue, forecast

    def forward(
        self,
        data: Tensor,
        feats: Optional[Tensor],
        nan_mask: Optional[BoolTensor],
        length: Optional[LongTensor],
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[BoolTensor], Optional[BoolTensor]
    ]:
        batch_size = data.size(0)
        full_length = data.size(1)

        # merge padding mask and missing mask
        if length is None:
            pad_mask = None
        else:
            # in our impl, padings are on left side
            pad_mask = pt.gt(
                pt.arange(full_length, 0, -1).to(length).unsqueeze(dim=0),
                length.unsqueeze(dim=1),
            )
        if (pad_mask is None) and (nan_mask is None):
            mask = None
        else:
            if pad_mask is None:
                pad_mask = pt.zeros_like(nan_mask)
            if nan_mask is None:
                nan_mask = pt.zeros_like(pad_mask)
            mask = pt.logical_or(pad_mask, nan_mask)

        # compute acutal horizons
        if isinstance(self.horizon, int):
            horizon = data.new_ones(batch_size, dtype=pt.long) * self.horizon
        else:
            if length is None:
                horizon = data.new_ones(batch_size, dtype=pt.long) * int(
                    full_length * self.horizon
                )
            else:
                horizon = length.to(pt.float).mul(self.horizon).to(pt.long)
        max_horizon = horizon.max().item()

        # split data into condition (input) and (forecasting) target
        if self.training:
            target = data[:, full_length - max_horizon : full_length]
            if mask is not None:
                bc_mask = mask
                fc_mask = pt.gt(
                    pt.arange(max_horizon, 0, -1).to(horizon).unsqueeze(dim=0),
                    horizon.unsqueeze(dim=1),
                )
                fc_mask = pt.logical_or(
                    fc_mask, mask[:, full_length - max_horizon : full_length]
                )
            else:
                bc_mask = fc_mask = None
            residue, forecast = self._rollout(data, feats, bc_mask)
            forecast = forecast[:, -(max_horizon + 1) : -1]
        else:
            if length is not None:
                # move part of paddings from left to right
                # pad_size = max_horizon - horizon
                start_index = pt.unsqueeze(max_horizon - horizon, dim=1)
                selected_index = (
                    pt.arange(full_length).to(horizon).unsqueeze(dim=0)
                    + start_index
                )
                selected_index = pt.where(
                    selected_index.lt(full_length),
                    selected_index,
                    selected_index - full_length,
                )
                data = pt.gather(
                    data,
                    dim=1,
                    index=selected_index.unsqueeze(dim=2).expand_as(data),
                )
                if feats is not None:
                    feats = pt.gather(
                        feats,
                        dim=1,
                        index=selected_index.unsqueeze(dim=2).expand_as(feats),
                    )
                mask = pt.gather(
                    mask,
                    dim=1,
                    index=selected_index,
                )

            # split condition and target
            cond_length = full_length - max_horizon
            data, target = pt.split(data, [cond_length, max_horizon], dim=1)
            # features are not split for predictions
            if mask is not None:
                bc_mask, fc_mask = pt.split(
                    mask, [cond_length, max_horizon], dim=1
                )
            else:
                bc_mask = fc_mask = None
            residue, forecast = self._rollout(
                data, feats, bc_mask, max_horizon
            )

        return residue, forecast, target, bc_mask, fc_mask

    @classmethod
    def from_configs(
        cls,
        d_data: int,
        d_feats: int,
        d_hidden: int,
        n_layer: int,
        horizon: Union[int, float],
        window_size: List[int],
        n_head: int = 4,
        n_enc_layer: int = 2,
        n_dec_layer: int = 2,
        symmetric: bool = False,
        share_values: bool = False,
        tie_casts: bool = False,
        tie_layers: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
        norm: str = "instance",
        add_dist: bool = False,
        normalize_input: bool = True,
        rescale_output: bool = True,
        layerwise_loss: bool = True,
    ):
        kernel = ExpKernel(
            d_hidden,
            n_head,
            n_enc_layer,
            n_dec_layer,
            symmetric=symmetric,
            share_values=share_values,
            dropout=dropout,
            temperature=temperature,
            norm=norm,
            add_dist=add_dist,
        )
        encoder = EncoderModule(
            d_data,
            d_feats,
            d_hidden,
            kernel.d_head if share_values else d_hidden,
            *window_size,
            tie_casts=tie_casts,
        )
        decoder = DecoderModule(d_data, d_hidden)
        if tie_layers:
            encoders = [encoder] * n_layer
            decoders = [decoder] * n_layer
            kernels = [kernel] * n_layer
        else:
            encoders = [deepcopy(encoder) for _ in range(n_layer)]
            decoders = [deepcopy(decoder) for _ in range(n_layer)]
            kernels = [deepcopy(kernel) for _ in range(n_layer)]

        return cls(
            encoders,
            kernels,
            decoders,
            horizon=horizon,
            normalize_input=normalize_input,
            rescale_output=rescale_output,
            layerwise_loss=layerwise_loss,
        )

    def create_twin_estimator(
        self,
        d_data: int,
        d_feats: int,
        **distinct_configs,
    ):
        kernels = [block.kernel for block in self.blocks]
        d_value = (
            kernels[0].d_head
            if kernels[0].share_values
            else kernels[0].d_hidden
        )
        d_hidden = kernels[0].d_hidden
        window_size = distinct_configs.get(
            "window_size", self.blocks[0].encoder.window_size
        )
        tie_casts = distinct_configs.get(
            "tie_casts", self.blocks[0].encoder.tie_casts
        )
        estimator_configs = dict(
            horizon=distinct_configs.get("horizon", self.horizon),
            loss_func=distinct_configs.get(
                "loss_func", type(self._loss_func)()
            ),
            normalize_input=distinct_configs.get(
                "normalize_input", self.normalize_input
            ),
            rescale_output=distinct_configs.get(
                "rescale_output", self.rescale_output
            ),
            layerwise_loss=distinct_configs.get(
                "layerwise_loss", self.layerwise_loss
            ),
        )

        encoder = EncoderModule(
            d_data,
            d_feats,
            d_hidden,
            d_value,
            *window_size,
            tie_casts=tie_casts,
        )
        decoder = DecoderModule(
            d_data,
            d_hidden,
        )
        if self.tie_layers:
            encoders = [encoder] * self.n_layer
            decoders = [decoder] * self.n_layer
        else:
            encoders = [deepcopy(encoder) for _ in range(self.n_layer)]
            decoders = [deepcopy(decoder) for _ in range(self.n_layer)]
        return type(self)(encoders, kernels, decoders, **estimator_configs)


class AdversarialEstimator(AttentionEstimator):
    def __init__(
        self,
        *lists_of_modules: List[nn.Module],
        horizon: Union[int, float],
        loss_func: LossFunction = MSELoss(),
        normalize_input: bool = True,
        rescale_output: bool = True,
        layerwise_loss: bool = True,
    ) -> None:
        super(AdversarialEstimator, self).__init__(
            *lists_of_modules,
            horizon=horizon,
            loss_func=loss_func,
            normalize_input=normalize_input,
            rescale_output=rescale_output,
            layerwise_loss=layerwise_loss,
        )
        self._generative = True

    def _create_modules(self, *modules):
        self.blocks = nn.ModuleList(
            [AdversarialBlock(*ms) for ms in zip(*modules)]
        )

    def generative(self) -> None:
        self._generative = True
        for block in self.blocks:
            block.generative()

    def discriminative(self) -> None:
        self._generative = False
        for block in self.blocks:
            block.discriminative()

    def generative_named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        for i, block in enumerate(self.blocks):
            for n, p in block.encoder.named_parameters():
                yield f"blocks.{i}.encoder.{n}", p
            for n, p in block.kernel.named_parameters():
                yield f"blocks.{i}.kernel.{n}", p
            for n, p in block.decoder.named_parameters():
                yield f"blocks.{i}.decoder.{n}", p

    def discriminative_named_parameters(
        self,
    ) -> Iterator[Tuple[str, Parameter]]:
        for i, block in enumerate(self.blocks):
            for n, p in block.disc.named_parameters():
                yield f"blocks.{i}.disc.{n}", p

    def generative_parameters(self) -> Iterator[Parameter]:
        for n, p in self.generative_named_parameters():
            yield p

    def discriminative_parameters(self) -> Iterator[Parameter]:
        for n, p in self.discriminative_named_parameters():
            yield p

    @property
    def prob_domain(self) -> Tensor:
        return pt.stack([block.prob_domain for block in self.blocks], dim=2)

    @classmethod
    def from_configs(
        cls,
        d_data: int,
        d_feats: int,
        d_hidden: int,
        n_layer: int,
        horizon: Union[int, float],
        window_size: List[int],
        n_head: int = 4,
        n_enc_layer: int = 2,
        n_dec_layer: int = 2,
        n_disc_layer: int = 2,
        symmetric: bool = False,
        share_values: bool = True,
        tie_casts: bool = False,
        tie_layers: bool = False,
        dropout: float = 0.0,
        temperature: float = 1.0,
        norm: str = "instance",
        add_dist: bool = False,
        normalize_input: bool = True,
        rescale_output: bool = True,
        layerwise_loss: bool = True,
    ):
        kernel = ExpKernel(
            d_hidden,
            n_head,
            n_enc_layer,
            n_dec_layer,
            symmetric=symmetric,
            share_values=share_values,
            dropout=dropout,
            temperature=temperature,
            norm=norm,
            add_dist=add_dist,
        )
        disc = SimpleDiscriminator(
            d_model=d_hidden,
            n_layer=n_disc_layer,
        )
        encoder = EncoderModule(
            d_data,
            d_feats,
            d_hidden,
            kernel.d_head if share_values else d_hidden,
            *window_size,
            tie_casts=tie_casts,
        )
        decoder = DecoderModule(
            d_data,
            d_hidden,
        )
        if tie_layers:
            encoders = [encoder] * n_layer
            decoders = [decoder] * n_layer
            kernels = [kernel] * n_layer
            discs = [disc] * n_layer
        else:
            encoders = [deepcopy(encoder) for _ in range(n_layer)]
            decoders = [deepcopy(decoder) for _ in range(n_layer)]
            kernels = [deepcopy(kernel) for _ in range(n_layer)]
            discs = [deepcopy(disc) for _ in range(n_layer)]

        return cls(
            encoders,
            kernels,
            decoders,
            discs,
            horizon=horizon,
            normalize_input=normalize_input,
            rescale_output=rescale_output,
            layerwise_loss=layerwise_loss,
        )

    @classmethod
    def from_base(
        cls,
        base: AttentionEstimator,
        n_disc_layer: int = 2,
    ):
        kernels = [deepcopy(block.kernel) for block in base.blocks]
        encoders = [deepcopy(block.encoder) for block in base.blocks]
        decoders = [deepcopy(block.decoder) for block in base.blocks]

        disc = SimpleDiscriminator(
            d_model=kernels[0].d_hidden, n_layer=n_disc_layer
        )
        n_layer = base.n_layer
        if base.tie_layers:
            discs = [disc] * n_layer
        else:
            discs = [deepcopy(disc) for _ in range(n_layer)]
        return cls(
            encoders,
            kernels,
            decoders,
            discs,
            horizon=base.horizon,
            loss_func=type(base._loss_func)(),
            normalize_input=base.normalize_input,
            rescale_output=base.rescale_output,
            layerwise_loss=base.layerwise_loss,
        )

    def create_twin_estimator(
        self,
        d_data: int,
        d_feats: int,
        **distinct_configs,
    ):
        kernels = [block.kernel for block in self.blocks]
        discs = [block.disc for block in self.blocks]
        d_value = (
            kernels[0].d_head
            if kernels[0].share_values
            else kernels[0].d_hidden
        )
        d_hidden = kernels[0].d_hidden
        window_size = distinct_configs.get(
            "window_size", self.blocks[0].encoder.window_size
        )
        tie_casts = distinct_configs.get(
            "tie_casts", self.blocks[0].encoder.tie_casts
        )
        estimator_configs = dict(
            horizon=distinct_configs.get("horizon", self.horizon),
            loss_func=distinct_configs.get(
                "loss_func", type(self._loss_func)()
            ),
            normalize_input=distinct_configs.get(
                "normalize_input", self.normalize_input
            ),
            rescale_output=distinct_configs.get(
                "rescale_output", self.rescale_output
            ),
            layerwise_loss=distinct_configs.get(
                "layerwise_loss", self.layerwise_loss
            ),
        )

        encoder = EncoderModule(
            d_data,
            d_feats,
            d_hidden,
            d_value,
            *window_size,
            tie_casts=tie_casts,
        )
        decoder = DecoderModule(
            d_data,
            d_hidden,
        )
        if self.tie_layers:
            encoders = [encoder] * self.n_layer
            decoders = [decoder] * self.n_layer
        else:
            encoders = [deepcopy(encoder) for _ in range(self.n_layer)]
            decoders = [deepcopy(decoder) for _ in range(self.n_layer)]
        return type(self)(
            encoders, kernels, decoders, discs, **estimator_configs
        )
