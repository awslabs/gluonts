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


from typing import TYPE_CHECKING, Optional

import torch as pt
from torch import Tensor, BoolTensor
from torch.nn import functional as F


class ZScoreNormalizer(object):
    def __init__(
        self,
        eps: float = 1e-5,
        rescale_loss: bool = True,
    ) -> None:
        self._buffers = {
            "offset": None,
            "scale": None,
        }
        self.eps = eps
        self.rescale_loss = rescale_loss

    def forward_pre_hook(self, module, input):
        data = input[0]
        feats = input[1] if len(input) > 1 else None
        nan_mask = input[2] if len(input) > 2 else None
        length = input[3] if len(input) > 3 else None
        if not nan_mask.any().item():
            nan_mask = None
        if (length is None) and (nan_mask is None):
            scale = data.std(dim=1, unbiased=False, keepdim=True)
            offset = data.mean(dim=1, keepdim=True)
        else:
            if nan_mask is None:
                count = length.unsqueeze(dim=1).float()
                tensor = data
            else:
                tensor = data.masked_fill(nan_mask.unsqueeze(dim=-1), 0.0)
                if length is None:
                    count = (
                        nan_mask.logical_not().float().sum(dim=1, keepdim=True)
                    )
                else:
                    count = length.unsqueeze(
                        dim=1
                    ).float() - nan_mask.float().sum(dim=1, keepdim=True)
            offset = tensor.sum(dim=1).div(count)
            offset = offset.unsqueeze(dim=1)
            scale = tensor.pow(2).sum(dim=1).div(count)
            scale = scale.unsqueeze(dim=1)
            scale = scale.sub(offset.pow(2))
            scale = scale.sqrt()
        scale = scale + self.eps
        data = data.sub(offset).div(scale)
        self._buffers["offset"] = offset
        self._buffers["scale"] = scale

        return data, feats, nan_mask, length

    def forward_hook(self, module, input, output):
        residue, forecast, target, bc_mask, fc_mask = output
        offset = self._buffers["offset"]
        scale = self._buffers["scale"]
        rescaled_residue = residue * scale.unsqueeze(dim=2)
        rescaled_forecast = forecast * scale.unsqueeze(
            dim=2
        ) + offset.unsqueeze(dim=2)
        module.residue = rescaled_residue
        module.forecast = rescaled_forecast
        if self.rescale_loss:
            target = target * scale + offset
            residue = rescaled_residue
            forecast = rescaled_forecast
        return residue, forecast, target, bc_mask, fc_mask


class LossFunction(object):
    def __call__(self, preds: Tensor, truth: Tensor) -> Tensor:
        raise NotImplementedError

    def _aggregate_loss(
        self, loss: Tensor, loss_mask: Optional[BoolTensor]
    ) -> Tensor:
        batch_size = loss.size(0)
        if loss_mask is None:
            length = loss.size(1)
            loss = loss.view(batch_size, -1).sum(dim=1)
            if length > 0:
                loss = loss / length
        else:
            loss = loss.masked_fill(
                loss_mask.view(*loss_mask.shape, 1, 1), 0.0
            )
            loss = loss.view(batch_size, -1).sum(dim=1)
            loss_count = loss_mask.logical_not().to(pt.float).sum(dim=1)
            loss = pt.where(loss_count.eq(0), loss, loss.div(loss_count))
        return loss

    def loss_hook(self, module, input, output):
        residue, forecast, target, bc_mask, fc_mask = output
        bc_loss = self(residue, pt.zeros_like(residue))
        fc_loss = self(forecast, target.unsqueeze(dim=2).expand_as(forecast))
        if not (module.training and module.layerwise_loss):
            bc_loss = bc_loss[..., -1:, :]
            fc_loss = fc_loss[..., -1:, :]
        bc_loss = self._aggregate_loss(bc_loss, bc_mask)
        fc_loss = self._aggregate_loss(fc_loss, fc_mask)

        module.bc_loss = bc_loss.detach()
        module.fc_loss = fc_loss.detach()
        module.denominator = self._aggregate_loss(
            target.unsqueeze(dim=2), fc_mask
        )

        loss = bc_loss + module.tradeoff * fc_loss
        return loss


class MSELoss(LossFunction):
    def __call__(self, preds: Tensor, truth: Tensor) -> Tensor:
        return F.mse_loss(preds, truth, reduction="none")

    def loss_hook(self, module, input, output):
        loss = super(MSELoss, self).loss_hook(module, input, output)
        module.bc_loss = module.bc_loss.sqrt()
        module.fc_loss = module.fc_loss.sqrt()
        return loss


class MAELoss(LossFunction):
    def __call__(self, preds: Tensor, truth: Tensor) -> Tensor:
        return pt.abs(preds - truth)
