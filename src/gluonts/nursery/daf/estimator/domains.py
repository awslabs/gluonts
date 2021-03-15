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


from typing import Optional, List, Tuple, Union, Iterator
from collections import defaultdict
import warnings
import math

import torch as pt
from torch import Tensor, LongTensor, BoolTensor
from torch import nn
from torch.nn import Parameter, functional as F

from .modules import AttentionEstimator, AdversarialEstimator
from ..network import AttentionKernel, AttentionBlock, AdversarialBlock


class DomAdaptEstimator(nn.Module):
    def __init__(
        self,
        src_module: AttentionEstimator,
        tgt_module: AttentionEstimator,
        balance_loss: bool = True,
        forecast_target: bool = True,
    ) -> None:
        super(DomAdaptEstimator, self).__init__()
        self.src = src_module
        self.tgt = tgt_module
        self.balance_loss = balance_loss
        self.forecast_target = forecast_target

    def forward(
        self,
        src_data: Tensor,
        tgt_data: Tensor,
        src_feats: Optional[Tensor] = None,
        tgt_feats: Optional[Tensor] = None,
        src_nan_mask: Optional[BoolTensor] = None,
        tgt_nan_mask: Optional[BoolTensor] = None,
        src_length: Optional[LongTensor] = None,
        tgt_length: Optional[LongTensor] = None,
    ) -> Tensor:
        src_loss = self.src(src_data, src_feats, src_nan_mask, src_length)
        tgt_loss = self.tgt(tgt_data, tgt_feats, tgt_nan_mask, tgt_length)
        if not self.forecast_target:
            tgt_loss = tgt_loss - self.tgt.tradeoff * self.tgt.fc_loss
        # after rescaling, balance two losses
        if self.balance_loss:
            src_scale = self.src._normalizer._buffers["scale"]
            tgt_scale = self.tgt._normalizer._buffers["scale"]
            src_scale = src_scale.view(src_scale.size(0), -1)
            tgt_scale = tgt_scale.view(tgt_scale.size(0), -1)
            weight = pt.mean(tgt_scale / src_scale, dim=1)
            src_loss = src_loss * weight
        loss = src_loss + tgt_loss
        return loss


class AdversarialDomAdaptEstimator(DomAdaptEstimator):
    def __init__(
        self,
        src_module: AdversarialEstimator,
        tgt_module: AdversarialEstimator,
        balance_loss: bool = True,
        forecast_target: bool = True,
        disc_lambda: float = 1.0,
    ) -> None:
        super(AdversarialDomAdaptEstimator, self).__init__(
            src_module,
            tgt_module,
            balance_loss,
            forecast_target,
        )
        self.disc_lambda = disc_lambda
        self._generative = True

    def generative(self):
        self._generative = True
        self.src.generative()
        self.tgt.generative()

    def discriminative(self):
        self._generative = False
        self.src.discriminative()
        self.tgt.discriminative()

    def forward(
        self,
        src_data: Tensor,
        tgt_data: Tensor,
        src_feats: Optional[Tensor] = None,
        tgt_feats: Optional[Tensor] = None,
        src_nan_mask: Optional[BoolTensor] = None,
        tgt_nan_mask: Optional[BoolTensor] = None,
        src_length: Optional[LongTensor] = None,
        tgt_length: Optional[LongTensor] = None,
    ) -> Tensor:
        gen_loss = super(AdversarialDomAdaptEstimator, self).forward(
            src_data,
            tgt_data,
            src_feats,
            tgt_feats,
            src_nan_mask,
            tgt_nan_mask,
            src_length,
            tgt_length,
        )

        batch_size = gen_loss.size(0)
        src_prob_domain = self.src.prob_domain.view(batch_size, -1)
        tgt_prob_domain = self.tgt.prob_domain.view(batch_size, -1)
        adv_type = "grad_rev"
        # adv_type = 'label_inv'

        # disc_loss = -log p(is_src | shape_src) - log(1 - p(is_src|shape_tgt))
        disc_loss = -src_prob_domain.mean(dim=1) - pt.log(
            1.0 - tgt_prob_domain.exp() + 1e-10
        ).mean(dim=1)
        if self._generative:
            if adv_type == "grad_rev":
                # DANN: gradient reversal operation
                # Generative model maximizes discriminative loss
                loss = gen_loss - self.disc_lambda * disc_loss
            elif adv_type == "label_inv":
                # ADDA: GAN-style minimax game
                # Invert the label of target data to make pseudo source samples
                # Do not update for real source samples
                # invert_tgt_loss = -log p(is_src | shape_tgt)
                invert_tgt_loss = -tgt_prob_domain.mean(dim=1)
                loss = gen_loss - self.disc_lambda * invert_tgt_loss
        else:
            loss = disc_loss

        return loss
