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


import torch as pt
from torch import Tensor, BoolTensor
from torch.nn.functional import l1_loss, mse_loss
from torch.distributions import Distribution, Normal as Gaussian


def quantile_error(preds: Tensor, target: Tensor, percentage: float) -> Tensor:
    diff = target - preds
    weight = pt.where(
        condition=diff > 0,
        input=diff.new_tensor(percentage),
        other=diff.new_tensor(percentage - 1),
    )
    return diff * weight


def quantile_percent_error(
    preds: Tensor, target: Tensor, percentage: float
) -> Tensor:
    qe = quantile_error(preds, target, percentage)
    qpe = qe * 2 / target
    qpe = qpe.masked_fill_(target <= 0, 0.0)
    return qpe


def absolute_error(preds: Tensor, target: Tensor) -> Tensor:
    return quantile_error(preds, target, 0.5) * 2


def absolute_percent_error(preds: Tensor, target: Tensor) -> Tensor:
    return quantile_percent_error(preds, target, 0.5)


def root_square_error(preds: Tensor, target: Tensor) -> Tensor:
    return mse_loss(preds, target, reduction="none").sqrt()


def root_square_percent_error(preds: Tensor, target: Tensor) -> Tensor:
    se = root_square_error(preds, target)
    rspe = se / target.abs()
    return rspe


def tweedie_loss(p: float):
    def tweedie_ll(preds: Tensor, target: Tensor):
        return -target * pt.pow(preds, 1 - p) / (1 - p) + pt.pow(
            preds, 2 - p
        ) / (2 - p)

    return tweedie_ll
