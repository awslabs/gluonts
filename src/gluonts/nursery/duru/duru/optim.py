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
import torch

# from apex.optimizers import FusedAdam as AdamW


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters

    return f


def init_optimizer_scheduler(H, hvae):
    # Note: Apex optimizer was used in the original VDVAE code --> speed difference?
    optimizer = torch.optim.AdamW(
        params=hvae.parameters(),
        weight_decay=H.adam_weight_decay,
        lr=H.lr,
        betas=(H.adam_beta_1, H.adam_beta_2),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=linear_warmup(H.warmup_iters)
    )

    return optimizer, scheduler


def optimizer_step_hvae_eval(hvae, hvae_eval, ema_rate):
    """
    Just as done in VDM (search for "exponential moving average", page 14).
    Performs this in-place.
    """
    for p1, p2 in zip(hvae.parameters(), hvae_eval.parameters()):
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))
