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


from typing import List

from torch import optim
from torch.optim import lr_scheduler


class InverseSquareRootScheduler(lr_scheduler._LRScheduler):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from zero until the configured learning rate (``--lr``).
    Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup::
      lrs = torch.linspace(0, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        init_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError(
                f"warmup steps should be nonnegative, but is set as {warmup_steps}"
            )
        self.warmup_steps = float(warmup_steps)
        self.init_lr = init_lr
        self.decay_factor = max(1, self.warmup_steps ** 0.5)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                max(
                    self.init_lr, base_lr * self.last_epoch / self.warmup_steps
                )
                for base_lr in self.base_lrs
            ]
        else:
            lr_factor = self.decay_factor * (self.last_epoch + 1) ** -0.5
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class ExponentialWithWarmupScheduler(lr_scheduler.ExponentialLR):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    We also support a warmup phase where we linearly increase the learning rate
    from zero until the configured learning rate (``--lr``).
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        warmup_steps (int): Warmup steps..
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        gamma: float,
        warmup_steps: int,
        init_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError(
                f"warmup steps should be nonnegative, but is set as {warmup_steps}"
            )
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                max(
                    self.init_lr, base_lr * self.last_epoch / self.warmup_steps
                )
                for base_lr in self.base_lrs
            ]
        else:
            lr_factor = self.gamma ** (self.last_epoch - self.warmup_steps)
            return [base_lr * lr_factor for base_lr in self.base_lrs]
