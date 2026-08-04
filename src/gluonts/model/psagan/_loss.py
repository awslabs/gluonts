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
import torch.nn as nn


class MommentLoss(nn.Module):
    """Momment Loss"""

    def __init__(self):
        super(MommentLoss, self).__init__()

    def _momment_loss(self, preds, target):
        std_loss = torch.abs(preds.std(axis=2) - target.std(axis=2)).mean()
        mean_loss = torch.abs(preds.mean(axis=2) - target.mean(axis=2)).mean()
        momment_loss = std_loss + mean_loss
        return momment_loss

    def forward(self, preds, target):
        return self._momment_loss(preds, target)


class SupervisedLoss(nn.Module):
    """Supervised Loss for pre-training the generator"""

    def __init__(self):
        super(SupervisedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.momment = MommentLoss()

    def forward(self, preds, target):
        l1 = self.l1(preds, target)
        momment = self.momment(preds, target)
        return l1 + momment
