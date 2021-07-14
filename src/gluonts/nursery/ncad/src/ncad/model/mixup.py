# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


import numpy as np
import torch

def mixup_batch(
            x: torch.Tensor,
            y: torch.Tensor,
            mixup_rate: float,
        ) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, ts channels, time)
            y : Tensor of shape (batch, )
            mixup_rate : Number of generated anomalies as proportion of the batch size.
        """

        if mixup_rate == 0:
            raise ValueError(f"mixup_rate must be > 0.")
        batch_size = x.shape[0]
        mixup_size = int(batch_size * mixup_rate) #

        # Select indices
        idx_1 = torch.arange(mixup_size)
        idx_2 = torch.arange(mixup_size)
        while torch.any(idx_1 == idx_2):
            idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()
            idx_2 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()

        # sample mixing weights:
        beta_param = float(0.05)
        beta_distr = torch.distributions.beta.Beta(torch.tensor([beta_param]), torch.tensor([beta_param]))
        weights = torch.from_numpy( np.random.beta(beta_param, beta_param, (mixup_size,)) ).type_as(x)
        oppose_weights = 1.0 - weights

        # Create contamination
        x_mix_1 = x[idx_1].clone()
        x_mix_2 = x[idx_1].clone()
        x_mixup = x_mix_1 * weights[:,None,None] + x_mix_2 * oppose_weights[:,None,None] # .detach()

        # Label as positive anomalies
        y_mixup = y[idx_1].clone() * weights + y[idx_2].clone() * oppose_weights

        return x_mixup, y_mixup
