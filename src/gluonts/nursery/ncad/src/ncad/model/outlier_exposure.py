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


def coe_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    coe_rate: float,
    suspect_window_length: int,
    random_start_end: bool = True,
) -> torch.Tensor:
    """Contextual Outlier Exposure.

    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        coe_rate : Number of generated anomalies as proportion of the batch size.
        random_start_end : If True, a random subset within the suspect segment is permuted between time series;
            if False, the whole suspect segment is randomly permuted.
    """

    if coe_rate == 0:
        raise ValueError(f"coe_rate must be > 0.")
    batch_size = x.shape[0]
    ts_channels = x.shape[1]
    oe_size = int(batch_size * coe_rate)

    # Select indices
    idx_1 = torch.arange(oe_size)
    idx_2 = torch.arange(oe_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()

    if ts_channels > 3:
        numb_dim_to_swap = np.random.randint(low=3, high=ts_channels, size=(oe_size))
        # print(numb_dim_to_swap)
    else:
        numb_dim_to_swap = np.ones(oe_size) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = np.random.randint(
        low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2)
    )
    oe_time_start_end.sort(axis=1)
    # for start, end in oe_time_start_end:
    for i in range(len(idx_2)):
        # obtain the dimensons to swap
        numb_dim_to_swap_here = int(numb_dim_to_swap[i])
        dims_to_swap_here = np.random.choice(
            range(ts_channels), size=numb_dim_to_swap_here, replace=False
        )

        # obtain start and end of swap
        start, end = oe_time_start_end[i]

        # swap
        x_oe[i, dims_to_swap_here, start:end] = x[idx_2[i], dims_to_swap_here, start:end]

    # Label as positive anomalies
    y_oe = torch.ones(oe_size).type_as(y)

    return x_oe, y_oe
