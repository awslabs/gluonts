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


from typing import List, NamedTuple
import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset


class HierarchicalMetaData(NamedTuple):
    S: np.ndarray
    freq: str
    nodes: List


class HierarchicalTrainDatasets(NamedTuple):
    train: ListDataset
    test: ListDataset
    metadata: HierarchicalMetaData


def sine7(seq_length: int, prediction_length: int):
    x = np.arange(0, seq_length)

    # Bottom layer (4 series)
    amps = [0.8, 0.9, 1, 1.1]
    freqs = [1 / 20, 1 / 30, 1 / 50, 1 / 100]

    b = np.zeros((4, seq_length))
    for i, f in enumerate(freqs):
        omega = 0
        if i == 3:
            np.random.seed(0)
            omega = np.random.uniform(0, np.pi)  # random phase shift
        b[i, :] = amps[i] * np.sin(2 * np.pi * x * f + omega)

    # Aggregation matrix S
    S = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )

    Y = S @ b

    # Indices and timestamps
    index = pd.period_range(
        start=pd.Period("2020-01-01", freq="D"),
        periods=Y.shape[1],
        freq="D",
    )

    metadata = HierarchicalMetaData(
        S=S, freq=index.freqstr, nodes=[2, [2] * 2]
    )

    train_dataset = ListDataset(
        [
            {
                "start": index[0],
                "item_id": "all_items",
                "target": Y[:, :-prediction_length],
            }
        ],
        freq=index.freqstr,
        one_dim_target=False,
    )

    test_dataset = ListDataset(
        [{"start": index[0], "item_id": "all_items", "target": Y}],
        freq=index.freqstr,
        one_dim_target=False,
    )

    assert Y.shape[0] == S.shape[0]
    return HierarchicalTrainDatasets(
        train=train_dataset, test=test_dataset, metadata=metadata
    )
