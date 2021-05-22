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

import numpy as np
import pandas as pd
import pytest

from gluonts.dataset.common import ListDataset


def point_process_dataset():
    ia_times = np.array([0.2, 0.7, 0.2, 0.5, 0.3, 0.3, 0.2, 0.1])
    marks = np.array([0, 1, 2, 0, 1, 2, 2, 2])

    lds = ListDataset(
        [
            {
                "target": np.c_[ia_times, marks].T,
                "start": pd.Timestamp("2011-01-01 00:00:00", freq="H"),
                "end": pd.Timestamp("2011-01-01 03:00:00", freq="H"),
            }
        ],
        freq="H",
        one_dim_target=False,
    )

    return lds


def point_process_dataset_2():
    lds = ListDataset(
        [
            {
                "target": np.c_[
                    np.array([0.2, 0.7, 0.2, 0.5, 0.3, 0.3, 0.2, 0.1]),
                    np.array([0, 1, 2, 0, 1, 2, 2, 2]),
                ].T,
                "start": pd.Timestamp("2011-01-01 00:00:00", freq="H"),
                "end": pd.Timestamp("2011-01-01 03:00:00", freq="H"),
            },
            {
                "target": np.c_[
                    np.array([0.2, 0.1, 0.2, 0.1, 0.3, 0.3, 0.5, 0.4]),
                    np.array([0, 1, 2, 0, 1, 2, 1, 1]),
                ].T,
                "start": pd.Timestamp("2011-01-01 00:00:00", freq="H"),
                "end": pd.Timestamp("2011-01-01 03:00:00", freq="H"),
            },
            {
                "target": np.c_[
                    np.array([0.2, 0.7, 0.2, 0.5, 0.1, 0.1, 0.2, 0.1]),
                    np.array([0, 1, 2, 0, 1, 0, 1, 2]),
                ].T,
                "start": pd.Timestamp("2011-01-01 00:00:00", freq="H"),
                "end": pd.Timestamp("2011-01-01 03:00:00", freq="H"),
            },
        ],
        freq="H",
        one_dim_target=False,
    )

    return lds
