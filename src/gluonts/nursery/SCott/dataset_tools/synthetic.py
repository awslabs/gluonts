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

from pts.dataset.repository.datasets import get_dataset, dataset_recipes
from pts.dataset.utils import to_pandas
import numpy as np
import pandas as pd
from pts.dataset import ListDataset
import torch
from pts.model.simple_feedforward import SimpleFeedForwardEstimator
import pickle
import json
import random


def _get_mixed_pattern(x, pattern):
    if pattern == "sin":
        return (torch.sin(x) + torch.normal(0, 0.1, size=x.shape)) * 5
    elif pattern == "linear":
        return x * 2 + torch.normal(0, 1, size=x.shape)
    elif pattern == "quadratic":
        return x**1.5 + torch.normal(0, 5, size=x.shape)
    else:
        return torch.sqrt(x) + torch.normal(0, 1, size=x.shape)


def get_mixed_pattern(unit_length=16, num_duplicates=1000):
    freq = "1H"
    context_length = 3 * unit_length
    prediction_length = unit_length
    len_sample = context_length + prediction_length

    dataset_group = [[] for j in range(16)]
    whole_data = []
    val_data = []
    ret = dict()
    start = pd.Timestamp("01-01-2000", freq=freq)
    patterns = [
        ["sin", "linear", "quadratic", "sqrt"],
        ["sqrt", "quadratic", "linear", "sin"],
        ["linear", "sqrt", "sin", "quadratic"],
        ["quadratic", "sin", "sqrt", "linear"],
    ]
    pattern_number = 4
    for m, pattern in enumerate(patterns):
        for gid in range(pattern_number):
            for j in range(num_duplicates):
                context = torch.arange(context_length, dtype=torch.float)
                for i in range(1, pattern_number):
                    context[
                        unit_length * (i - 1) : unit_length * i
                    ] = _get_mixed_pattern(
                        context[unit_length * (i - 1) : unit_length * i]
                        - unit_length * (i - 1),
                        pattern[(gid + i) % pattern_number],
                    )
                ts_sample = torch.cat(
                    [
                        context,
                        _get_mixed_pattern(
                            torch.arange(prediction_length, dtype=torch.float),
                            pattern[gid],
                        ),
                    ]
                )
                whole_data.append({"target": ts_sample, "start": start})
                if j % 5 == 0:
                    val_data.append(
                        {
                            "target": ts_sample
                            + torch.normal(0, 1, ts_sample.shape),
                            "start": start,
                        }
                    )
                dataset_group[m * 4 + gid].append(
                    {"target": ts_sample, "start": start}
                )
    print(len(whole_data))
    print(len(val_data))
    ret["group_ratio"] = [len(i) / len(whole_data) for i in dataset_group]
    print(ret["group_ratio"])
    random.shuffle(whole_data)
    group_data = []
    ret["whole_data"] = ListDataset(whole_data, freq=freq)
    ret["val_data"] = ListDataset(val_data, freq=freq)
    for group in dataset_group:
        random.shuffle(group)
        group_data.append(ListDataset(group, freq=freq))
    ret["group_data"] = group_data

    # save to files
    with open("../dataset/mix.csv", "wb") as output:
        pickle.dump(ret, output)

    return True
