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
import os
import numpy as np
import pandas as pd
from pts.dataset import ListDataset
import torch
import pickle
import json
import random
import os


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
    print(
        "Generating the synthetic training data, the total number of training"
        " examples:",
        len(whole_data),
    )
    ret["group_ratio"] = [len(i) / len(whole_data) for i in dataset_group]
    random.shuffle(whole_data)
    group_data = []
    ret["whole_data"] = ListDataset(whole_data, freq=freq)
    ret["val_data"] = ListDataset(val_data, freq=freq)
    for group in dataset_group:
        random.shuffle(group)
        group_data.append(ListDataset(group, freq=freq))
    ret["group_data"] = group_data
    # save to files
    os.makedirs("./dataset", exist_ok=True)
    with open("./dataset/synthetic.csv", "wb") as output:
        pickle.dump(ret, output)
    print("Finished the pre-processing of synthetic dataset")

    return True


def group_electricity_cv(
    num_ts=10,
    num_groups=14,
    context_length=72,
    prediction_length=12,
    file_name="default",
):
    dataset = get_dataset("electricity", regenerate=True)
    len_sample = context_length + prediction_length
    dataset_group = [[] for i in range(num_groups)]
    train_full_data = []
    test_full_data = []
    ret = dict()
    train_it = iter(dataset.train)
    test_it = iter(dataset.test)
    date_checkpoint = [
        "2012-03-01",
        "2012-06-01",
        "2012-09-01",
        "2012-12-01",
        "2013-03-01",
        "2013-06-01",
        "2013-09-01",
        "2013-12-01",
        "2014-03-01",
    ]
    # get ready the training data
    for i in range(num_ts):
        train_entry = next(train_it)
        unsplit_ts = train_entry["target"]
        unsplit_start = train_entry["start"]
        t = unsplit_start
        start_date = 4

        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            for j, date_ckpt in enumerate(date_checkpoint):
                if unsplit_start < pd.Timestamp(date_ckpt):
                    sid = j
                    break
                elif unsplit_start > pd.Timestamp(date_checkpoint[-1]):
                    sid = len(date_checkpoint)
                    break
            gid = ((start_date + 1) % 7) + sid * 7
            start_date += 1
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            train_full_data.append(
                {
                    "target": ts_slice,
                    "start": t,
                    "feat_static_cat": np.array([gid]),
                }
            )
            dataset_group[gid].append(
                {
                    "target": ts_slice,
                    "start": t,
                    "feat_static_cat": np.array([gid]),
                }
            )
            unsplit_start += pd.Timedelta(hours=prediction_length)

    # get ready the test data
    for i in range(int(num_ts * 0.2)):
        test_entry = next(test_it)
        unsplit_ts = test_entry["target"]
        unsplit_start = test_entry["start"]
        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            test_full_data.append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": test_entry["feat_static_cat"],
                }
            )

    print(
        "Generating the electricity training data, the total number of"
        " training examples:",
        len(train_full_data),
    )
    ret["group_ratio"] = [len(i) / len(train_full_data) for i in dataset_group]
    random.shuffle(train_full_data)
    ret["whole_data"] = ListDataset(
        train_full_data, freq=dataset.metadata.freq
    )
    random.shuffle(test_full_data)
    ret["val_data"] = ListDataset(test_full_data, freq=dataset.metadata.freq)
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq=dataset.metadata.freq))
    ret["group_data"] = group_data_list
    os.makedirs("./dataset", exist_ok=True)
    with open("./dataset/" + file_name + ".csv", "wb") as output:
        pickle.dump(ret, output)
    print("Finished pre-processing of the electricity dataset")
    return True

    dataset = get_dataset("traffic")
    len_sample = context_length + prediction_length
    dataset_group = [[] for i in range(num_groups)]
    train_full_data = []
    test_full_data = []
    ret = dict()
    train_it = iter(dataset.train)
    test_it = iter(dataset.test)
    # num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    date_checkpoint = ["2016-01-01"]
    # get ready the training data
    for i in range(num_ts):
        train_entry = next(train_it)
        unsplit_ts = train_entry["target"]
        unsplit_start = train_entry["start"]
        t = unsplit_start
        start_date = 4

        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            for j, date_ckpt in enumerate(date_checkpoint):
                if unsplit_start < pd.Timestamp(date_ckpt):
                    sid = j
                    break
                elif unsplit_start > pd.Timestamp(date_checkpoint[-1]):
                    sid = len(date_checkpoint)
                    break
            gid = ((start_date + 1) % 7) + sid * 7
            start_date += 1
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            train_full_data.append(
                {
                    "target": ts_slice,
                    "start": t,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            dataset_group[gid].append(
                {
                    "target": ts_slice,
                    "start": t,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            unsplit_start += pd.Timedelta(hours=prediction_length)

    # get ready the test data
    for i in range(int(num_ts * 0.2)):
        test_entry = next(test_it)
        unsplit_ts = test_entry["target"]
        unsplit_start = test_entry["start"]
        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            test_full_data.append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": test_entry["feat_static_cat"],
                }
            )

    print("total number of training examples: ", len(train_full_data))
    ret["group_ratio"] = [len(i) / len(train_full_data) for i in dataset_group]
    print("ratio for each group: ", ret["group_ratio"])
    random.shuffle(train_full_data)
    ret["whole_data"] = ListDataset(
        train_full_data, freq=dataset.metadata.freq
    )
    random.shuffle(test_full_data)
    ret["val_data"] = ListDataset(test_full_data, freq=dataset.metadata.freq)
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq=dataset.metadata.freq))
    ret["group_data"] = group_data_list
    os.makedirs("./dataset", exist_ok=True)
    with open("./dataset/" + file_name + ".csv", "wb") as output:
        pickle.dump(ret, output)
    return True


def group_exchangerate_cv(
    num_ts=10,
    num_groups=14,
    context_length=15,
    prediction_length=10,
    file_name="default",
):
    dataset = get_dataset("exchange_rate", regenerate=True)
    len_sample = context_length + prediction_length
    dataset_group = [[] for i in range(num_groups)]
    train_full_data = []
    test_full_data = []
    ret = dict()
    train_it = iter(dataset.train)
    test_it = iter(dataset.test)
    # num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    date_checkpoint = ["1994-01-01", "1998-01-01", "2002-01-01"]
    for i in range(num_ts):
        train_entry = next(train_it)
        unsplit_ts = train_entry["target"]
        unsplit_start = train_entry["start"]
        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            for j, date_ckpt in enumerate(date_checkpoint):
                if unsplit_start < pd.Timestamp(date_ckpt):
                    sid = j
                    break
                elif unsplit_start > pd.Timestamp(date_checkpoint[-1]):
                    sid = len(date_checkpoint)
                    break
            gid = i * 4 + sid
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            train_full_data.append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            dataset_group[gid].append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            unsplit_start += pd.Timedelta("1D") * prediction_length
    # get ready the test data
    for i in range(int(num_ts * 0.2)):
        test_entry = next(test_it)
        unsplit_ts = test_entry["target"]
        unsplit_start = test_entry["start"]
        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            test_full_data.append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": test_entry["feat_static_cat"],
                }
            )
    print(
        "Generating the exchange rate training data, the total number of"
        " training examples:",
        len(train_full_data),
    )
    ret["group_ratio"] = [len(i) / len(train_full_data) for i in dataset_group]
    random.shuffle(train_full_data)
    ret["whole_data"] = ListDataset(
        train_full_data, freq=dataset.metadata.freq
    )
    random.shuffle(test_full_data)
    ret["val_data"] = ListDataset(test_full_data, freq=dataset.metadata.freq)
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq=dataset.metadata.freq))
    ret["group_data"] = group_data_list
    os.makedirs("./dataset", exist_ok=True)
    with open("./dataset/" + file_name + ".csv", "wb") as output:
        pickle.dump(ret, output)
    print("Finished pre-processing the exchange rate dataset")
    return True


def group_traffic_cv(
    num_ts=10,
    num_groups=14,
    context_length=72,
    prediction_length=12,
    file_name="default",
):
    dataset = get_dataset("traffic", regenerate=True)
    len_sample = context_length + prediction_length
    dataset_group = [[] for i in range(num_groups)]
    train_full_data = []
    test_full_data = []
    ret = dict()
    train_it = iter(dataset.train)
    test_it = iter(dataset.test)
    # num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    date_checkpoint = [
        "2015-03-01",
        "2015-06-01",
        "2015-09-01",
        "2015-12-01",
        "2016-03-01",
        "2016-06-01",
    ]
    # get ready the training data
    for i in range(num_ts):
        train_entry = next(train_it)
        unsplit_ts = train_entry["target"]
        unsplit_start = train_entry["start"]
        t = unsplit_start
        start_date = 4
        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            for j, date_ckpt in enumerate(date_checkpoint):
                if unsplit_start < pd.Timestamp(date_ckpt):
                    sid = j
                    break
                elif unsplit_start > pd.Timestamp(date_checkpoint[-1]):
                    sid = len(date_checkpoint)
                    break
            gid = ((start_date + 1) % 7) + sid * 7
            start_date += 1
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            train_full_data.append(
                {
                    "target": ts_slice,
                    "start": t,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            dataset_group[gid].append(
                {
                    "target": ts_slice,
                    "start": t,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            unsplit_start += pd.Timedelta(hours=prediction_length)

    # get ready the test data
    for i in range(int(num_ts * 0.2)):
        test_entry = next(test_it)
        unsplit_ts = test_entry["target"]
        unsplit_start = test_entry["start"]
        for ts_sample_start in range(
            0, len(unsplit_ts) - len_sample, prediction_length
        ):
            ts_slice = unsplit_ts[
                ts_sample_start : ts_sample_start + len_sample
            ]
            test_full_data.append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": test_entry["feat_static_cat"],
                }
            )

    print(
        "Generating the traffic training data, the total number of training"
        " examples:",
        len(train_full_data),
    )
    ret["group_ratio"] = [len(i) / len(train_full_data) for i in dataset_group]
    random.shuffle(train_full_data)
    ret["whole_data"] = ListDataset(
        train_full_data, freq=dataset.metadata.freq
    )
    random.shuffle(test_full_data)
    ret["val_data"] = ListDataset(test_full_data, freq=dataset.metadata.freq)
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq=dataset.metadata.freq))
    ret["group_data"] = group_data_list
    os.makedirs("./dataset", exist_ok=True)
    with open("./dataset/" + file_name + ".csv", "wb") as output:
        pickle.dump(ret, output)
    print("Finished pre-processing the traffic dataset")
    return True


get_mixed_pattern(unit_length=24, num_duplicates=2000)
group_traffic_cv(
    num_ts=800,
    num_groups=49,
    context_length=72,
    prediction_length=24,
    file_name="traffic",
)
group_exchangerate_cv(
    num_ts=8,
    num_groups=32,
    context_length=8,
    prediction_length=1,
    file_name="exchange_rate",
)
group_electricity_cv(
    num_ts=300,
    num_groups=70,
    context_length=72,
    prediction_length=24,
    file_name="electricity",
)
print(
    "Finished the preprocessing data, please verify ./dataset/ contains four"
    " .csv files"
)
