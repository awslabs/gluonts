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


def get_m4_by_freq(
    context_length=72,
    prediction_length=24,
    len_per_ts=200,
    num_ts=50,
    num_groups=6,
    file_name="m4_freq",
):
    dataset_group = [[] for i in range(num_groups)]
    whole_data = []
    ret = dict()
    datasets_name = [
        "m4_hourly",
        "m4_daily",
        "m4_weekly",
        "m4_monthly",
        "m4_quarterly",
        "m4_yearly",
    ]
    hours_factor = [
        1,
        24,
        24 * 7,
        24 * 7 * 30,
        24 * 7 * 30 * 3,
        24 * 7 * 30 * 3 * 4,
    ]
    for i in range(num_groups):
        dataset = get_dataset(datasets_name[i])
        len_sample = context_length + prediction_length
        it = iter(dataset.train)
        for j in range(num_ts):
            train_entry = next(it)
            unsplit_ts = train_entry["target"]
            # unsplit_start = train_entry['start']
            unsplit_start = pd.Timestamp("1990-01-01")
            for ts_sample_start in range(
                0, len_per_ts - len_sample, prediction_length
            ):
                if len_sample > len(unsplit_ts):
                    continue
                ts_slice = unsplit_ts[
                    ts_sample_start : ts_sample_start + len_sample
                ]
                if len(ts_slice) < len_sample:
                    continue
                nu = 1 + sum(ts_slice) / len_sample
                ts_slice = [i / nu for i in ts_slice]
                whole_data.append(
                    {
                        "target": ts_slice,
                        "start": unsplit_start,
                        "feat_static_cat": train_entry["feat_static_cat"],
                    }
                )
                dataset_group[i].append(
                    {
                        "target": ts_slice,
                        "start": unsplit_start,
                        "feat_static_cat": train_entry["feat_static_cat"],
                    }
                )
                # unsplit_start += pd.Timedelta(hours=prediction_length*hours_factor[i])
                unsplit_start += pd.Timedelta(hours=prediction_length)
    # for j in range(len(dataset_group)):
    #    print(len(dataset_group[i]))
    # import pdb;pdb.set_trace()
    print(len(whole_data))
    ret["group_ratio"] = [len(i) / len(whole_data) for i in dataset_group]
    print(ret["group_ratio"])
    random.shuffle(whole_data)
    ret["whole_data"] = ListDataset(whole_data, freq=dataset.metadata.freq)
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq=dataset.metadata.freq))
    ret["group_data"] = group_data_list
    print("write whole data")
    with open("synthetic_" + file_name + "_whole_data.csv", "wb") as output:
        pickle.dump(ret["whole_data"], output)
    print("write group data")
    with open("synthetic_" + file_name + "_group_data.csv", "wb") as output:
        pickle.dump(ret, output)
    return True


def get_temperature_data(
    context_length=24, prediction_length=4, samples_per_ts=2000, num_groups=8
):
    ts_file = pd.read_csv("temperature.csv")
    city_names = [
        "Vancouver",
        "Los Angeles",
        "Las Vegas",
        "San Diego",
        "Philadelphia",
        "Montreal",
        "Boston",
        "Haifa",
    ]
    datetime = ts_file["datetime"]
    dataset_group = [[] for i in range(num_groups)]
    whole_data = []
    ret = dict()
    for gid in range(num_groups):
        ts = ts_file[city_names[gid]]
        num_samples = 0
        index = 0
        while True:
            num_samples += 1
            index += 1
            ts_slice = torch.tensor(
                ts[index : index + context_length + prediction_length].values
            )
            nu = 1 + sum(ts_slice) / len(ts_slice)
            ts_slice /= nu
            if torch.sum(torch.isnan(ts_slice)).item() == 0:
                dataset_group[gid].append(
                    {
                        "target": ts_slice,
                        "start": pd.Timestamp(datetime[index]),
                    }
                )
                whole_data.append(
                    {
                        "target": ts_slice,
                        "start": pd.Timestamp(datetime[index]),
                    }
                )
            if num_samples == samples_per_ts:
                break
    random.shuffle(whole_data)
    ret["whole_data"] = ListDataset(whole_data, freq="1H")
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq="1H"))
    ret["group_data"] = group_data_list
    print("write whole data")
    with open("synthetic_temperature_whole_data.csv", "wb") as output:
        pickle.dump(ret["whole_data"], output)
    print("write group data")
    with open("synthetic_temperature_group_data.csv", "wb") as output:
        pickle.dump(ret, output)
    return True


def get_amazon_sales():
    f = open("./dataset/sgc_train.json", encoding="utf-8")
    dataset_group = [[] for i in range(8)]
    whole_data = []
    ret = dict()
    X = []
    Y = []
    # split_grid = [0.04, 0.1, 0.25, 0.5, 1, 10, 100, 5000]
    for line in f.readlines():
        # gid = 0
        dic = json.loads(line)
        var = torch.var(torch.FloatTensor(dic["target"])).item()
        if var > 5000:
            continue
        nu = 1 + sum(dic["target"]) / len(dic["target"])
        ts = [i / nu for i in dic["target"]]
        start = dic["start"]
        # ts = dic['target']
        if len(ts) < 28:
            continue
        X.append((ts, var))
    X = sorted(X, key=lambda x: x[1])
    X = [x[0] for x in X]
    length = int(len(X) / 8)
    for gid in range(8):
        for j in range(gid * length, (gid + 1) * length):
            whole_data.append({"target": X[j], "start": start})
            dataset_group[gid].append({"target": X[j], "start": start})
    random.shuffle(whole_data)
    ret["whole_data"] = ListDataset(whole_data, freq="1H")
    group_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_list.append(ListDataset(group, freq="1H"))
    ret["group_data"] = group_list
    print("write whole data")
    with open("synthetic_sales_time_whole_data.csv", "wb") as output:
        pickle.dump(ret["whole_data"], output)
    print("write group data")
    with open("synthetic_sales_time_group_data.csv", "wb") as output:
        pickle.dump(ret, output)
    return True


def get_group_data_by_var(name, num_groups, len_sample=9):
    dataset = get_dataset(name)
    dataset_group = [[] for i in range(num_groups)]
    whole_data = []
    ret = []
    it = iter(dataset.train)
    num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    group_boundary = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
    for i in range(num_ts):
        train_entry = next(it)
        unsplit_ts = train_entry["target"][0:800]
        unsplit_start = train_entry["start"]
        whole_data.append({"target": unsplit_ts, "start": unsplit_start})
        for ts_sample_start in range(len(unsplit_ts) - len_sample):
            group_id = 0
            print(
                torch.var(
                    torch.FloatTensor(
                        unsplit_ts[
                            ts_sample_start : ts_sample_start + len_sample
                        ]
                    )
                )
            )
            continue
            dataset_group[group_id].append(
                {
                    "target": unsplit_ts[
                        ts_sample_start : ts_sample_start + len_sample
                    ],
                    "start": unsplit_start,
                }
            )
            unsplit_start += pd.Timedelta(hours=1)
    import pdb

    pdb.set_trace()
    random.shuffle(whole_data)
    print("append once")
    ret.append(ListDataset(whole_data, freq=dataset.metadata.freq))
    print("append twice")
    ret.append(ListDataset(whole_data, freq=dataset.metadata.freq))
    print("append data")
    for group in dataset_group:
        random.shuffle(group)
        ret.append(ListDataset(group, freq=dataset.metadata.freq))
    print("write whole data")
    with open("synthetic_traffic_time_whole_data.csv", "wb") as output:
        pickle.dump(ret[0:2], output)
    print("write group data")
    with open("synthetic_traffic_time_group_data.csv", "wb") as output:
        pickle.dump(ret, output)
    return True


def get_group_data_by_hash(name, q, num_groups):
    dataset = get_dataset(name)
    dataset_group = [[] for i in range(num_groups)]
    whole_data = []
    ret = []
    it = iter(dataset.train)
    num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    for i in range(num_ts):
        train_entry = next(it)
        dataset_group[i % num_groups].append(
            {"target": train_entry["target"], "start": train_entry["start"]}
        )
        whole_data.append(
            {"target": train_entry["target"], "start": train_entry["start"]}
        )
    random.shuffle(whole_data)
    ret.append(ListDataset(whole_data, freq=dataset.metadata.freq))
    for group in dataset_group:
        ret.append(ListDataset(group, freq=dataset.metadata.freq))
    return ret, dataset.metadata.freq


def get_group_data_by_duplicate(name, num_duplicates, num_groups):
    dataset = get_dataset(name)
    dataset_group = [[] for i in range(num_groups)]
    whole_data_list = []
    no_duplicate_whole_data_list = []
    ret = []
    it = iter(dataset.train)
    num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    for i in range(num_ts):
        train_entry = next(it)
        no_duplicate_whole_data_list.append(
            {"target": train_entry["target"], "start": train_entry["start"]}
        )
        for j in range(num_duplicates):
            dataset_group[i % num_groups].append(
                {
                    "target": train_entry["target"],
                    "start": train_entry["start"],
                }
            )
            whole_data_list.append(
                {
                    "target": train_entry["target"],
                    "start": train_entry["start"],
                }
            )
    random.shuffle(whole_data_list)
    random.shuffle(no_duplicate_whole_data_list)
    ret.append(
        ListDataset(no_duplicate_whole_data_list, freq=dataset.metadata.freq)
    )
    ret.append(ListDataset(whole_data_list, freq=dataset.metadata.freq))
    for group in dataset_group:
        random.shuffle(group)
        ret.append(ListDataset(group, freq=dataset.metadata.freq))
    return ret, dataset.metadata.freq


def get_whole_data_by_duplicate(name, num_duplicates):
    dataset = get_dataset(name)
    dataset_group = []
    ret = []
    no_duplicate_whole_data_list = []
    it = iter(dataset.train)
    num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    for i in range(num_ts):
        train_entry = next(it)
        no_duplicate_whole_data_list.append(
            {"target": train_entry["target"], "start": train_entry["start"]}
        )
        for j in range(num_duplicates):
            dataset_group.append(
                {
                    "target": train_entry["target"],
                    "start": train_entry["start"],
                }
            )
    random.shuffle(dataset_group)
    random.shuffle(no_duplicate_whole_data_list)
    ret.append(
        ListDataset(no_duplicate_whole_data_list, freq=dataset.metadata.freq)
    )
    ret.append(ListDataset(dataset_group, freq=dataset.metadata.freq))
    return ret, dataset.metadata.freq


def get_group_data(name):
    dataset = get_dataset(name)
    dataset_group = []
    it = iter(dataset.train)
    num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    for i in range(num_ts):
        train_entry = next(it)
        dataset_group.append(
            ListDataset(
                [
                    {
                        "target": train_entry["target"],
                        "start": train_entry["start"],
                    }
                ],
                freq=dataset.metadata.freq,
            )
        )
    return dataset_group


def get_whole_data(name):
    dataset = get_dataset(name)
    dataset_group = []
    it = iter(dataset.train)
    num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    for i in range(num_ts):
        train_entry = next(it)
        dataset_group.append(
            {"target": train_entry["target"], "start": train_entry["start"]}
        )
    return ListDataset(dataset_group, freq=dataset.metadata.freq)


def get_synthetic_data(model_name=None, num_groups=8, mean_boundary=1):
    assert num_groups > 1
    prediction_length = 1
    context_length = 5
    num_time_steps = 1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    net = SimpleFeedForwardEstimator(
        freq="1H",
        prediction_length=prediction_length,
        context_length=context_length,
    ).create_training_network(device)
    delta = 2 * mean_boundary / num_groups
    dataset_group = []
    whole_data_list = []
    start = pd.Timestamp("01-01-2019", freq="1H")
    for gid in range(num_groups):
        parameter_mean = -mean_boundary + gid * delta
        # change the parameters of the model
        for p in net.parameters():
            p.data = torch.normal(parameter_mean, 0.1, size=p.data.shape)
        ts = torch.normal(0, 0.1, size=(1, context_length))
        for num_ts in range(num_time_steps):
            ts_slice = torch.Tensor(ts[0][-context_length:]).view(
                1, context_length
            )
            prediction = net.get_distr(ts_slice).sample((5000,))
            prediction = sum(prediction) / len(prediction)
            ts = torch.cat([ts, prediction], dim=1)
        whole_data_list.append(
            {
                "target": ts.view(
                    len(ts[0]),
                )[context_length:],
                "start": start,
            }
        )
        dataset_group.append(
            ListDataset(
                [
                    {
                        "target": ts.view(
                            len(ts[0]),
                        )[context_length:],
                        "start": start,
                    }
                ],
                freq="1H",
            )
        )
    dataset = ListDataset(whole_data_list, freq="1H")
    dataset_group = [dataset] + dataset_group

    # save to files
    with open("synthetic_whole_data.csv", "wb") as output:
        pickle.dump(dataset, output)

    with open("synthetic_group_data.csv", "wb") as output:
        pickle.dump(dataset_group, output)
    return True


def get_synthetic_data_mlp(
    model_name=None, num_groups=8, mean_boundary=0.5, num_duplicates=16
):
    assert num_groups > 1
    prediction_length = 1
    context_length = 12
    device = "cpu"
    dataset_group = []
    whole_data_list = []
    start = pd.Timestamp("01-01-2019", freq="1H")
    for gid in range(num_groups):
        net = SimpleFeedForwardEstimator(
            freq="1H",
            prediction_length=prediction_length,
            context_length=context_length,
        ).create_training_network(device)
        for p in net.parameters():
            p.data = torch.normal(0, 0.1, size=p.data.shape)
        pattern_group = []
        # for j in range(num_duplicates):
        # ts = torch.Uniform(0, 1, size=(1, context_length))
        while True:
            ts = torch.rand(size=(1, context_length))
            ts_slice = torch.Tensor(ts[0][-context_length:]).view(
                1, context_length
            )
            prediction = net.get_distr(ts_slice).sample((1000,))
            prediction = sum(prediction) / len(prediction)
            if abs(torch.norm(prediction)) <= 1:
                break
        ts = torch.cat([ts, prediction], dim=1)
        ts = ts.view(
            len(ts[0]),
        )  # [context_length:]
        for j in range(num_duplicates):
            ts_sample = ts + torch.normal(0, 0.1, size=ts.shape)
            whole_data_list.append({"target": ts_sample, "start": start})
            pattern_group.append({"target": ts_sample, "start": start})
        dataset_group.append(ListDataset(pattern_group, freq="1H"))
    random.shuffle(whole_data_list)
    random.shuffle(dataset_group)
    dataset = ListDataset(whole_data_list, freq="1H")
    ret = []
    ret.append(dataset)
    ret.append(dataset)
    dataset_group = [dataset] + dataset_group
    dataset_group = [dataset] + dataset_group

    # save to files
    with open("synthetic_mlp_whole_data.csv", "wb") as output:
        pickle.dump(ret, output)

    with open("synthetic_mlp_group_data.csv", "wb") as output:
        pickle.dump(dataset_group, output)
    return True


def get_synthetic_data_linear(
    context_length=24,
    prediction_length=8,
    num_groups=8,
    steps_per_ts=1,
    num_duplicates=16,
):
    assert num_groups > 1
    freq = "1H"
    len_sample = context_length + prediction_length

    dataset_group = []
    whole_data = []
    ret = dict()
    start = pd.Timestamp("01-01-2000", freq=freq)
    for gid in range(num_groups):
        model1 = torch.nn.Linear(context_length, prediction_length)
        model2 = torch.nn.Linear(context_length, prediction_length)
        # model1 = torch.sin
        # model2 = torch.cos
        pattern_group1 = []
        pattern_group2 = []
        sample_context = torch.rand(context_length)
        for t_step in range(2 * steps_per_ts):
            while True:
                with torch.no_grad():
                    if t_step <= steps_per_ts:
                        prediction = model1(sample_context)
                    else:
                        prediction = model2(sample_context)
                    if (
                        torch.norm(prediction) < prediction_length
                    ):  # and prediction_length*0.1 < torch.norm(prediction):
                        # prediction = torch.sin(prediction)
                        # prediction /= torch.max(prediction)
                        break
                    # prediction *= 10
                    # prediction += torch.normal(0, 0.1, size=prediction.shape)
            ts_sample = torch.cat([sample_context, prediction])
            # print(ts_sample)
            for j in range(num_duplicates):
                ts_sample += torch.normal(0, 0.1, size=ts_sample.shape)
                whole_data.append({"target": ts_sample, "start": start})
                if t_step <= steps_per_ts:
                    pattern_group1.append(
                        {"target": ts_sample, "start": start}
                    )
                else:
                    pattern_group2.append(
                        {"target": ts_sample, "start": start}
                    )
            sample_context = ts_sample[-context_length:]
            start += pd.Timedelta(hours=prediction_length)
        dataset_group.append(ListDataset(pattern_group1, freq=freq))
        """
        dataset_group.append(
            ListDataset(
                pattern_group2,
                freq=freq
            )
        )
        """
    print(len(whole_data))
    ret["group_ratio"] = [len(i) / len(whole_data) for i in dataset_group]
    print(ret["group_ratio"])
    random.shuffle(whole_data)
    ret["whole_data"] = ListDataset(whole_data, freq=freq)
    ret["group_data"] = dataset_group

    # save to files
    with open("synthetic_linear_new_whole_data.csv", "wb") as output:
        pickle.dump(ret["whole_data"], output)

    with open("synthetic_linear_new_group_data.csv", "wb") as output:
        pickle.dump(ret, output)

    return True


def get_synthetic_data_linear_simple(
    model_name=None, num_groups=8, mean_boundary=1, num_duplicates=16
):
    assert num_groups > 1
    num_time_steps = 100

    dataset_group = []
    whole_data_list = []
    start = pd.Timestamp("01-01-2019", freq="1D")
    for gid in range(num_groups):
        base = np.linspace(0, 10, num_time_steps)
        pattern_group = []
        ts = (
            (gid + 1)
            * mean_boundary
            * torch.FloatTensor(base).view(1, num_time_steps)
        )
        for j in range(num_duplicates):
            ts += torch.normal(0, 0.01, size=ts.shape)
            whole_data_list.append(
                {
                    "target": ts.view(
                        len(ts[0]),
                    ),
                    "start": start,
                }
            )
            pattern_group.append(
                {
                    "target": ts.view(
                        len(ts[0]),
                    ),
                    "start": start,
                }
            )
        dataset_group.append(ListDataset(pattern_group, freq="1D"))

    random.shuffle(whole_data_list)

    dataset = ListDataset(whole_data_list, freq="1D")
    dataset_group = [dataset] + dataset_group

    # save to files
    with open("synthetic_linear_simple_whole_data.csv", "wb") as output:
        pickle.dump(dataset, output)

    with open("synthetic_linear_simple_group_data.csv", "wb") as output:
        pickle.dump(dataset_group, output)
    return True


def get_synthetic_data_sin(
    model_name=None, num_groups=32, mean_boundary=1, num_duplicates=50
):
    assert num_groups > 1
    num_time_steps = 100

    dataset_group = []
    whole_data_list = []
    no_duplicate_whole_data_list = []
    start = pd.Timestamp("01-01-2019", freq="1D")
    for gid in range(num_groups):
        mean = (gid + 1) * mean_boundary
        base = np.linspace(0, mean, num_time_steps)
        pattern_group = []
        ts = (gid + 1) * torch.sin(torch.FloatTensor(base)).view(
            1, num_time_steps
        )
        ts += torch.FloatTensor((gid + 1) * base).view(1, num_time_steps)
        no_duplicate_whole_data_list.append(
            {
                "target": ts.view(
                    len(ts[0]),
                ),
                "start": start,
            }
        )
        for j in range(num_duplicates):
            ts += torch.normal(0, 0.1, size=ts.shape)
            whole_data_list.append(
                {
                    "target": ts.view(
                        len(ts[0]),
                    ),
                    "start": start,
                }
            )
            pattern_group.append(
                {
                    "target": ts.view(
                        len(ts[0]),
                    ),
                    "start": start,
                }
            )
        dataset_group.append(ListDataset(pattern_group, freq="1D"))

    random.shuffle(whole_data_list)
    random.shuffle(no_duplicate_whole_data_list)
    ret_whole_dataset = []
    dataset = ListDataset(whole_data_list, freq="1D")
    no_duplicate_dataset = ListDataset(no_duplicate_whole_data_list, freq="1D")
    ret_whole_dataset.append(dataset)
    ret_whole_dataset.append(dataset)
    dataset_group = [dataset] + dataset_group
    dataset_group = [dataset] + dataset_group

    # save to files
    with open("synthetic_complexsin_whole_data.csv", "wb") as output:
        pickle.dump(ret_whole_dataset, output)

    with open("synthetic_complexsin_group_data.csv", "wb") as output:
        pickle.dump(dataset_group, output)
    return True
