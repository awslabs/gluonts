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
import time
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor


def KMeans(x, K=10, Niter=10000, verbose=True):
    """
    Implements Lloyd's algorithm for the Euclidean metric.
    """
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        # print(D_ij.argmin(dim=1))
        # import pdb;pdb.set_trace()
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if (
        verbose
    ):  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension"
            f" {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def _get_pre_features(x):
    df = pd.DataFrame(x.numpy())
    df = df.rename(
        columns={
            0: "mean",
            1: "var",
            2: "week_id",
            3: "month_id",
        }
    )
    df["mean"] = (df["mean"] - df["mean"].min()) / (
        df["mean"].max() - df["mean"].min()
    )
    df["var"] = (df["var"] - df["var"].min()) / (
        df["var"].max() - df["var"].min()
    )
    df = pd.get_dummies(df, columns=["week_id", "month_id"], dummy_na=True)
    return torch.from_numpy(df.to_numpy())


def KMeans_inside_dataset(
    num_ts_=1,
    num_groups=16,
    context_length=72,
    prediction_length=24,
    file_name="default",
):
    dataset = get_dataset("traffic")
    dataset_group = [[] for i in range(num_groups)]
    whole_data = []
    ret = dict()
    it = iter(dataset.train)
    # num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    num_ts = num_ts_
    len_sample = context_length + prediction_length
    index = 0
    feature = torch.Tensor([])
    for i in range(num_ts):
        train_entry = next(it)
        target = train_entry["target"]

        for ts_sample_start in range(
            0, len(target) - len_sample, prediction_length
        ):
            ts_slice = target[ts_sample_start : ts_sample_start + len_sample]
            feature = torch.cat(
                (
                    feature,
                    torch.Tensor(
                        [
                            ts_slice.mean(),
                            ts_slice.var(),
                            index % 7,
                            index // 90,
                        ]
                    ),
                )
            )
            index += 1
    feature = feature.reshape(index, 4)
    feature = _get_pre_features(feature).contiguous()
    # print(feature)
    # import pdb;pdb.set_trace()
    cl, c = KMeans(feature, num_groups)
    it = iter(dataset.train)
    sample_id = 0
    for i in range(num_ts):
        train_entry = next(it)
        target = train_entry["target"]
        unsplit_start = train_entry["start"]
        for ts_sample_start in range(
            0, len(target) - len_sample, prediction_length
        ):
            ts_slice = target[ts_sample_start : ts_sample_start + len_sample]
            gid = cl[sample_id]
            dataset_group[gid].append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            whole_data.append(
                {
                    "target": ts_slice,
                    "start": unsplit_start,
                    "feat_static_cat": train_entry["feat_static_cat"],
                }
            )
            unsplit_start += pd.Timedelta(hours=prediction_length)
            sample_id += 1
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


def KMeans_m5_dataset(
    num_ts_=1,
    num_groups=16,
    context_length=72,
    prediction_length=24,
    file_name="default",
):
    df = pd.read_csv("sales_train_evaluation.csv")
    dataset_group = [[] for i in range(num_groups)]
    whole_data = []
    ret = dict()
    # num_ts = int(dataset.metadata.feat_static_cat[0].cardinality)
    num_ts = num_ts_
    len_sample = context_length + prediction_length
    # compute mean and variance
    df = df.iloc[:num_ts, :]
    df["mean"] = df.iloc[:, 1947 - len_sample : 1947].mean(axis=1)
    df["var"] = df.iloc[:, 1947 - len_sample : 1947].var(axis=1)
    df["mean"] = (df["mean"] - df["mean"].min()) / (
        df["mean"].max() - df["mean"].min()
    )
    df["var"] = (df["var"] - df["var"].min()) / (
        df["var"].max() - df["var"].min()
    )
    df_feature = df.iloc[:, 2:6]
    df_feature = pd.get_dummies(df_feature, dummy_na=True)
    df_feature = pd.concat([df_feature, df.iloc[:, -2]], axis=1)
    feature = torch.from_numpy(df_feature.to_numpy()).contiguous()
    cl, c = KMeans(feature, num_groups)
    # print(cl)
    # import pdb;pdb.set_trace()
    sample_id = 0
    for i in range(num_ts):
        ts_slice = df.iloc[i : i + 1, 1947 - len_sample : 1947]
        ts_slice = torch.from_numpy(ts_slice.to_numpy())[0]
        # print(ts_slice)
        # import pdb;pdb.set_trace()
        gid = cl[sample_id]
        unsplit_start = pd.Timestamp("1990-01-01")
        dataset_group[gid].append(
            {
                "target": ts_slice,
                "start": unsplit_start,
            }  # , 'feat_static_cat': train_entry['feat_static_cat']}
        )
        whole_data.append(
            {
                "target": ts_slice,
                "start": unsplit_start,
            }  # , 'feat_static_cat': train_entry['feat_static_cat']}
        )
        sample_id += 1
    print(len(whole_data))
    ret["group_ratio"] = [len(i) / len(whole_data) for i in dataset_group]
    print(ret["group_ratio"])
    random.shuffle(whole_data)
    ret["whole_data"] = ListDataset(whole_data, freq="1H")
    group_data_list = []
    for group in dataset_group:
        random.shuffle(group)
        group_data_list.append(ListDataset(group, freq="1H"))
    ret["group_data"] = group_data_list
    print("write whole data")
    with open("synthetic_" + file_name + "_whole_data.csv", "wb") as output:
        pickle.dump(ret["whole_data"], output)
    print("write group data")
    with open("synthetic_" + file_name + "_group_data.csv", "wb") as output:
        pickle.dump(ret, output)
    return True
