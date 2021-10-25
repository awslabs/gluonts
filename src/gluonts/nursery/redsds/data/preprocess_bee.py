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

import os
import numpy as np


def preprocess_seq(seq_path):
    mapping = {"waggle": 0, "turn_right": 1, "turn_left": 2}
    x_btf = os.path.join(seq_path, "btf/ximage.btf")
    x = np.loadtxt(x_btf)
    x = (x - x.mean(0)) / x.std(0)
    y_btf = os.path.join(seq_path, "btf/yimage.btf")
    y = np.loadtxt(y_btf)
    y = (y - y.mean(0)) / y.std(0)
    theta_btf = os.path.join(seq_path, "btf/timage.btf")
    theta = np.loadtxt(theta_btf)
    label_btf = os.path.join(seq_path, "btf/label0.btf")
    labels = np.loadtxt(label_btf, dtype=str).tolist()
    labels = np.array([mapping[lab] for lab in labels])
    observations = np.stack([x, y, np.sin(theta), np.cos(theta)], 1)
    return observations, labels


def preprocess(seq_paths, npz_path):
    data_y = []
    data_z = []
    ctx = 120
    count = 0
    for seq_path in seq_paths:
        observations, labels = preprocess_seq(seq_path)
        cp = np.where(np.concatenate([[1], np.diff(labels)]) != 0)[0]
        for c in cp:
            if labels[c : c + ctx].shape[0] == ctx:
                data_y.append(observations[c : c + ctx])
                data_z.append(labels[c : c + ctx])
                count += 1
    data_y = np.stack(data_y, 0)
    data_z = np.stack(data_z, 0)
    np.savez(npz_path, y=data_y, z=data_z)


if __name__ == "__main__":
    base_path = "./psslds/zips"
    train_seqs = [
        os.path.join(base_path, "data/sequence%d") % i for i in [1, 3, 4, 5, 6]
    ]
    preprocess(train_seqs, "bee.npz")
    test_seqs = [os.path.join(base_path, "data/sequence%d") % i for i in [2]]
    preprocess(test_seqs, "bee_test.npz")
