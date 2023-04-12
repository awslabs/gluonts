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

import pickle
import numpy as np
import math


def create_table(path):
    objects = []
    if "adv" in path:
        types = "adv"
    elif "_rs" in path:
        types = "rs"
    elif "rt" in path:
        types = "da"
    else:
        types = "no_defense"
    print(types)
    with open(path, "rb") as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    mse = []
    mape = []
    wql = []
    sparsity = []

    result = objects[0]
    batch = len(result.mse["1"])
    c = 1.96 / math.sqrt(batch)

    for result in objects:
        print("mse loss:")
        for key in result.mse.keys():
            sparsity.append(key)
            print(
                key,
                np.asarray(result.mse[key]).mean(),
                "+-",
                np.asarray(result.mse[key]).std() * c,
            )
            mse.append(
                (
                    np.asarray(result.mse[key]).mean(),
                    np.asarray(result.mse[key]).std() * c,
                )
            )

        print("mape loss:")
        for key in result.mape.keys():
            print(
                key,
                np.asarray(result.mape[key]).mean(),
                "+-",
                np.asarray(result.mape[key]).std() * c,
            )
            mape.append(
                (
                    np.asarray(result.mape[key]).mean(),
                    np.asarray(result.mape[key]).std() * c,
                )
            )

        print("wQL:")
        for key in result.ql.keys():
            print(
                key,
                np.asarray(result.ql[key]).mean(),
                "+-",
                np.asarray(result.ql[key]).std() * c,
            )
            wql.append(
                (
                    np.asarray(result.ql[key]).mean(),
                    np.asarray(result.ql[key]).std() * c,
                )
            )

    with open("table_" + types + ".txt", "w") as f:
        for i in range(len(mse)):
            # with std
            print(
                sparsity[i]
                + " & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f}\\\\".format(
                    mape[i][0],
                    mape[i][1],
                    mse[i][0],
                    mse[i][1],
                    wql[i][0],
                    wql[i][1],
                ),
                file=f,
            )
            # without std
            # print(sparsity[i] + ' & {:.4f} & {:.4f} & {:.4f}\\\\'.format(mape[i][0], mse[i][0], wql[i][0]), file=f)
