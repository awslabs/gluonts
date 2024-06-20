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
import matplotlib.pyplot as plt
import gluonts
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

print(f"Available datasets: {list(dataset_recipes.keys())}")

dataset = get_dataset("electricity")

print("new")

entry = next(iter(dataset.train))

train_series = to_pandas(entry)
train_series.plot()
plt.grid(which="both")
plt.legend(["test series"], loc="upper left")
plt.show()
print("new")


import torch
from gluonts.dataset.repository.datasets import get_dataset

import gluonts
