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

from functools import partial

from gluonts.dataset.repository.datasets import dataset_recipes

from ._m5 import generate_pts_m5_dataset

dataset_recipes["pts_m5"] = partial(
    generate_pts_m5_dataset, pandas_freq="D", prediction_length=28
)
