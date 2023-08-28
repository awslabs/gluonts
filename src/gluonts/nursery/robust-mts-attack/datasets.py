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

from multivariate.datasets.dataset import (
    electricity,
    exchange_rate,
    solar,
    traffic,
    wiki,
    taxi_30min,
)

DATASETS = ["electricity", "exchange_rate", "solar", "traffic", "wiki", "taxi"]


def get_dataset(dataset, max_target_dim):
    if dataset == "electricity":
        return electricity(max_target_dim=max_target_dim)
    elif dataset == "exchange_rate":
        return exchange_rate(max_target_dim=max_target_dim)
    elif dataset == "solar":
        return solar(max_target_dim=max_target_dim)
    elif dataset == "traffic":
        return traffic(max_target_dim=max_target_dim)
    elif dataset == "wiki":
        return wiki(max_target_dim=max_target_dim)
    elif dataset == "taxi":
        return taxi_30min(max_target_dim=max_target_dim)
