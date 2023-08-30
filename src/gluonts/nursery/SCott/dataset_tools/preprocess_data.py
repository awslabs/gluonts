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

from traffic import *
from synthetic import *
from exchange_rate import *
from electricity import *

# from algo_clustering import *

# Please uncomment the following four commands to generate the corresponding dataset.
# The four lines corresponds to sythetic, traffic, exchange rate, electricity datasets, respectively.

# ===================================================================================================
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
# ===================================================================================================


# (Ignore) the following are auxiliary functions for other datasets that are not included in the paper

# ===================================================================================================
# KMeans_inside_dataset(num_ts_=10, num_groups=32, file_name='traffic_toy32')
# KMeans_m5_dataset(num_ts_=500, num_groups=8, file_name='m5_10000')
# get_m4_by_freq(context_length=72, prediction_length=24, len_per_ts=200, num_ts=50)
# get_temperature_data(context_length=12, prediction_length=4, samples_per_ts=1000)
# ===================================================================================================
