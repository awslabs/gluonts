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

import numpy as np

# Workaround needed due to a known issue with np.quantile(inp, quant) returning unsorted values.
# We fix this by ensuring that the obtained bin_centers are monotonically increasing.
# Tracked in the following issues:
# - https://github.com/numpy/numpy/issues/14685
# - https://github.com/numpy/numpy/issues/12282
def ensure_binning_monotonicity(bin_centers: np.ndarray):
    for i in range(1, len(bin_centers)):
        if bin_centers[i] < bin_centers[i - 1]:
            bin_centers[i] = bin_centers[i - 1]
    return bin_centers


def bin_edges_from_bin_centers(bin_centers: np.ndarray):
    lower_edge = -np.inf
    upper_edge = np.inf
    bin_edges = np.concatenate(
        [
            [lower_edge],
            (bin_centers[1:] + bin_centers[:-1]) / 2.0,
            [upper_edge],
        ]
    )
    return bin_edges
