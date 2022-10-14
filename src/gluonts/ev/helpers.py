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

from typing import Dict, Optional
import numpy as np


EvalData = Dict[str, np.ndarray]


def axis_is_zero_or_none(axis: Optional[int]) -> bool:
    return axis == 0 or axis is None


def create_eval_data(
    inputs: np.ndarray, labels: np.ndarray, forecasts: dict[np.ndarray]
):
    return {
        "input": np.stack([entry["target"] for entry in inputs]),
        "label": np.stack([entry["target"] for entry in labels]),
        **{name: np.stack(value) for name, value in forecasts.items()},
    }
