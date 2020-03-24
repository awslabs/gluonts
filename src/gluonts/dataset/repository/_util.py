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

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np


def to_dict(
    target_values: np.ndarray,
    start: str,
    cat: Optional[List[int]] = None,
    item_id: Optional[Any] = None,
):
    def serialize(x):
        if np.isnan(x):
            return "NaN"
        else:
            # return x
            return float("{0:.6f}".format(float(x)))

    res = {
        "start": str(start),
        "target": [serialize(x) for x in target_values],
    }

    if cat is not None:
        res["feat_static_cat"] = cat

    if item_id is not None:
        res["item_id"] = item_id

    return res


def save_to_file(path: Path, data: List[Dict]):
    print(f"saving time-series into {path}")
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    with open(path, "wb") as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode("utf-8"))


def metadata(
    cardinality: Union[int, List[int]], freq: str, prediction_length: int
):
    return {
        "freq": freq,
        "prediction_length": prediction_length,
        "feat_static_cat": [
            {"name": "feat_static_cat", "cardinality": str(cardinality)}
        ],
    }
