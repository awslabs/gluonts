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

from typing import Tuple

import numpy as np


def unlist(l):
    if (
        type(l).__name__.endswith("Vector")
        and type(l).__name__ != "ListVector"
    ):
        return [unlist(x) for x in l]
    elif type(l).__name__ == "ListVector":
        return [unlist(x) for x in l]
    elif type(l).__name__ == "Matrix":
        return np.array(l)
    else:
        return l


def interval_to_quantile_level(interval_level: int, side: str) -> float:
    """Convert a prediction interval level (upper or lower) into a quantile level."""
    if side == "upper":
        level = 50 + interval_level / 2
    elif side == "lower":
        level = 50 - interval_level / 2
    else:
        raise ValueError(side)
    return level / 100


def quantile_to_interval_level(quantile_level: float) -> Tuple[int, str]:
    """Convert a quantile level into a prediction interval level (upper or lower)."""
    side = "upper" if quantile_level >= 0.5 else "lower"
    return round(200 * abs(0.5 - quantile_level)), side
