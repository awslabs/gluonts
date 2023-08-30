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

from typing import List

import numpy as np


def stack(data):
    if isinstance(data[0], np.ndarray):
        return np.array(data)

    elif isinstance(data[0], (list, tuple)):
        return list(map(stack, zip(*data)))

    return data


def batchify(data: List[dict]) -> dict:
    keys = data[0].keys()
    return {key: stack(data=[item[key] for item in data]) for key in keys}
