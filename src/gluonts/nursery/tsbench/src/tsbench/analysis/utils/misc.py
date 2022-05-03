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

from collections import defaultdict
from typing import Dict, List
from tsbench.config import ModelConfig


def union_dicts(
    dicts: List[Dict[str, ModelConfig]]
) -> Dict[str, List[ModelConfig]]:
    """
    Merges the dicts by aggregating model configurations with the same key into
    a list.
    """
    result = defaultdict(list)
    for item in dicts:
        for k, v in item.items():
            result[k].append(v)
    return result
