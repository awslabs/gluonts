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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gluonts.dataset import Dataset


@dataclass
class HierarchicalDataset:
    data: Dataset
    S: np.ndarray

    def __iter__(self):
        for entry in self.data:
            entry = entry.copy()
            entry["target"] = self.S @ np.array(entry["target"])

            yield entry

    def __len__(self) -> int:
        return len(self.data)
