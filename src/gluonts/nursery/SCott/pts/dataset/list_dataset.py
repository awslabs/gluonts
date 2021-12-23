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


import random
from typing import Iterable

from .common import DataEntry, Dataset, SourceContext
from .process import ProcessDataEntry


class ListDataset(Dataset):
    def __init__(
        self,
        data_iter: Iterable[DataEntry],
        freq: str,
        one_dim_target: bool = True,
        shuffle: bool = False,
    ) -> None:
        process = ProcessDataEntry(freq, one_dim_target)
        self.list_data = [process(data) for data in data_iter]
        if shuffle:
            random.shuffle(self.list_data)

    def __iter__(self):
        source_name = "list_data"
        for row_number, data in enumerate(self.list_data, start=1):
            data["source"] = SourceContext(source=source_name, row=row_number)
            yield data

    def __len__(self):
        return len(self.list_data)
