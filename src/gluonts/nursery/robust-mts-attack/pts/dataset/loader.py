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

from typing import Optional
import itertools

from torch.utils.data import IterableDataset

from gluonts.dataset.common import Dataset
from gluonts.transform import Transformation, TransformedDataset
from gluonts.itertools import Cyclic, PseudoShuffled, Cached


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length

        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if not cache_data else Cached(Cyclic(dataset)),
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        else:
            shuffled = PseudoShuffled(
                self.transformed_dataset,
                shuffle_buffer_length=self.shuffle_buffer_length,
            )
            return iter(shuffled)
