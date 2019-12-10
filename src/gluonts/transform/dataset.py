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


from typing import List, Iterator

from gluonts.dataset.common import Dataset, DataEntry
from gluonts.transform import Chain, Transformation


class TransformedDataset(Dataset):
    """
    A dataset that corresponds to applying a list of transformations to each
    element in the base_dataset.
    This only supports SimpleTransformations, which do the same thing at
    prediction and training time.


    Parameters
    ----------
    base_dataset
        Dataset to transform
    transformations
        List of transformations to apply
    """

    def __init__(
        self, base_dataset: Dataset, transformations: List[Transformation]
    ) -> None:
        self.base_dataset = base_dataset
        self.transformations = Chain(transformations)

    def __iter__(self) -> Iterator[DataEntry]:
        yield from self.transformations(self.base_dataset, is_train=True)

    def __len__(self):
        return sum(1 for _ in self)
