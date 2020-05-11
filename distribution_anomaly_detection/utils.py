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

# Standard library imports
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

# Third-party imports
import mxnet as mx

from gluonts import transform
from gluonts.dataset.common import Dataset
from gluonts.transform import TransformedDataset


def get_ds_list(dataset: Dataset, detection_length):
    """
    Constructs list of dataset obtained through a sliding window, such that the targets are
    [ dataset.target[...,:T-detection_length], ...,  dataset.target[...,:T-1]]

    Parameters:
    ----------
    dataset
        Gluonts dataset, the field target will be scliced with sliding window
    detection_length
        number of sliced_datasets to return

    Returns:
    --------
    list_ds
        List of datasets
    """

    def truncate_target(data, remove_n_item):
        data = data.copy()
        target = data["target"]
        assert target.shape[-1] >= remove_n_item
        data["target"] = target[..., :-remove_n_item]
        return data

    list_ds = []
    for k in range(detection_length, 0, -1):
        list_ds += list(
            TransformedDataset(
                dataset,
                transformations=[
                    transform.AdhocTransform(lambda ds: truncate_target(ds, k))
                ],
            )
        )

    return list_ds
