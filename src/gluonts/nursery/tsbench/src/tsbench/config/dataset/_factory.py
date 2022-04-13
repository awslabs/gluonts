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

from pathlib import Path
from typing import Dict, Type, TypeVar, Union
from tsbench.constants import DEFAULT_DATA_PATH
from ._base import DatasetConfig

DATASET_REGISTRY: Dict[str, Type[DatasetConfig]] = {}


D = TypeVar("D", bound=Type[DatasetConfig])


def register_dataset(cls: D) -> D:
    """
    Registers the provided class in the global dataset registry.
    """
    DATASET_REGISTRY[cls.name()] = cls
    return cls


def get_dataset_config(
    name: str, path: Union[Path, str] = DEFAULT_DATA_PATH
) -> DatasetConfig:
    """
    This method creates the dataset configuration of the model with the
    specified name.

    Args:
        name: The canonical name of the dataset. See `DATASET_REGISTRY`.
        path: The root of the dataset directory.

    Returns:
        The dataset configuration.
    """
    # Get the dataset
    assert name in DATASET_REGISTRY, f"Dataset name '{name}' is unknown."
    return DATASET_REGISTRY[name](Path(path))
