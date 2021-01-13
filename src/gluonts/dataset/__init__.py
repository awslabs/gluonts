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

# !!! DO NOT MODIFY !!! (pkgutil-style namespace package)

from pkgutil import extend_path
from .common import ListDataset
from .field_names import FieldName
from .repository.datasets import get_dataset, dataset_recipes, load_datasets
from .loader import TrainDataLoader
from .rolling_dataset import StepStrategy, generate_rolling_dataset

__path__ = extend_path(__path__, __name__)  # type: ignore
__all__ = [
    "ListDataset",
    "FieldName",
    "get_dataset",
    "load_datasets",
    "dataset_recipes",
    "TrainDataLoader",
    "StepStrategy",
    "generate_rolling_dataset",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
