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

from .datasets import sample_datasets, DATASETS_FILTERED
from .registry import DATA_MODULE_REGISTRY, get_data_module
from .gluonts import GluonTSDataModule
from .artificial import ArtificialDataModule
from .cheat import CheatArtificialDataModule
from .super import SuperDataModule
from .m1 import M1DataModule
from .m3 import M3DataModule
from .m4 import M4DataModule

__all__ = [
    "DATA_MODULE_REGISTRY",
    "get_data_module",
    "sample_datasets",
    "DATASETS_FILTERED",
]
