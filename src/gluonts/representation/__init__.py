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

# Relative imports
from .representation import Representation
from .binning import Binning
from .global_relative_binning import GlobalRelativeBinning
from .local_absolute_binning import LocalAbsoluteBinning
from .hybrid_representation import HybridRepresentation
from .mean_scaling import MeanScaling
from .nop_scaling import NOPScaling
from .dim_expansion import DimExpansion
from .embedding import Embedding

__all__ = [
    "Representation",
    "Binning",
    "GlobalRelativeBinning",
    "LocalAbsoluteBinning",
    "HybridRepresentation",
    "MeanScaling",
    "NOPScaling",
    "DimExpansion",
    "Embedding",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
