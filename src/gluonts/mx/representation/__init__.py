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

from .custom_binning import CustomBinning
from .dim_expansion import DimExpansion
from .discrete_pit import DiscretePIT
from .embedding import Embedding
from .global_relative_binning import GlobalRelativeBinning
from .hybrid_representation import HybridRepresentation
from .local_absolute_binning import LocalAbsoluteBinning
from .mean_scaling import MeanScaling
from .representation import Representation
from .representation_chain import RepresentationChain

__all__ = [
    "Representation",
    "CustomBinning",
    "GlobalRelativeBinning",
    "LocalAbsoluteBinning",
    "HybridRepresentation",
    "MeanScaling",
    "DimExpansion",
    "Embedding",
    "DiscretePIT",
    "RepresentationChain",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
