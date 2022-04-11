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

from ._base import Surrogate
from ._factory import (
    create_ensemble_surrogate,
    create_surrogate,
    ENSEMBLE_SURROGATE_REGISTRY,
    SURROGATE_REGISTRY,
)
from .autogluon import AutoGluonSurrogate
from .deepset import DeepSetSurrogate
from .mlp import MLPSurrogate
from .nonparametric import NonparametricSurrogate
from .random import RandomSurrogate
from .random_forest import RandomForestSurrogate
from .xgboost import XGBoostSurrogate

__all__ = [
    "AutoGluonSurrogate",
    "DeepSetSurrogate",
    "ENSEMBLE_SURROGATE_REGISTRY",
    "MLPSurrogate",
    "NonparametricSurrogate",
    "RandomForestSurrogate",
    "RandomSurrogate",
    "SURROGATE_REGISTRY",
    "Surrogate",
    "XGBoostSurrogate",
    "create_ensemble_surrogate",
    "create_surrogate",
]

# We need to set some parallelism flags to ensure that PyTorch behaves well on beastier machines
import torch  # pylint: disable=wrong-import-order

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
