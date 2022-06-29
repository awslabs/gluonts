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

import typing

import mxnet as mx

__all__ = [
    "Tensor",
    "as_in_context",
    "batchify",
    "DistributionOutput",
    "GluonEstimator",
    "GluonPredictor",
    "RepresentableBlockPredictor",
    "Trainer",
]

# Tensor type for HybridBlocks in Gluon
Tensor = typing.Union[mx.nd.NDArray, mx.sym.Symbol]

from . import prelude as _
from .batchify import as_in_context, batchify
from .block.scaler import MeanScaler, NOPScaler
from .distribution import DistributionOutput, GaussianOutput
from .kernels import RBFKernel
from .model.estimator import GluonEstimator
from .model.predictor import (
    GluonPredictor,
    RepresentableBlockPredictor,
)
from .trainer import Trainer
from .util import copy_parameters, get_hybrid_forward_input_names
