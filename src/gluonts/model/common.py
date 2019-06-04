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
import types
import typing

# Third-party imports
import mxnet as mx
import numpy as np

# Tensor type for HybridBlocks in Gluon
Tensor = typing.Union[mx.nd.NDArray, mx.sym.Symbol]

# Type of tensor-transforming functions in Gluon
TensorTransformer = typing.Callable[[types.ModuleType, Tensor], Tensor]

# untyped global configuration passed to model components
GlobalConfig = typing.Dict[str, typing.Any]

# to annotate Numpy parameter
NPArrayLike = typing.Union[int, float, np.ndarray]
