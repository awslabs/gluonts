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

import typing
from pkgutil import extend_path

import mxnet as mx

__path__ = extend_path(__path__, __name__)  # type: ignore

# Tensor type for HybridBlocks in Gluon
Tensor = typing.Union[mx.nd.NDArray, mx.sym.Symbol]

from gluonts.mx.prelude import *
