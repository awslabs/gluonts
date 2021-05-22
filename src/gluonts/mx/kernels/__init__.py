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

from ._kernel import Kernel
from ._kernel_output import KernelOutput, KernelOutputDict
from ._periodic_kernel import PeriodicKernel, PeriodicKernelOutput
from ._rbf_kernel import RBFKernel, RBFKernelOutput

__all__ = [
    "Kernel",
    "PeriodicKernel",
    "RBFKernel",
    "PeriodicKernelOutput",
    "RBFKernelOutput",
    "KernelOutput",
    "KernelOutputDict",
]

for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
