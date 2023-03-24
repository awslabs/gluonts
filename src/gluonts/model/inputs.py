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

from collections import UserDict
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Callable, Dict, Type, Tuple

from toolz import valmap


@dataclass
class Input:
    shape: Tuple[int, ...]
    dtype: Any
    required: bool = True


@dataclass
class InputSpec(UserDict):
    data: Dict[str, Input]
    zeros_fn: Callable

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return valmap(attrgetter("shape"), self)

    @property
    def dtypes(self) -> Dict[str, Type]:
        return valmap(attrgetter("dtype"), self)

    def zeros(self):
        return {
            name: self.zeros_fn(input.shape, dtype=input.dtype)
            for name, input in self.items()
        }
