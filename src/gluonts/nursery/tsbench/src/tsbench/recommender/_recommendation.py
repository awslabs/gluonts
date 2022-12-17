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

from dataclasses import dataclass
from typing import Generic, TypeVar
from tsbench.evaluations.metrics import Performance

T = TypeVar("T")


@dataclass
class Recommendation(Generic[T]):
    """
    A recommendation provides a configuration along with its expected
    performance.
    """

    config: T
    performance: Performance
