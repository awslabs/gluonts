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

__all__ = [
    "PositiveInt",
    "PositiveFloat",
    "NonNegativeInt",
    "NonNegativeFloat",
]

# re-export
from pydantic import PositiveInt, PositiveFloat

from pydantic import ConstrainedInt, ConstrainedFloat, confloat


class NonNegativeInt(ConstrainedInt):
    ge = 0


class NonNegativeFloat(ConstrainedFloat):
    ge = 0.0


class Interval01(ConstrainedFloat):
    # values 0 < x < 1
    gt = 0
    lt = 1
