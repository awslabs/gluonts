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
from typing import Tuple

# First-party imports
from gluonts.core.component import validated
from gluonts.distribution.bijection import Bijection
from gluonts.distribution.distribution_output import Output
from gluonts.model.common import Tensor


class BijectionOutput(Output):
    """
    Class to connect a network to a bijection
    """

    bij_cls: type

    @validated()
    def __init__(self) -> None:
        pass

    def domain_map(self, F, *args: Tensor):
        raise NotImplementedError()

    def bijection(self, bij_args: Tensor) -> Bijection:
        return self.bij_cls(*bij_args)

    @property
    def event_shape(self) -> Tuple:
        raise NotImplementedError()
