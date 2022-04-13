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

from typing import Any
from sagemaker.estimator import Framework
from sagemaker.model import Model


class CustomFramework(Framework):  # type: ignore
    """
    A custom framework is a dummy implementation which allows instantiating a
    custom AWS Sagemaker framework.
    """

    _framework_name = "custom"

    def create_model(self, **kwargs: Any) -> Model:
        raise NotImplementedError
