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

from .session import account_id, default_session


def image_uri(path: str) -> str:
    """
    Returns the ECR image URI for the model at the specified path.

    Args:
        path: The path, including the tag.

    Returns:
        The image URI.
    """
    return f"{account_id()}.dkr.ecr.{default_session().region_name}.amazonaws.com/{path}"
