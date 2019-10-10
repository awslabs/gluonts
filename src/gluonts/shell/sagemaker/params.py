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
import json
from typing import Union, Any

# First-party imports
from gluonts.core.serde import load_json
from gluonts.support.util import map_dct_values


def decode_sagemaker_parameter(value: str) -> Union[list, dict, str]:
    """
    All values passed through the SageMaker API are encoded as strings. Thus
    we pro-actively decode values that seem like arrays or dicts.

    Integer values (e.g. `"1"`) are handled by pydantic models further down
    the pipeline.
    """
    value = value.strip()

    # TODO: is this the right way to do things?
    #       what about fields which start which match the pattern for
    #       some reason?
    is_list = value.startswith("[") and value.endswith("]")
    is_dict = value.startswith("{") and value.endswith("}")

    if is_list or is_dict:
        return load_json(value)
    else:
        return value


def encode_sagemaker_parameter(value: Any) -> str:
    """
    All values passed through the SageMaker API must be encoded as strings.
    """
    if not isinstance(value, str):
        return json.dumps(value)
    else:
        return value


def decode_sagemaker_parameters(encoded_params: dict) -> dict:
    """Decode a SageMaker parameters dictionary where all values are strings.

    Example:

    >>> decode_sagemaker_parameters({
    ...     "foo": "[1, 2, 3]",
    ...     "bar": "hello"
    ... })
    {'foo': [1, 2, 3], 'bar': 'hello'}
    """
    return map_dct_values(decode_sagemaker_parameter, encoded_params)


def encode_sagemaker_parameters(decoded_params: dict) -> dict:
    """Encode a SageMaker parameters dictionary where all values are strings.

    Example:

    >>> encode_sagemaker_parameters({
    ...     "foo": [1, 2, 3],
    ...     "bar": "hello"
    ... })
    {'foo': '[1, 2, 3]', 'bar': 'hello'}
    """
    return map_dct_values(encode_sagemaker_parameter, decoded_params)
