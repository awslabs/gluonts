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


from itertools import count
from typing import Any, Union

from gluonts.core.serde import dump_json, load_json
from gluonts.itertools import batcher
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
        return dump_json(value)
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


def detrim_and_decode_sagemaker_parameters(trimmed_params: dict) -> dict:
    """Decode a SageMaker parameters dictionary where all values are strings.

    Example:

    >>> detrim_and_decode_sagemaker_parameters({
    ...     '_0_foo': '[1, ',
    ...     '_1_foo': '2, 3',
    ...     '_2_foo': ']',
    ...     '_0_bar': 'hell',
    ...     '_1_bar': 'o'
    ... })
    {'foo': [1, 2, 3], 'bar': 'hello'}
    """
    encoded_params = detrim_sagemaker_parameters(trimmed_params)
    return map_dct_values(decode_sagemaker_parameter, encoded_params)


def encode_and_trim_sagemaker_parameters(
    decoded_params: dict, max_len: int = 256
) -> dict:
    """Encode a SageMaker parameters dictionary where all values are strings then
    trim them to account for Sagemaker character size limit.

    >>> encode_and_trim_sagemaker_parameters({
    ...     "foo": [1, 2, 3],
    ...     "bar": "hello"
    ... }, max_len = 4)
    {'_0_foo': '[1, ',
     '_1_foo': '2, 3',
     '_2_foo': ']',
     '_0_bar': 'hell',
     '_1_bar': 'o'}
    """
    endoded_params = map_dct_values(encode_sagemaker_parameter, decoded_params)
    return trim_encoded_sagemaker_parameters(endoded_params, max_len)


def trim_encoded_sagemaker_parameters(
    encoded_params: dict, max_len: int = 256
) -> dict:
    """Trim parameters that have already been encoded to a given max length.

    Example:

    >>> trim_encoded_sagemaker_parameters({
    ...     'foo': '[1, 2, 3]',
    ...     'bar': 'hello'
    ... }, max_len = 4)
    {'_0_foo': '[1, ',
     '_1_foo': '2, 3',
     '_2_foo': ']',
     '_0_bar': 'hell',
     '_1_bar': 'o'}
    """
    trimmed_params = {}
    for key, value in encoded_params.items():
        if len(value) > max_len:
            for idx, substr in enumerate(batcher(value, max_len)):
                trimmed_params[f"_{idx}_{key}"] = "".join(substr)
        else:
            trimmed_params[key] = value
    return trimmed_params


def detrim_sagemaker_parameters(trimmed_params: dict) -> dict:
    """DE-trim parameters that have already been trimmed.

    Example:

    >>> detrim_sagemaker_parameters({
    ...     '_0_foo': '[1, ',
    ...     '_1_foo': '2, 3',
    ...     '_2_foo': ']',
    ...     '_0_bar': 'hell',
    ...     '_1_bar': 'o'
    ... })
    {'foo': '[1, 2, 3]', 'bar': 'hello'}
    """
    detrimmed_params = trimmed_params.copy()

    trimmed_param_names = [
        param[3:] for param in detrimmed_params if param.startswith("_0_")
    ]

    for name in trimmed_param_names:
        value = ""
        for idx in count():
            part = detrimmed_params.pop(f"_{idx}_{name}", None)
            if part is None:
                break
            value += part

        detrimmed_params[name] = value

    return detrimmed_params
