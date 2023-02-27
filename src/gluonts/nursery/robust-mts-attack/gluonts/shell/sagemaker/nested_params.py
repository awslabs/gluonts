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
import re
from collections import defaultdict

from toolz.dicttoolz import valmap

from gluonts.core import serde


def split_by_prefix(data: dict) -> dict:
    rx = re.compile(r"\$(\w+)\.")

    namespace: dict = defaultdict(dict)

    def split_key(key):
        key_parts = rx.split(key, 1)
        if len(key_parts) == 1:
            prefix = ""
            suffix = key_parts[0]
        else:
            _, prefix, suffix = key_parts

        return prefix, suffix

    for key, value in data.items():
        prefix, suffix = split_key(key)
        namespace[prefix][suffix] = value

    return dict(namespace)


def decode_nested_parameters(parameters: dict) -> dict:
    inputs = split_by_prefix(parameters)
    return valmap(serde.flat.decode, inputs)


def encode_nested_parameters(obj) -> dict:
    return serde.flat.encode(obj)
