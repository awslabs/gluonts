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

"""
JSON Serialization/Deserialization
----------------------------------

The canonical way to do this is to define and `default` and `object_hook`
parameters to the json.dumps and json.loads methods. Unfortunately, due
to https://bugs.python.org/issue12657 this is not possible at the moment,
as support for custom NamedTuple serialization is broken.

To circumvent the issue, we pass the input value through custom encode
and decode functions that map nested object terms to JSON-serializable
data structures with explicit recursion.
"""

import json
from typing import Any, Optional

from ._base import decode, encode


def dump_json(o: Any, indent: Optional[int] = None) -> str:
    """
    Serializes an object to a JSON string.

    Parameters
    ----------
    o
        The object to serialize.
    indent
        An optional number of spaced to use as an indent.

    Returns
    -------
    str
        A string representing the object in JSON format.

    See Also
    --------
    load_json
        Inverse function.
    """
    return json.dumps(encode(o), indent=indent, sort_keys=True)


def load_json(s: str) -> Any:
    """
    Deserializes an object from a JSON string.

    Parameters
    ----------
    s
        A string representing the object in JSON format.

    Returns
    -------
    Any
        The deserialized object.

    See Also
    --------
    dump_json
        Inverse function.
    """
    return decode(json.loads(s))
