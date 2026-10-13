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

import pytest

from gluonts.core import serde
from gluonts.core.component import equals
from gluonts.core.serde._base import MAX_DECODE_DEPTH
from gluonts.pydantic import BaseModel


class A(BaseModel):
    value: int


class B(BaseModel):
    a: A
    b: int


def test_nested_params():
    b = B(a=A(value=42), b=99)

    assert equals(b, serde.flat.decode(serde.flat.encode(b)))

    b2 = serde.flat.clone(b, {"a.value": 999})
    assert b2.a.value == 999


def test_kind():
    assert serde.flat.clone(A) is A


def test_decode_rejects_excessive_depth():
    value = None
    for _ in range(MAX_DECODE_DEPTH + 1):
        value = [value]

    with pytest.raises(ValueError, match="Maximum serde decoding depth"):
        serde.flat.decode({"value": value})


def test_decode_rejects_excessive_path_depth():
    path = ".".join(["value"] * (MAX_DECODE_DEPTH + 1))

    with pytest.raises(ValueError, match="Maximum serde decoding depth"):
        serde.flat.decode({path: None})
