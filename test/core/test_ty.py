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

import random

import pytest

from gluonts.core import serde, ty


class StatefulClass(serde.Stateful):
    def __init__(self):
        self.n = random.random()


def test_stateful():
    o = StatefulClass()
    o2 = serde.decode(serde.encode(o))
    assert o.n == o2.n


class StatelessClass(serde.Stateless):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 3


def test_stateless():
    m = StatelessClass(1, 2)

    m2 = serde.decode(serde.encode(m))
    assert m.x == m2.x
    assert m.y == m2.y
    assert m.z == m2.z


def test_stateless_immutable():
    m = StatelessClass(1, 2)
    with pytest.raises(ValueError):
        m.x = 3


class StatelessClassTyped(serde.Stateless):
    @ty.checked
    def __init__(self, a: int):
        self.a = a


def test_stateless_typed():
    m = StatelessClassTyped(a="42")
    assert m.a == 42
    assert serde.encode(m)["kwargs"]["a"] == 42


@ty.checked
def foo(x: int):
    return x


def test_checked():
    assert foo("1") == 1


def test_checked_invalid():
    with pytest.raises(TypeError):
        foo("x")


@ty.checked
def varargs(a, b: str, *args, **kwargs):
    return a, b, args, kwargs


def test_ty_varargs():
    a, b, args, kwargs = varargs(1, 2, 3, 4, 5, x=2)

    assert a == 1
    assert b == "2"
    assert args == (3, 4, 5)
    assert kwargs == {"x": 2}
