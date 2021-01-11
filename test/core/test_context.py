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

from gluonts.core.context import Context, let, inject


class MyContext(Context):
    foo: str = "bar"


ctx = MyContext()


def test_functional():
    ctx = Context()
    ctx._declare("foo", str, default="bar")

    assert ctx.foo == "bar"


def test_declarative():
    assert ctx.foo == "bar"


def test_get():
    with ctx._let(foo="hello"):
        assert ctx["foo"] == "hello"
        assert ctx.foo == "hello"
        assert ctx._get("foo") == "hello"


def test_let():
    assert ctx.foo == "bar"

    with let(ctx, foo="hello"):
        assert ctx.foo == "hello"

    assert ctx.foo == "bar"

    with ctx._let(foo="hello"):
        assert ctx.foo == "hello"

    assert ctx.foo == "bar"


def test_inject():
    @ctx._inject("a")
    def x(a, b):
        return a, b

    with ctx._let(a=42):
        assert x(b=0) == (42, 0)
        assert x(1, 2) == (1, 2)

    @ctx._inject("b")
    def x(a, b):
        return a, b

    with ctx._let(b=42):
        assert x(0) == (0, 42)
        assert x(1, 2) == (1, 2)
