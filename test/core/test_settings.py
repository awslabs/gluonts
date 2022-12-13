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

from pydantic import BaseModel

from gluonts.core.settings import Settings, let


class MySettings(Settings):
    foo: str = "bar"


class Args(BaseModel):
    a: int = 1
    b: int = 2


settings = MySettings()
settings._declare("args", Args, default=Args())


def test_functional():
    settings = Settings()
    settings._declare("foo", str, default="bar")

    assert settings.foo == "bar"


def test_declarative():
    assert settings.foo == "bar"


def test_get():
    with settings._let(foo="hello"):
        assert settings["foo"] == "hello"
        assert settings.foo == "hello"
        assert settings._get("foo") == "hello"


def test_let():
    assert settings.foo == "bar"

    with let(settings, foo="hello"):
        assert settings.foo == "hello"

    assert settings.foo == "bar"

    with settings._let(foo="hello"):
        assert settings.foo == "hello"

    assert settings.foo == "bar"


def test_inject():
    @settings._inject("a")
    def x(a, b):
        return a, b

    with settings._let(a=42):
        assert x(b=0) == (42, 0)
        assert x(1, 2) == (1, 2)

    @settings._inject("b")
    def x(a, b):
        return a, b

    with settings._let(b=42):
        assert x(0) == (0, 42)
        assert x(1, 2) == (1, 2)

    @settings._inject("a")
    def x(a=1):
        return a

    assert x() == 1

    @settings._inject(a="args.a", b="args.b")
    def x(a, b):
        return a, b

    assert x() == (1, 2)

    with settings._let(args={"a": 4, "b": 5}):
        assert x() == (4, 5)


def test_partial_assignment():
    assert settings.args.a == 1

    settings._set("args", {"a": 9})
    assert settings.args.a == 9

    settings._set("args", {"b": 42})
    assert settings.args.b == 42

    with settings._let(args=dict(b=3)):
        assert settings.args.a == 9
        assert settings.args.b == 3

    settings._set("args", {"a": "1"})
    assert settings.args.a == 1
