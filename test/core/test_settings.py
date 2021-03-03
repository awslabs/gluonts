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

from gluonts.core.settings import Settings, let, inject


class MySettings(Settings):
    foo: str = "bar"


settings = MySettings()


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
