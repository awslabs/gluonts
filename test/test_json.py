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

from io import StringIO, BytesIO

import pytest

pytest.importorskip("orjson")
pytest.importorskip("ujson")

from gluonts.json import _orjson, _ujson, _json


class AttrDict(dict):
    def __getattr__(self, key):
        return dict.__getitem__(self, key)


@pytest.fixture(params=["json", "orjson", "ujson"])
def json(request):
    factory = {
        "json": _json,
        "orjson": _orjson,
        "ujson": _ujson,
    }[request.param]

    return AttrDict(factory())


@pytest.fixture(params=[True, False])
def nl(request):
    return request.param


def test_dumps(json):
    assert json.dumps([1]) == "[1]"


def test_dump(json, nl):
    buff = StringIO()
    json.dump([1], buff, nl=nl)
    buff.seek(0)
    nl = "\n" if nl else ""
    assert buff.read() == f"[1]{nl}"


def test_bdump(json, nl):
    buff = BytesIO()
    json.bdump([1], buff, nl=nl)
    buff.seek(0)
    nl = "\n" if nl else ""
    assert buff.read() == f"[1]{nl}".encode()


def test_bdumps(json):
    assert json.bdumps([1]) == b"[1]"


def test_loads(json):
    assert json.loads("[1]") == [1]
    assert json.loads(b"[1]") == [1]


def test_load(json):
    buff = StringIO("[1]")
    assert json.load(buff) == [1]

    buff = BytesIO(b"[1]")
    assert json.load(buff) == [1]
