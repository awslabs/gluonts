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
This modules wraps json libraries, namely `usjons` and `orjson` ans defaults to
just `json` if none of the others is installed. The idea is to use a high
performance variant, if available.

We expose the normal `json` functions `dump`, `dumps`, `load` and `loads`.

In addition, we define `bdump` and `bdumps`, which emit `byte` instead of
`str`.

Both `dump` and `bdump` expose a `nl` interpreter, which appends a newline
character if set to `True`.
"""

__all__ = [  # noqa
    "variant",
    "dump",
    "dumps",
    "load",
    "loads",
    "bdump",
    "bdumps",
]


def _orjson():
    import orjson

    def dumps(obj):
        # Since orjson returns bytes, we need to call `decode` on the result
        return orjson.dumps(obj).decode()

    def dump(obj, fp, nl=False):
        end = "\n" if nl else ""
        print(dumps(obj), file=fp, end=end)

    # orjson has no operators on files
    def load(fp):
        return orjson.loads(fp.read())

    def bdump(obj, fp, nl=False):
        fp.write(orjson.dumps(obj))
        if nl:
            fp.write(b"\n")

    return {
        "variant": "orjson",
        "load": load,
        "loads": orjson.loads,
        "dumps": dumps,
        "dump": dump,
        "bdump": bdump,
        "bdumps": orjson.dumps,
    }


def _ujson():
    import ujson

    def dump(obj, fp, nl=False):
        end = "\n" if nl else ""
        print(ujson.dumps(obj), file=fp, end=end)

    def bdumps(obj):
        return ujson.dumps(obj).encode()

    def bdump(obj, fp, nl=False):
        fp.write(bdumps(obj))
        if nl:
            fp.write(b"\n")

    return {
        "variant": "ujson",
        "load": ujson.load,
        "loads": ujson.loads,
        "dump": dump,
        "dumps": ujson.dumps,
        "bdump": bdump,
        "bdumps": bdumps,
    }


def _json():
    import json
    import warnings

    warnings.warn(
        "Using `json`-module for json-handling. "
        "Consider installing one of `orjson`, `ujson` "
        "to speed up serialization and deserialization."
    )

    def dump(obj, fp, nl=False):
        end = "\n" if nl else ""
        print(json.dumps(obj), file=fp, end=end)

    def bdumps(obj):
        return json.dumps(obj).encode()

    def bdump(obj, fp, nl=False):
        fp.write(bdumps(obj))
        if nl:
            fp.write(b"\n")

    return {
        "variant": "json",
        "load": json.load,
        "loads": json.loads,
        "dump": dump,
        "dumps": json.dumps,
        "bdump": bdump,
        "bdumps": bdumps,
    }


for fn in _orjson, _ujson, _json:
    try:
        globals().update(**fn())
        break
    except ImportError:
        continue
