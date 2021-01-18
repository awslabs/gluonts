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
Flat encoding for serde.

`flat.encode` always returns a flat dictionary, where keys contain information
for nested objects::

    class Inner(NamedTuple):
        val: int

    class Outer(NamedTuple):
        inner: Inner

    value = Outer(inner=Inner(val=42))

    assert encode(value) == {
        '()': '__main__.Outer',
        'inner.()': '__main__.Inner',
        'inner.val': 42},
    }

"""

from collections import defaultdict
from itertools import count
from typing import Any

from toolz.dicttoolz import keymap, valmap

from ._base import encode as base_encode, Kind, decode as base_decode


def join(a, b, sep="."):
    """Joins `a` and `b` using `sep`."""
    if not a:
        return b

    return f"{a}{sep}{b}"


def _encode(data: Any, path: str, result: dict):
    if isinstance(data, dict) and "__kind__" in data:
        kind = data["__kind__"]

        if kind == Kind.Instance:
            result[join(path, "()")] = data["class"]

            for n, arg in enumerate(data.get("args", [])):
                _encode(arg, join(path, n), result)

            for name, value in data["kwargs"].items():
                _encode(value, join(path, name), result)

        elif kind == Kind.Stateful:
            result[join(path, "!")] = data["class"]

            for name, value in data["kwargs"].items():
                _encode(value, join(path, name), result)

        elif kind == Kind:
            result[join(path, "#")] = data["class"]

        else:
            raise ValueError(f"Unsupported kind `{kind}`.")

    else:
        result[path] = data


def _asdict(trie):
    if not isinstance(trie, defaultdict):
        return trie
    return {k: _asdict(v) for k, v in trie.items()}


def nest(data):
    Trie = lambda: defaultdict(Trie)
    trie = Trie()

    for parts, value in data.items():
        *parts, key = parts

        # walk
        current = trie
        for part in parts:
            current = current[part]

        current[key] = value

    return _asdict(trie)


def get_args(data):
    args = []
    for idx in map(str, count(start=0)):
        if idx not in data:
            return args
        args.append(_translate(data.pop(idx)))


def _translate(data):
    if isinstance(data, dict):
        if "()" in data:
            ty = data.pop("()")
            args = get_args(data)
            kwargs = valmap(_translate, data)

            return {
                "__kind__": "instance",
                "class": ty,
                "args": args,
                "kwargs": kwargs,
            }

        if "!" in data:
            ty = data.pop("!")
            kwargs = valmap(_translate, data)

            return {
                "__kind__": "stateful",
                "class": ty,
                "kwargs": kwargs,
            }

        if "#" in data:
            return {
                "__kind__": "type",
                "class": data.pop("#"),
            }

        return valmap(_translate, data)

    if isinstance(data, list):
        return list(map(_translate, data))

    return data


def decode(data: dict) -> Any:
    def split_path(s):
        return tuple(s.split("."))

    nested = nest(keymap(split_path, data))
    encoded = _translate(nested)
    return base_decode(encoded)


def encode(obj) -> dict:
    """Encode a given object into a flat-dictionary.

    It uses the default-encoding, to then flatten the output.
    """
    encoded = base_encode(obj)

    result: dict = {}
    _encode(encoded, "", result)
    return result


def clone(data, kwargs=None):
    """Create a copy of a given value, by calling `encode` and `decode` on it.

    If `kwargs` is provided, it's possible to overwrite nested values.
    """
    encoded = encode(data)
    if kwargs:
        encoded.update(kwargs)

    return decode(encoded)
