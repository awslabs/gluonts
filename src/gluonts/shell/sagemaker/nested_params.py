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
from itertools import count
from pydoc import locate
from typing import List

from toolz.dicttoolz import keymap

from gluonts.core import serde


def asdict(trie):
    if not isinstance(trie, defaultdict):
        return trie
    return {k: asdict(v) for k, v in trie.items()}


class Trie:
    def __init__(self, dct=None):
        trie = lambda: defaultdict(trie)
        self.trie = trie()

        if dct is not None:
            for key, val in dct.items():
                self[key] = val

    def __setitem__(self, path, value):
        trie = self.trie
        for part in path[:-1]:
            trie = trie[part]

        trie[path[-1]] = value

    def asdict(self):
        return asdict(self.trie)


def load_references(parameters: dict) -> dict:
    """Return a copy of `parameters`, where we load objects, where keys end in
    `@`:

    >>> load_references({"@": "operator.add"})
    {'@': <built-in function add>}
    """
    copy = dict(parameters)
    copy.update(
        {
            key: locate(value)
            for key, value in parameters.items()
            if key[-1] == "@"
        }
    )
    return copy


def eval_nested(data):
    if isinstance(data, dict) and "@" in data:
        Type = data.pop("@")

        args = []
        for idx in map(str, count(start=0)):
            if idx not in data:
                break
            args.append(eval_nested(data.pop(idx)))

        kwargs = {key: eval_nested(value) for key, value in data.items()}
        return Type(*args, **kwargs)

    return data


def load(mapping: dict) -> object:
    def split_path(s):
        return tuple(s.split("."))

    pathes = keymap(split_path, mapping)
    objects = load_references(pathes)
    nested = Trie(objects).asdict()
    return eval_nested(nested)


def decode_nested_parameters(parameters: dict) -> dict:
    inputs = split_by_prefix(parameters)
    return {prefix: load(args) for prefix, args in inputs.items()}


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


class Path:
    def __init__(self, *parts):
        self.parts = list(parts)

    def __truediv__(self, other):
        return Path(*self.parts + [other])

    def __repr__(self):
        return ".".join(self.parts)


def _encode(data, path, result):
    if isinstance(data, dict) and data.get("__kind__") == "instance":
        result[str(path / "@")] = data["class"]

        for n, arg in enumerate(data["args"]):
            _encode(arg, path / f"{n}", result)

        for name, value in data["kwargs"].items():
            _encode(value, path / name, result)
    else:
        result[str(path)] = data


def encode_nested_parameters(obj) -> dict:
    encoded = serde.encode(obj)

    result: dict = {}
    _encode(encoded, Path(), result)
    return result
