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

# Standard library imports
import importlib
import itertools
import json
import math
import pickle
import re
import textwrap
from functools import singledispatch
from pathlib import Path
from pydoc import locate
from typing import Any, Optional

# Third-party imports
import mxnet as mx
from pydantic import BaseModel

# Relative imports
from ._base import fqname_for

bad_type_msg = textwrap.dedent(
    '''
    Cannot serialize type {}. You can make
    this type serializable by defining __getnewargs_ex__(). See

        https://docs.python.org/3/library/pickle.html#object.__getnewargs_ex__

    for more information.
    '''
).lstrip()


# Binary Serialization/Deserialization
# ------------------------------------


def dump_binary(o: Any) -> bytes:
    """Serializes an object to binary format."""
    return pickle.dumps(o)


def load_binary(b: bytes) -> Any:
    """Deserializes an object from binary format."""
    return pickle.loads(b)


# JSON Serialization/Deserialization
# ----------------------------------

# The canonical way to do this is to define and `default` and `object_hook`
# parameters to the json.dumps and json.loads methods. Unfortunately, due
# to https://bugs.python.org/issue12657 this is not possible at the moment,
# as support for custom NamedTuple serialization is broken.
#
# To circumvent the issue, we pass the input value through custom encode
# and decode functions that map nested object terms to JSON-serializable
# data structures with explicit recursion.


def dump_json(o: Any, indent: Optional[int] = None) -> str:
    """Serializes an object to a JSON string."""
    return json.dumps(encode(o), indent=indent, sort_keys=True)


def load_json(b: str) -> Any:
    """Deserializes an object from a JSON string."""
    return decode(json.loads(b))


# Code Serialization/Deserialization
# ----------------------------------


def dump_code(o: Any) -> str:
    """Serializes an object to a Python code string."""

    def _dump_code(x: Any) -> str:
        # r = { 'class': ..., 'args': ... }
        # r = { 'class': ..., 'kwargs': ... }
        if type(x) == dict and x.get('__kind__') == kind_inst:
            args = x['args'] if 'args' in x else []
            kwargs = x['kwargs'] if 'kwargs' in x else {}
            return '{fqname}({bindings})'.format(
                fqname=x["class"],
                bindings=', '.join(
                    itertools.chain(
                        [_dump_code(v) for v in args],
                        [f'{k}={_dump_code(v)}' for k, v in kwargs.items()],
                    )
                ),
            )
        if type(x) == dict and x.get('__kind__') == kind_type:
            return x['class']
        if isinstance(x, dict):
            elems = [f'{_dump_code(k)}: {_dump_code(v)}' for k, v in x.items()]
            return '{' + ', '.join(elems) + '}'
        elif isinstance(x, list):
            elems = [dump_code(v) for v in x]
            return '[' + ', '.join(elems) + ']'
        elif isinstance(x, tuple):
            elems = [dump_code(v) for v in x]
            return '(' + ', '.join(elems) + ',)'
        elif isinstance(x, str):
            return '"' + x + '"'  # TODO: escape comp characters
        elif isinstance(x, float):
            return str(x) if math.isfinite(x) else 'float("' + str(x) + '")'
        elif isinstance(x, int) or x is None:
            return str(x)
        else:
            x = fqname_for(x.__class__)
            raise RuntimeError(f'Unexpected element type {x}')

    return _dump_code(encode(o))


def load_code(c: str) -> Any:
    """Deserializes an object from a Python code string."""

    def _load_code(code: str, modules=None):
        if modules is None:
            modules = {}
        try:
            return eval(code, modules)
        except NameError as e:
            m = re.match(r"name '(?P<module>.+)' is not defined", str(e))
            if m:
                name = m['module']
                return _load_code(
                    code,
                    {**(modules or {}), name: importlib.import_module(name)},
                )
            else:
                raise e
        except AttributeError as e:
            m = re.match(
                r"module '(?P<module>.+)' has no attribute '(?P<package>.+)'",
                str(e),
            )
            if m:
                name = m['module'] + '.' + m['package']
                return _load_code(
                    code,
                    {**(modules or {}), name: importlib.import_module(name)},
                )
            else:
                raise e
        except Exception as e:
            raise e

    return _load_code(c)


# Structural encoding/decoding
# ----------------------------

kind_type = 'type'
kind_inst = 'instance'


@singledispatch
def encode(v: Any) -> Any:
    """Encode a value `v` to a serializable intermediate representation."""
    if isinstance(v, type(None)):
        return None
    elif isinstance(v, (float, int, str)):
        return v
    elif isinstance(v, list) or type(v) == tuple:
        return [encode(v) for v in v]
    elif isinstance(v, tuple) and not hasattr(v, '_asdict'):
        return tuple([encode(v) for v in v])
    elif isinstance(v, dict):
        return {k: encode(v) for k, v in v.items()}
    elif isinstance(v, type):
        return {'__kind__': kind_type, 'class': fqname_for(v)}
    elif isinstance(v, tuple) and hasattr(v, '_asdict'):
        return {
            '__kind__': kind_inst,
            'class': fqname_for(v.__class__),
            'kwargs': encode(getattr(v, '_asdict')()),
        }
    elif hasattr(v, '__getnewargs_ex__'):
        args, kwargs = getattr(v, '__getnewargs_ex__')()
        return {
            '__kind__': kind_inst,
            'class': fqname_for(v.__class__),
            'args': encode(args),
            'kwargs': encode(kwargs),
        }
    else:
        raise RuntimeError(bad_type_msg.format(fqname_for(v.__class__)))


@encode.register(Path)
def equals_path(v: Path) -> Any:
    return {
        '__kind__': kind_inst,
        'class': fqname_for(v.__class__),
        'args': encode([str(v)]),
    }


@encode.register(BaseModel)
def equals_base_model(v: BaseModel) -> Any:
    return {
        '__kind__': kind_inst,
        'class': fqname_for(v.__class__),
        'kwargs': encode(v.__values__),
    }


@encode.register(mx.Context)
def equals_mx_context(v: mx.Context) -> Any:
    return {
        '__kind__': kind_inst,
        'class': fqname_for(v.__class__),
        'args': encode([v.device_type, v.device_id]),
    }


def decode(r: Any) -> Any:
    """Decode a value from an intermediate representation `r`."""

    # structural recursion over the possible shapes of r
    # r = { 'class': ..., 'args': ... }
    # r = { 'class': ..., 'kwargs': ... }
    if type(r) == dict and r.get('__kind__') == kind_inst:
        cls = locate(r["class"])
        args = decode(r['args']) if 'args' in r else []
        kwargs = decode(r['kwargs']) if 'kwargs' in r else {}
        return cls(*args, **kwargs)
    # r = { 'class': ..., 'args': ... }
    # r = { 'class': ..., 'kwargs': ... }
    if type(r) == dict and r.get('__kind__') == kind_type:
        return locate(r["class"])
    # r = { k1: v1, ..., kn: vn }
    elif type(r) == dict:
        return {k: decode(v) for k, v in r.items()}
    # r = ( y1, ..., yn )
    elif type(r) == tuple:
        return tuple([decode(y) for y in r])
    # r = [ y1, ..., yn ]
    elif type(r) == list:
        return [decode(y) for y in r]
    # r = a
    else:
        return r
