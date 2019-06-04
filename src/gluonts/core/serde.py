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
from gluonts.core import fqname_for

bad_type_msg = textwrap.dedent(
    '''
    Cannot serialize type {}. See the documentation of the `encode` and
    `validate` functions at

        http://gluon-ts.mxnet.io/api/gluonts/gluonts.html

    and the Python documentation of the `__getnewargs_ex__` magic method at

        https://docs.python.org/3/library/pickle.html#object.__getnewargs_ex__

    for more information how to make this type serializable.
    '''
).lstrip()


# Binary Serialization/Deserialization
# ------------------------------------


def dump_binary(o: Any) -> bytes:
    """
    Serializes an object ``o`` to binary format.

    Parameters
    ----------
    o
        The object to serialize.

    Returns
    -------
    bytes
        A sequence of bytes representing the serialized object.

    See Also
    --------
    load_binary
        Inverse function.
    """
    return pickle.dumps(o)


def load_binary(b: bytes) -> Any:
    """
    Deserializes an object from binary format.

    Parameters
    ----------
    b
        A sequence of bytes representing the serialized object.

    Returns
    -------
    Any
        The deserialized object.

    See Also
    --------
    dump_binary
        Inverse function.
    """
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
    """
    Serializes an object to a JSON string.

    Parameters
    ----------
    o
        The object to serialize.
    indent
        An optional number of spaced to use as an indent.

    Returns
    -------
    str
        A string representing the object in JSON format.

    See Also
    --------
    load_json
        Inverse function.
    """
    return json.dumps(encode(o), indent=indent, sort_keys=True)


def load_json(s: str) -> Any:
    """
    Deserializes an object from a JSON string.

    Parameters
    ----------
    s
        A string representing the object in JSON format.

    Returns
    -------
    Any
        The deserialized object.

    See Also
    --------
    dump_json
        Inverse function.
    """
    return decode(json.loads(s))


# Code Serialization/Deserialization
# ----------------------------------


def dump_code(o: Any) -> str:
    """
    Serializes an object to a Python code string.

    Parameters
    ----------
    o
        The object to serialize.

    Returns
    -------
    str
        A string representing the object as Python code.

    See Also
    --------
    load_code
        Inverse function.
    """

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
    """
    Deserializes an object from a Python code string.

    Parameters
    ----------
    c
        A string representing the object as Python code.

    Returns
    -------
    Any
        The deserialized object.

    See Also
    --------
    dump_code
        Inverse function.
    """

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
    """
    Transforms a value `v` as a serializable intermediate representation (for
    example, named tuples are encoded as dictionaries). The intermediate
    representation is then recursively traversed and serialized either as
    Python code or as JSON string.

    This function is decorated with :func:`~functools.singledispatch` and can
    be specialized by clients for families of types that are not supported by
    the basic implementation (explained below).

    Examples
    --------

    The conversion logic implemented by the basic implementation is used
    as a fallback and is best explained by a series of examples.

    Lists (as lists).

    >>> encode([1, 2.0, '3'])
    [1, 2.0, '3']

    Tuples (as lists).

    >>> encode((1, 2.0, '3'))
    [1, 2.0, '3']

    Dictionaries (as dictionaries).

    >>> encode({'a': 1, 'b': 2.0, 'c': '3'})
    {'a': 1, 'b': 2.0, 'c': '3'}

    Named tuples (as dictionaries with a ``'__kind__': 'instance'`` member).

    >>> from pprint import pprint
    >>> from typing import NamedTuple
    >>> class ComplexNumber(NamedTuple):
    ...     x: float = 0.0
    ...     y: float = 0.0
    >>> pprint(encode(ComplexNumber(4.0, 2.0)))
    {'__kind__': 'instance',
     'class': 'gluonts.core.serde.ComplexNumber',
     'kwargs': {'x': 4.0, 'y': 2.0}}

    Classes with a :func:`~gluonts.core.component.validated` initializer (as
    dictionaries with a ``'__kind__': 'instance'`` member).

    >>> from gluonts.core.component import validated
    >>> class ComplexNumber:
    ...     @validated()
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y
    >>> pprint(encode(ComplexNumber(4.0, 2.0)))
    {'__kind__': 'instance',
     'args': [],
     'class': 'gluonts.core.serde.ComplexNumber',
     'kwargs': {'x': 4.0, 'y': 2.0}}

    Classes with a ``__getnewargs_ex__`` magic method (as dictionaries with a
    ``'__kind__': 'instance'`` member).

    >>> from gluonts.core.component import validated
    >>> class ComplexNumber:
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y
    ...     def __getnewargs_ex__(self):
    ...         return [], {'x': self.x, 'y': self.y}
    >>> pprint(encode(ComplexNumber(4.0, 2.0)))
    {'__kind__': 'instance',
     'args': [],
     'class': 'gluonts.core.serde.ComplexNumber',
     'kwargs': {'x': 4.0, 'y': 2.0}}


    Types (as dictionaries with a ``'__kind__': 'type' member``).

    >>> encode(ComplexNumber)
    {'__kind__': 'type', 'class': 'gluonts.core.serde.ComplexNumber'}

    Parameters
    ----------
    v
        The value to be encoded.

    Returns
    -------
    Any
        An encoding of ``v`` that can be serialized to Python code or
        JSON string.

    See Also
    --------
    decode
        Inverse function.
    dump_json
        Serializes an object to a JSON string.
    dump_code
        Serializes an object to a Python code string.
    """
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
def encode_path(v: Path) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~Path` class.
    """
    return {
        '__kind__': kind_inst,
        'class': fqname_for(v.__class__),
        'args': encode([str(v)]),
    }


@encode.register(BaseModel)
def encode_pydantic_model(v: BaseModel) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~BaseModel` class.
    """
    return {
        '__kind__': kind_inst,
        'class': fqname_for(v.__class__),
        'kwargs': encode(v.__values__),
    }


@encode.register(mx.Context)
def encode_mx_context(v: mx.Context) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~mxnet.Context` class.
    """
    return {
        '__kind__': kind_inst,
        'class': fqname_for(v.__class__),
        'args': encode([v.device_type, v.device_id]),
    }


def decode(r: Any) -> Any:
    """
    Decodes a value from an intermediate representation `r`.

    Parameters
    ----------
    r
        An intermediate representation to be decoded.

    Returns
    -------
    Any
        A Python data structure corresponding to the decoded version of ``r``.

    See Also
    --------
    encode
        Inverse function.
    """

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
