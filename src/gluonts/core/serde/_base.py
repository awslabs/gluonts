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

import dataclasses
import textwrap
from enum import Enum
from functools import singledispatch, partial
from pathlib import PurePath
from pydoc import locate
from typing import Any, NamedTuple, cast

from toolz.dicttoolz import valmap

from gluonts.core import fqname_for
from gluonts.pydantic import BaseModel

bad_type_msg = textwrap.dedent(
    """
    Cannot serialize type {}. See the documentation of the `encode` and
    `validate` functions at

        http://gluon-ts.mxnet.io/api/gluonts/gluonts.html

    and the Python documentation of the `__getnewargs_ex__` magic method at

        https://docs.python.org/3/library/pickle.html#object.__getnewargs_ex__

    for more information how to make this type serializable.
    """
).lstrip()


class StatelessMeta(type):
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        if isinstance(self, cls):
            self.__init__(*args, **kwargs)
            self.__init_args__ = args, kwargs
            self.__sealed__ = True
        return self


class Stateless(metaclass=StatelessMeta):
    def __getnewargs_ex__(self):
        return self.__init_args__

    def __setattr__(self, name, value):
        if hasattr(self, "__sealed__"):
            classname = self.__class__.__name__
            raise ValueError(
                f"Assignment to `{name}` outside of `{classname}.__init__`."
            )
        return object.__setattr__(self, name, value)


class Stateful:
    pass


class Kind(str, Enum):
    Type = "type"
    Instance = "instance"
    Stateful = "stateful"


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

    Dictionaries (as dictionaries).

    >>> encode({'a': 1, 'b': 2.0, 'c': '3'})
    {'a': 1, 'b': 2.0, 'c': '3'}

    Named tuples (as dictionaries with a
    ``'__kind__': <Kind.Instance: 'instance'>`` member).

    >>> from pprint import pprint
    >>> from typing import NamedTuple
    >>> class ComplexNumber(NamedTuple):
    ...     x: float = 0.0
    ...     y: float = 0.0
    >>> pprint(encode(ComplexNumber(4.0, 2.0)))
    {'__kind__': <Kind.Instance: 'instance'>,
     'class': 'gluonts.core.serde._base.ComplexNumber',
     'kwargs': {'x': 4.0, 'y': 2.0}}

    Classes with a :func:`~gluonts.core.component.validated` initializer (as
    dictionaries with a ``'__kind__': <Kind.Instance: 'instance'>`` member).

    >>> from gluonts.core.component import validated
    >>> class ComplexNumber:
    ...     @validated()
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y
    >>> pprint(encode(ComplexNumber(4.0, 2.0)))
    {'__kind__': <Kind.Instance: 'instance'>,
     'args': [],
     'class': 'gluonts.core.serde._base.ComplexNumber',
     'kwargs': {'x': 4.0, 'y': 2.0}}

    Classes with a ``__getnewargs_ex__`` magic method (as dictionaries with a
    ``'__kind__': <Kind.Instance: 'instance'>`` member).

    >>> from gluonts.core.component import validated
    >>> class ComplexNumber:
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y
    ...     def __getnewargs_ex__(self):
    ...         return [], {'x': self.x, 'y': self.y}
    >>> pprint(encode(ComplexNumber(4.0, 2.0)))
    {'__kind__': <Kind.Instance: 'instance'>,
     'args': [],
     'class': 'gluonts.core.serde._base.ComplexNumber',
     'kwargs': {'x': 4.0, 'y': 2.0}}


    Types (as dictionaries with a ``'__kind__': <Kind.Type: 'type'> member``).

    >>> encode(ComplexNumber)
    {'__kind__': <Kind.Type: 'type'>,
     'class': 'gluonts.core.serde._base.ComplexNumber'}

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
    """
    if v is None:
        return None

    if isinstance(v, (float, int, str)):
        return v

    # check for namedtuples first, to encode them not as plain tuples
    if isinstance(v, tuple) and hasattr(v, "_asdict"):
        v = cast(NamedTuple, v)
        return {
            "__kind__": Kind.Instance,
            "class": fqname_for(v.__class__),
            "kwargs": encode(v._asdict()),
        }

    if isinstance(v, (tuple, set)):
        return {
            "__kind__": Kind.Instance,
            "class": fqname_for(type(v)),
            "args": [list(map(encode, v))],
        }

    if isinstance(v, list):
        return list(map(encode, v))

    if isinstance(v, dict):
        return valmap(encode, v)

    if isinstance(v, type):
        return {"__kind__": Kind.Type, "class": fqname_for(v)}

    if hasattr(v, "__init_passed_kwargs__"):
        return {
            "__kind__": Kind.Instance,
            "class": fqname_for(v.__class__),
            "kwargs": encode(v.__init_passed_kwargs__),  # mypy: ignore
        }

    if hasattr(v, "__getnewargs_ex__"):
        args, kwargs = v.__getnewargs_ex__()  # mypy: ignore

        return {
            "__kind__": Kind.Instance,
            "class": fqname_for(v.__class__),
            # args need to be a list, since we encode tuples explicitly
            "args": encode(list(args)),
            "kwargs": encode(kwargs),
        }

    try:
        # as fallback, we try to just take the path of the value
        fqname = fqname_for(v)
        assert (
            "<lambda>" not in fqname
        ), f"Can't serialize lambda function {fqname}"

        if hasattr(v, "__self__") and hasattr(v, "__func__"):
            # v is a method
            # to model`obj.method`, we encode `getattr(obj, "method")`
            return {
                "__kind__": Kind.Instance,
                "class": fqname_for(getattr),
                "args": encode((v.__self__, v.__func__.__name__)),
            }

        return {"__kind__": Kind.Type, "class": fqname_for(v)}
    except AttributeError:
        pass

    raise RuntimeError(bad_type_msg.format(fqname_for(v.__class__)))


@encode.register(Stateful)
def encode_from_state(v: Stateful) -> Any:
    return {
        "__kind__": Kind.Stateful,
        "class": fqname_for(v.__class__),
        "kwargs": encode(v.__dict__),
    }


@encode.register(PurePath)
def encode_path(v: PurePath) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~PurePath` class.
    """
    return {
        "__kind__": Kind.Instance,
        "class": fqname_for(v.__class__),
        "args": [str(v)],
    }


@encode.register(BaseModel)
def encode_pydantic_model(v: BaseModel) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~BaseModel` class.
    """
    return {
        "__kind__": Kind.Instance,
        "class": fqname_for(v.__class__),
        "kwargs": encode(v.__dict__),
    }


@encode.register(partial)
def encode_partial(v: partial) -> Any:
    args = (v.func,) + v.args
    return {
        "__kind__": Kind.Instance,
        "class": fqname_for(v.__class__),
        "args": encode(args),
        "kwargs": encode(v.keywords),
    }


decode_disallow = [
    eval,
    exec,
    compile,
    open,
    input,
]

# `decode` only instantiates types that `encode` is known to produce. A class
# not covered by the checks below is rejected rather than instantiated.

# Builtins that `encode` emits directly (tuples/sets, methods via `getattr`,
# `functools.partial`).
decode_allow_builtins = frozenset(
    {tuple, set, frozenset, list, dict, getattr, partial}
)

# Fully-qualified names of the constructors emitted by the registered `encode`
# implementations (see serde/np.py, serde/pd.py, mx/serde.py,
# zebras/_period.py). Backends that add new `encode.register` handlers must add
# their target here as well.
decode_allow_fqnames = frozenset(
    {
        "numpy.dtype",
        "numpy.array",
        "numpy.datetime64",
        "pandas.Timestamp",
        "pandas.Period",
        "pandas.tseries.frequencies.to_offset",
        "mxnet.nd.array",
        "mxnet.context.Context",
        "gluonts.zebras.periods",
    }
)


def _is_decode_safe(cls: Any, class_name: str) -> bool:
    """
    Whether ``cls`` (located from ``class_name``) is a type that
    :func:`decode` should instantiate. Only targets that :func:`encode` is
    known to produce are allowed.
    """
    if class_name in decode_allow_fqnames:
        return True

    if cls in decode_allow_builtins:
        return True

    if isinstance(cls, type):
        # classes that explicitly opt into serde
        if issubclass(cls, (Stateless, Stateful, PurePath, BaseModel)):
            return True
        # NamedTuple subclasses, encoded as instances
        if (
            issubclass(cls, tuple)
            and hasattr(cls, "_fields")
            and hasattr(cls, "_make")
        ):
            return True
        # dataclasses are data containers; `encode` reconstructs them from
        # their fields. This covers both `serde.dataclass` and plain
        # dataclasses that expose `__init_passed_kwargs__` on instances.
        if dataclasses.is_dataclass(cls):
            return True

    # `@validated` classes carry the pydantic model on their wrapped __init__;
    # this marker is attached at import time (see component.validated).
    if getattr(getattr(cls, "__init__", None), "Model", None) is not None:
        return True

    # `serde.dataclass` classes are turned into pydantic dataclasses at import
    # time, which attach this marker (see serde._dataclass).
    if hasattr(cls, "__pydantic_model__") or hasattr(
        cls, "__pydantic_fields__"
    ):
        return True

    # classes that expose their init kwargs, or are reconstructable via the
    # pickle `__getnewargs_ex__` protocol -- both are encoding paths `encode`
    # relies on for such types
    if hasattr(cls, "__init_passed_kwargs__") or hasattr(
        cls, "__getnewargs_ex__"
    ):
        return True

    return False


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
    if isinstance(r, dict) and "__kind__" in r:
        kind = r["__kind__"]
        cls = cast(Any, locate(r["class"]))

        if cls is None:
            raise ValueError(f"Cannot locate {r['class']}.")
        if cls in decode_disallow:
            raise ValueError(f"{r['class']} cannot be run.")

        if kind == Kind.Type:
            # `Kind.Type` returns the located object without calling it.
            return cls

        # `Kind.Instance` and `Kind.Stateful` both instantiate `cls`, so it is
        # restricted to the types `encode` can produce.
        if not _is_decode_safe(cls, r["class"]):
            raise ValueError(
                f"{r['class']} is not a serde-decodable type and "
                f"cannot be instantiated during decoding."
            )

        args = decode(r.get("args", []))
        kwargs = decode(r.get("kwargs", {}))

        if kind == Kind.Instance:
            return cls(*args, **kwargs)

        if kind == Kind.Stateful:
            obj = cls.__new__(cls)
            obj.__dict__.update(kwargs)
            return obj

        raise ValueError(f"Unknown kind {kind}.")

    if isinstance(r, dict):
        return valmap(decode, r)

    if isinstance(r, list):
        return list(map(decode, r))

    return r
