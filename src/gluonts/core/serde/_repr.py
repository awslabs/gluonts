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
import ast
import importlib
import itertools
import json
import math
import re
from functools import singledispatch
from typing import Any


import numpy as np

from gluonts.core import fqname_for

from ._base import Kind, encode, decode
from ._parse import parse


@singledispatch
def as_repr(x):
    if isinstance(x, (int, type(None))):
        return str(x)

    raise RuntimeError(f"Unexpected element type {fqname_for(x.__class__)}")


@as_repr.register(str)
def as_repr_str(x: str):
    # json.dumps escapes the string
    return json.dumps(x)


@as_repr.register(list)
def as_repr_list(x: list):
    inner = ", ".join(map(as_repr, x))
    return f"[{inner}]"


@as_repr.register(float)
def as_repr_float(x: float):
    if math.isfinite(x):
        return str(x)
    else:
        # e.g. `nan` needs to be encoded as `float("nan")`
        return 'float("{x}")'


@as_repr.register(dict)
def as_repr_dict(x: dict):
    kind = x.get("__kind__")

    if kind == Kind.Stateful:
        raise ValueError(
            f"Can not encode create representation for stateful object {x}."
        )

    if kind == Kind.Type:
        return x["class"]

    if kind == Kind.Instance:
        if x["class"] == "builtins.tuple":
            data = x["args"][0]
            inner = ", ".join(map(as_repr, data))
            # account for the extra `,` in `(x,)`
            if len(data) == 1:
                inner += ","
            return f"({inner})"

        if x["class"] == "builtins.set":
            data = x["args"][0]
            return f"set({as_repr(data)})"

        args = x.get("args", [])
        kwargs = x.get("kwargs", {})

        fqname = x["class"]
        bindings = ", ".join(
            itertools.chain(
                map(as_repr, args),
                [f"{k}={as_repr(v)}" for k, v in kwargs.items()],
            )
        )
        return f"{fqname}({bindings})"

    inner = ", ".join(f"{as_repr(k)}: {as_repr(v)}" for k, v in x.items())
    return f"{{{inner}}}"


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

    return as_repr(encode(o))


def load_code(s):
    return decode(parse(s))
