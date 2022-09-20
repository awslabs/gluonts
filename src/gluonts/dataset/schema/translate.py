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

``gluonts.dataset.schema.translate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides a ``Translator`` class, which can be used to translate
dictionaries. It is intended to be used with GluonTS datasets, to allow for
more flexibility in the input data::

    tl = Translator.parse(target="demand", feat_dynamic_real="[price]")

"""

import re
from dataclasses import dataclass, InitVar
from typing import Any, Dict, List, Union, Optional, ClassVar

import numpy as np
from toolz import valfilter, valmap


class Op:
    def __call__(self, item):
        raise NotImplementedError


@dataclass
class Get(Op):
    """Extracts the field ``name`` from the input."""

    name: str

    def __call__(self, item):
        return item[self.name]


@dataclass
class Method(Op):
    obj: Op
    args: list

    def __call__(self, item):
        return self.obj(item)(*self.args)


@dataclass
class GetAttr(Op):
    """Invokes ``obj.name``"""

    obj: Op
    name: str

    def __call__(self, item):
        return getattr(
            self.obj(item),
            self.name,
        )


@dataclass
class GetItem(Op):
    obj: Op
    dims: List[int]

    def __call__(self, item):
        return self.obj(item).__getitem__(self.dims)


@dataclass
class Stack(Op):
    objects: List[Op]

    def __call__(self, item):
        return np.stack([obj(item) for obj in self.objects])


def one_of(s):
    options = "|".join(map(re.escape, s))

    return rf"[{options}]"


@dataclass
class Token:
    name: str
    value: str
    match: Any


@dataclass
class TokenStream:
    TOKENS: ClassVar[dict] = {
        "DOT": re.escape("."),
        "COMMA": re.escape(","),
        "PAREN_OPEN": one_of("[("),
        "PARAN_CLOSE": one_of("])"),
        "NUMBER": r"\-?\d+",
        "NAME": r"\w+",
        "WHITESPACE": r"\s+",
        "INVALID": r".+",
    }
    RX: ClassVar[str] = "|".join(
        f"(?P<{name}>{pattern})" for name, pattern in TOKENS.items()
    )

    tokens: List[Token]
    idx: InitVar[int] = 0

    @classmethod
    def from_str(cls, s):
        stream = cls(
            [
                Token(name, value, match)
                for match in re.finditer(cls.RX, s)
                for name, value in valfilter(bool, match.groupdict()).items()
                if name != "WHITESPACE"
            ]
        )

        for token in stream:
            if token.name == "INVALID":
                raise ValueError(f"Invalid token: {token}")

        return stream

    def pop(self, ty=None, val=None):
        token = self.tokens[self.idx]

        assert check_type(token, ty, val), f"Expected {ty}, got {token}."

        self.idx += 1
        return token

    def pop_if(self, ty=None, val=None):
        if self.peek(ty, val):
            return self.pop()

    def peek(self, ty=None, val=None):
        if self:
            token = self.tokens[self.idx]
            if check_type(token, ty, val):
                return token

        return None

    def __len__(self):
        return len(self.tokens) - self.idx

    def __repr__(self):
        return "".join(token.value for token in self.tokens[self.idx :])

    def __iter__(self):
        yield from self.tokens[self.idx :]


def check_type(token, ty, val):
    def matches(tok: Token):
        return tok.name == ty and (val is None or tok.value == val)

    if ty is None:
        return True

    if isinstance(ty, str):
        return matches(token)

    if isinstance(ty, list):
        return any(map(matches, token))

    return False


@dataclass
class Parser:
    stream: TokenStream

    def parse_number(self):
        return int(self.stream.pop("NUMBER").value)

    def parse_args(self):
        args = []

        # no args: `f()`
        if self.stream.peek("PARAN_CLOSE", ")"):
            return args

        while True:
            args.append(self.parse_number())

            if self.stream.pop_if("COMMA"):
                continue

            if self.stream.pop_if("PARAN_CLOSE", ")"):
                return args

            raise ValueError(f"Invalid token {self.stream.peak()}")

    def parse_getitem(self, obj):
        self.stream.pop("PAREN_OPEN", "[")

        dims = [self.parse_number()]

        while self.stream.pop_if("COMMA"):
            dims.append(self.parse_number())

        self.stream.pop("PARAN_CLOSE", "]")

        if len(dims) == 1:
            return GetItem(obj, dims[0])
        else:
            return GetItem(obj, tuple(dims))

    def parse_dot(self, obj):
        self.stream.pop("DOT")

        name = self.stream.pop("NAME").value

        return GetAttr(obj, name)

    def parse_invoke(self, obj):
        self.stream.pop("PAREN_OPEN", "(")
        args = self.parse_args()
        self.stream.pop("PARAN_CLOSE", ")")

        return Method(obj, args)

    def parse_expr(self):
        if self.stream.peek("PAREN_OPEN", "["):
            self.stream.pop()
            expr = [self.parse_expr()]

            while self.stream.pop_if("COMMA"):
                expr.append(self.parse_expr())

            self.stream.pop("PARAN_CLOSE", "]")

            obj = Stack(expr)
        else:
            token = self.stream.pop("NAME")
            obj = Get(token.value)

        while self.stream:
            if self.stream.peek("DOT"):
                obj = self.parse_dot(obj)

            elif self.stream.peek("PAREN_OPEN", "("):
                obj = self.parse_invoke(obj)

            elif self.stream.peek("PAREN_OPEN", "["):
                obj = self.parse_getitem(obj)

            else:
                break
                # raise ValueError(f"Invalid token {self.stream.peek()}")

        return obj


def parse(x: Union[str, list]) -> Op:
    if isinstance(x, list):
        return Stack(list(map(parse, x)))
    else:
        ts = TokenStream.from_str(x)
        return Parser(ts).parse_expr()


@dataclass
class Translator:
    """Simple translation for GluonTS Datasets.

    A given translator transforms an input dictionary (data-entry) into an
    output dictionary.

    Basic usage::

        >>> tl = Translator.parse(x="a[0]")
        >>> data = {"a": [1, 2, 3]}
        >>> assert tl(data)["x"] == 1

    A translator first copies all input fields into a new dictionary, before
    applying the translations. Thus, an empty `Translator` acts like the
    identity function for dictionaries:

        >>> identity = Translator()
        >>> data = {"a": 1, "b": 2, "c": 3}
        >>> assert identity(data) == data

    Using ``Translator.parse(...)```, one can define expressions to be applied
    to the input data. For example, ``Translator.parse(x="y")`` will write the
    the value of `y` to the `x` column in the output.

    These right-hand expressions support indexing (e.g. ``y[1]``), attribute
    access (e.g. ``x.T``) and method invocation (e.g. ``y.transpose(1, 0)``).
    """

    fields: Dict[str, Op]

    @staticmethod
    def parse(fields: Optional[dict] = None, **kwargs_fields) -> "Translator":
        fields_ = {}
        if fields is not None:
            fields_.update(fields)

        fields_.update(kwargs_fields)

        return Translator(valmap(parse, fields_))

    def __call__(self, item):
        result = dict(item)
        result.update(
            {name: field(item) for name, field in self.fields.items()}
        )

        return result
