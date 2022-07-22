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
gluonts.core.serde
~~~~~~~~~~~~~~~~~~~~~

Simple Serialization/Deserialization framework that uses human-readable
json-encodings to represent data.

In some ways, it is an alternative to `pickle`, but instead of using a binary
encoding, `serde` outputs json-compatible text. This has two main advantages:
It is human-readable and the encoding works across different Python versions.

Values which can be directly represented in `json` are not changed, thus
`serde.encode` is virtually the same as `json.dumps` for values such as `1`,
`["a", "b", "c"]` or `{"foo": "bar"}`.

However, more complex values are encoded differently. For these, we create an
object which uses a special `__kind__` attribute, which can be one of::

    class Kind(str, Enum):
        Type = "type"
        Instance = "instance"
        Stateful = "stateful"


A type is just a path to a class or function, an instance is an object which
can be re-constructed by passing arguments to its constructor, and stateful
represents objects that are decoded by setting the `__dict__` attribute on
an otherwise empty instance of the type.

`serde.encode` uses `functools.singledispatch` to encode a given object. It is
implemented for a variety of existing types, such as named-tuples, paths, and
`pydantic.BaseModel`.

In addition, one can derive from `serde.Stateful` or `serde.Stateless` to opt
into one of the behaviours. The latter requires that the class supports
`__getnewargs_ex__` from the pickle protocol.

To encode custom values, one can use the `serde.encode.register` decorator.
For example, support for numpy's arrays are added by::

    @encode.register(np.ndarray)
    def encode_np_ndarray(v: np.ndarray) -> Any:
        return {
            "__kind__": Kind.Instance,
            "class": "numpy.array",
            "args": encode([v.tolist(), v.dtype]),
        }

There is no need to implement `decode` for a given type since the encoding
should contain the information on how the object can be constructed.

Similarly to `json`, `serde` does not support object-identity. This means that
if an object exists twice in an object graph, it is encoded at each of the
occurances. Consequently, ciruclar references do not work.

`dump_json` and `load_json` are simple helpers, which use `encode` and
`decode` internally.
"""
from . import flat
from ._base import Stateful, Stateless, decode, encode
from ._dataclass import dataclass, EVENTUAL, Eventual, OrElse
from ._json import dump_json, load_json
from ._repr import dump_code, load_code


# TODO: remove
# These are needed because we implement `encode` for numpy and pandas types in
# submodules.
from .np import *  # noqa
from .pd import *  # noqa


__all__ = [
    "flat",
    "encode",
    "decode",
    "dump_code",
    "load_code",
    "dump_json",
    "load_json",
    "Stateful",
    "Stateless",
    "dataclass",
    "EVENTUAL",
    "Eventual",
    "OrElse",
]
