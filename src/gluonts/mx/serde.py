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

from typing import Any

import mxnet as mx

from gluonts.core import fqname_for
from gluonts.core.serde import Kind, encode


@encode.register(mx.Context)
def encode_mx_context(v: mx.Context) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~mxnet.Context` class.
    """
    return {
        "__kind__": Kind.Instance,
        "class": fqname_for(v.__class__),
        "args": encode([v.device_type, v.device_id]),
    }


@encode.register(mx.nd.NDArray)
def encode_mx_ndarray(v: mx.nd.NDArray) -> Any:
    return {
        "__kind__": Kind.Instance,
        "class": "mxnet.nd.array",
        "args": encode([v.asnumpy().tolist()]),
        "kwargs": {"dtype": encode(v.dtype)},
    }
