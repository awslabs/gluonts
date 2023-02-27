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

import numpy as np

import mxnet as mx

from gluonts.core.component import (
    equals,
    equals_default_impl,
    skip_encoding,
    tensor_to_numpy,
)


@equals.register(mx.gluon.ParameterDict)
def equals_parameter_dict(
    this: mx.gluon.ParameterDict, that: mx.gluon.ParameterDict
) -> bool:
    """
    Structural equality check between two :class:`~mxnet.gluon.ParameterDict`
    objects.

    Two parameter dictionaries ``this`` and ``that`` are considered
    *structurally equal* if the following conditions are satisfied:

    1. They contain the same keys (modulo the key prefix which is stripped).
    2. The data in the corresponding value pairs is equal, as defined by the
       :func:`~mxnet.test_utils.almost_equal` function (in this case we call
       the function with ``equal_nan=True``, that is, two aligned ``NaN``
       values are always considered equal).

    Specializes :func:`equals` for invocations where the first parameter is an
    instance of the :class:`~mxnet.gluon.ParameterDict` class.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals
        Dispatching function.
    """
    if type(this) != type(that):
        return False

    def strip_prefix_enumeration(key, prefix):
        if key.startswith(prefix):
            name = key[len(prefix) :]
        else:
            prefix, args = key.split("_", 1)
            name = prefix.rstrip("0123456789") + args

        return name

    this_param_names_stripped = [
        strip_prefix_enumeration(key, this.prefix) for key in this.keys()
    ]
    that_param_names_stripped = [
        strip_prefix_enumeration(key, that.prefix) for key in that.keys()
    ]

    if not this_param_names_stripped == that_param_names_stripped:
        return False

    for this_param_name, that_param_name in zip(this.keys(), that.keys()):
        x = this[this_param_name].data().asnumpy()
        y = that[that_param_name].data().asnumpy()
        if not mx.test_utils.almost_equal(x, y, equal_nan=True):
            return False

    return True


@equals.register(mx.gluon.HybridBlock)
def equals_representable_block(
    this: mx.gluon.HybridBlock, that: mx.gluon.HybridBlock
) -> bool:
    """
    Structural equality check between two :class:`~mxnet.gluon.HybridBlock`
    objects with :func:`validated` initializers.

    Two blocks ``this`` and ``that`` are considered *structurally equal* if all
    the conditions of :func:`equals` are met, and in addition their parameter
    dictionaries obtained with
    :func:`~mxnet.gluon.block.Block.collect_params` are also structurally
    equal.

    Specializes :func:`equals` for invocations where the first parameter is an
    instance of the :class:`~mxnet.gluon.HybridBlock` class.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals
        Dispatching function.
    equals_parameter_dict
        Specialization of :func:`equals` for Gluon
        :class:`~mxnet.gluon.ParameterDict` input arguments.
    """
    if not equals_default_impl(this, that):
        return False

    if not equals_parameter_dict(this.collect_params(), that.collect_params()):
        return False

    return True


@skip_encoding.register(mx.gluon.ParameterDict)
def skip_encoding_mx_gluon_parameterdict(v: mx.gluon.ParameterDict) -> bool:
    return True


@tensor_to_numpy.register(mx.ndarray.NDArray)
def _(tensor: mx.ndarray.NDArray) -> np.ndarray:
    return tensor.asnumpy()
