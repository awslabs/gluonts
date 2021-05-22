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
import itertools
import tempfile
from pathlib import Path
from typing import List

import mxnet as mx
import numpy as np
import pytest

from gluonts.mx import Tensor
from gluonts.mx.util import (
    cumsum,
    export_symb_block,
    hybrid_block_to_symbol_block,
    import_symb_block,
    weighted_average,
    mx_switch,
)
from gluonts.support.util import erf, erfinv


@pytest.mark.parametrize("vec", [[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]])
def test_cumsum(vec) -> None:

    forward_cumsum = cumsum(mx.nd, mx.nd.array(vec)).asnumpy()
    np_forward_cumsum = np.cumsum(vec, axis=-1)
    assert np.all(forward_cumsum == np_forward_cumsum), (
        f"forward cumsum did not match: "
        f"expected: {np_forward_cumsum}, obtained: {forward_cumsum}"
    )

    reverse_cumsum = cumsum(mx.nd, mx.nd.array(vec), reverse=True).asnumpy()
    np_reverse_cumsum = np.flip(
        np.cumsum(np.flip(vec, axis=-1), axis=-1), axis=-1
    )
    assert np.all(reverse_cumsum == np_reverse_cumsum), (
        f"reverse cumsum did not match: "
        f"expected: {np_reverse_cumsum}, obtained: {reverse_cumsum}"
    )

    forward_cumsum_excl = cumsum(
        mx.nd, mx.nd.array(vec), exclusive=True
    ).asnumpy()
    np_forward_cumsum_excl = np.insert(
        np_forward_cumsum[..., :-1], 0, 0, axis=-1
    )
    assert np.all(forward_cumsum_excl == np_forward_cumsum_excl), (
        f"forward cumsum (exclusive) did not match: "
        f"expected: {np_forward_cumsum_excl}, obtained: {forward_cumsum_excl}"
    )

    reverse_cumsum_excl = cumsum(
        mx.nd, mx.nd.array(vec), exclusive=True, reverse=True
    ).asnumpy()
    np_reverse_cumsum_excl = np.insert(
        np_reverse_cumsum[..., 1:], np.shape(vec)[-1] - 1, 0, axis=-1
    )
    assert np.all(reverse_cumsum_excl == np_reverse_cumsum_excl), (
        f"reverse cumsum (exclusive) did not match: "
        f"expected: {np_reverse_cumsum_excl}, obtained: {reverse_cumsum_excl}"
    )


def test_erf() -> None:
    try:
        from scipy.special import erf as scipy_erf
    except:
        pytest.skip("scipy not installed skipping test for erf")

    x = np.array(
        [-1000, -100, -10]
        + np.linspace(-5, 5, 1001).tolist()
        + [10, 100, 1000]
    )
    y_scipy = scipy_erf(x)

    # Text np
    y_np = erf(x)
    assert np.allclose(y_np, y_scipy, atol=1e-7)


def test_erfinv() -> None:
    try:
        from scipy.special import erfinv as scipy_erfinv
    except:
        pytest.skip("scipy not installed skipping test for erf")

    x = np.linspace(-1.0 + 1.0e-4, 1 - 1.0e-4, 11)
    y_scipy = scipy_erfinv(x)

    # Text np
    y_np = erfinv(x)
    assert np.allclose(y_np, y_scipy, rtol=1e-3)


def sym_block_import_export_test_cases():
    # single nested input
    class TestBlock1(mx.gluon.HybridBlock):
        def hybrid_forward(self, F, x1: Tensor, x2: List[Tensor]):
            return F.broadcast_mul(x1, x2[0])

    # multiple nested inputs
    class TestBlock2(mx.gluon.HybridBlock):
        def hybrid_forward(self, F, x1: Tensor, x2: List[Tensor]):
            return F.broadcast_add(
                F.broadcast_mul(x1, x2[0]), F.broadcast_mul(x1, x2[1])
            )

    # multiple nested inputs, and parameterized
    class TestBlock3(mx.gluon.HybridBlock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            with self.name_scope():
                self.my_param = self.params.get(
                    "my_param",
                    shape=(1,),
                    init=mx.init.Constant(5),
                    allow_deferred_init=True,
                )

        def hybrid_forward(self, F, x1: Tensor, x2: List[Tensor], my_param):
            y = F.broadcast_mul(x2[1], my_param)
            return F.broadcast_add(F.broadcast_mul(x1, x2[0]), y)

    # multiple nested inputs, parameterized, with sub-block
    class TestBlock4(mx.gluon.HybridBlock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            with self.name_scope():
                self.my_param = self.params.get(
                    "my_param",
                    shape=(1,),
                    init=mx.init.Constant(5),
                    allow_deferred_init=True,
                )
                self.dense_layer = mx.gluon.nn.Dense(3)

        def hybrid_forward(self, F, x1: Tensor, x2: List[Tensor], my_param):
            y = self.dense_layer(F.broadcast_mul(x2[1], my_param))
            return F.broadcast_add(F.broadcast_mul(x1, x2[0]), y)

    # TODO TestBlock1 is broken when using mxnet 1.6 on linux
    # TODO which apparently doesn't like that x2[1] is not used
    # return [TestBlock1, TestBlock2, TestBlock3, TestBlock4]
    return [TestBlock2, TestBlock3, TestBlock4]


@pytest.mark.parametrize(
    ["block_type", "hybridize"],
    itertools.product(sym_block_import_export_test_cases(), [True, False]),
)
def test_symb_block_export_import_nested_array(block_type, hybridize) -> None:
    x1 = mx.nd.array([1, 2, 3])
    x2 = [mx.nd.array([1, 5, 5]), mx.nd.array([2, 3, 3])]

    my_block = block_type()
    my_block.collect_params().initialize()
    if hybridize:
        my_block.hybridize()
    my_block(x1, x2)

    sb = hybrid_block_to_symbol_block(my_block, [x1, x2])

    assert np.allclose(sb(x1, x2).asnumpy(), my_block(x1, x2).asnumpy())


@pytest.mark.parametrize("block_type", sym_block_import_export_test_cases())
def test_symb_block_import_backward_compatible(block_type) -> None:
    x1 = mx.nd.array([1, 2, 3])
    x2 = [mx.nd.array([1, 5, 5]), mx.nd.array([2, 3, 3])]

    my_block = block_type()
    my_block.collect_params().initialize()
    my_block.hybridize()
    my_block(x1, x2)

    with tempfile.TemporaryDirectory(
        prefix="gluonts-estimator-temp-"
    ) as temp_dir:
        temp_path = Path(temp_dir)

        export_symb_block(my_block, temp_path, "gluonts-model")

        format_json_path = temp_path / "gluonts-model-in_out_format.json"

        assert format_json_path.exists()
        try:
            format_json_path.unlink()
            import_symb_block(3, temp_path, "gluonts-model")
        except FileNotFoundError:
            pytest.fail(
                "Symbol block import fails when format json is not in path"
            )


@pytest.mark.parametrize("x", [[1, 2, 3, 4]])
@pytest.mark.parametrize("weights", [[1, 0, 1, 0]])
def test_weighted_average(x, weights) -> None:
    x = mx.nd.array(x)
    weights = mx.nd.array(weights)
    assert weighted_average(
        F=mx.nd, x=x, weights=weights, axis=0
    ) == mx.nd.array([2.0])
    assert (
        weighted_average(
            F=mx.nd,
            x=x,
            weights=weights,
            axis=0,
            include_zeros_in_denominator=True,
        )
        == mx.nd.array([1.0])
    )


def test_mx_switch() -> None:
    a = (mx.nd.array([[1, 1, 0, 0]]), mx.nd.array([[1, 1, 1, 1]]))
    b = (mx.nd.array([[1, 0, 1, 0]]), mx.nd.array([[2, 2, 2, 2]]))
    c = mx.nd.array([[3, 3, 3, 3]])
    assert (
        (mx_switch(mx.nd, a, b, c) == mx.nd.array([1.0, 1.0, 2.0, 3.0]))
        .asnumpy()
        .all()
    )
