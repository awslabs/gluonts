# Third-party imports
import mxnet as mx
import numpy as np
import pytest

# First-party imports
from gluonts.support import util


@pytest.mark.parametrize("vec", [[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]])
def test_cumsum(vec) -> None:

    forward_cumsum = util.cumsum(mx.nd, mx.nd.array(vec)).asnumpy()
    np_forward_cumsum = np.cumsum(vec, axis=-1)
    assert np.all(forward_cumsum == np_forward_cumsum), (
        f"forward cumsum did not match: "
        f"expected: {np_forward_cumsum}, obtained: {forward_cumsum}"
    )

    reverse_cumsum = util.cumsum(
        mx.nd, mx.nd.array(vec), reverse=True
    ).asnumpy()
    np_reverse_cumsum = np.flip(
        np.cumsum(np.flip(vec, axis=-1), axis=-1), axis=-1
    )
    assert np.all(reverse_cumsum == np_reverse_cumsum), (
        f"reverse cumsum did not match: "
        f"expected: {np_reverse_cumsum}, obtained: {reverse_cumsum}"
    )

    forward_cumsum_excl = util.cumsum(
        mx.nd, mx.nd.array(vec), exclusive=True
    ).asnumpy()
    np_forward_cumsum_excl = np.insert(
        np_forward_cumsum[..., :-1], 0, 0, axis=-1
    )
    assert np.all(forward_cumsum_excl == np_forward_cumsum_excl), (
        f"forward cumsum (exclusive) did not match: "
        f"expected: {np_forward_cumsum_excl}, obtained: {forward_cumsum_excl}"
    )

    reverse_cumsum_excl = util.cumsum(
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
    y_mxnet = util.erf(mx.nd, mx.nd.array(x)).asnumpy()
    y_scipy = scipy_erf(x)
    assert np.allclose(y_mxnet, y_scipy)
