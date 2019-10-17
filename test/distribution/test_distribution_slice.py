import pytest

import mxnet as mx

from gluonts.distribution.gaussian import Gaussian

DISTR_SHAPE = (3, 4)

DISTR_CASES = [
    Gaussian(
        mu=mx.nd.random.normal(shape=DISTR_SHAPE),
        sigma=mx.nd.random.uniform(shape=DISTR_SHAPE),
    )
]

SLICE_AXIS_CASES = [[(0, 0, None), 3], [(0, 1, 3), 2], [(1, -1, None), 1]]


@pytest.mark.parametrize(
    "slice_axis_args, expected_axis_length", SLICE_AXIS_CASES
)
@pytest.mark.parametrize("distr", DISTR_CASES)
def test_distr_slice_axis(distr, slice_axis_args, expected_axis_length):
    axis, begin, end = slice_axis_args
    distr_sliced = distr.slice_axis(axis, begin, end)

    assert distr_sliced.batch_shape[axis] == expected_axis_length
