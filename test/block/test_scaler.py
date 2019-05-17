import pytest
import mxnet as mx
import numpy as np

from gluonts.block import scaler


test_cases = [
    (
        scaler.MeanScaler(),
        mx.nd.array(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        mx.nd.array([1.0, 3.0, 1.5, 1.00396824, 1.00396824]),
    ),
    (
        scaler.MeanScaler(keepdims=True),
        mx.nd.array(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        mx.nd.array([1.0, 3.0, 1.5, 1.00396824, 1.00396824]).expand_dims(
            axis=1
        ),
    ),
    (
        scaler.MeanScaler(),
        mx.nd.array(
            [
                [[1.0]] * 50,
                [[0.0]] * 25 + [[3.0]] * 25,
                [[2.0]] * 49 + [[1.5]] * 1,
                [[0.0]] * 50,
                [[1.0]] * 50,
            ]
        ),
        mx.nd.array(
            [
                [[1.0]] * 50,
                [[0.0]] * 25 + [[1.0]] * 25,
                [[0.0]] * 49 + [[1.0]] * 1,
                [[1.0]] * 50,
                [[0.0]] * 50,
            ]
        ),
        mx.nd.array([1.0, 3.0, 1.5, 1.00396824, 1.00396824]).expand_dims(
            axis=1
        ),
    ),
    (
        scaler.MeanScaler(minimum_scale=1e-8),
        mx.nd.array(
            [
                [[1.0, 2.0]] * 50,
                [[0.0, 0.0]] * 25 + [[3.0, 6.0]] * 25,
                [[2.0, 4.0]] * 49 + [[1.5, 3.0]] * 1,
                [[0.0, 0.0]] * 50,
                [[1.0, 2.0]] * 50,
            ]
        ),
        mx.nd.array(
            [
                [[1.0, 1.0]] * 50,
                [[0.0, 1.0]] * 25 + [[1.0, 0.0]] * 25,
                [[1.0, 0.0]] * 49 + [[0.0, 1.0]] * 1,
                [[1.0, 0.0]] * 50,
                [[0.0, 1.0]] * 50,
            ]
        ),
        mx.nd.array(
            [
                [1.0, 2.0],
                [3.0, 1.61111116],
                [2.0, 3.0],
                [1.28160918, 1.61111116],
                [1.28160918, 2.0],
            ]
        ),
    ),
    (
        scaler.MeanScaler(),
        mx.nd.array(
            [
                [120.0] * 25 + [150.0] * 25,
                [0.0] * 10 + [3.0] * 20 + [61.0] * 20,
                [0.0] * 50,
                [2e-2] * 10 + [0.0] * 30 + [3e-2] * 10,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 25 + [1.0] * 25,
                [0.0] * 10 + [1.0] * 20 + [1.0] * 20,
                [0.0] * 50,
                [1.0] * 10 + [0.0] * 30 + [1.0] * 10,
            ]
        ),
        mx.nd.array([135.0, 32.0, 73.00454712, 2.5e-2]),
    ),
    (
        scaler.MeanScaler(),
        mx.nd.random.normal(shape=(5, 30)),
        mx.nd.zeros(shape=(5, 30)),
        1e-10 * mx.nd.ones(shape=(5,)),
    ),
    (
        scaler.MeanScaler(minimum_scale=1e-6),
        mx.nd.random.normal(shape=(5, 30, 1)),
        mx.nd.zeros(shape=(5, 30, 1)),
        1e-6 * mx.nd.ones(shape=(5, 1)),
    ),
    (
        scaler.MeanScaler(minimum_scale=1e-12),
        mx.nd.random.normal(shape=(5, 30, 3)),
        mx.nd.zeros(shape=(5, 30, 3)),
        1e-12 * mx.nd.ones(shape=(5, 3)),
    ),
    (
        scaler.NOPScaler(),
        mx.nd.random.normal(shape=(10, 20, 30)),
        mx.nd.random.normal(shape=(10, 20, 30)) > 0,
        mx.nd.ones(shape=(10, 30)),
    ),
    (
        scaler.NOPScaler(),
        mx.nd.random.normal(shape=(10, 20, 30)),
        mx.nd.ones(shape=(10, 20, 30)),
        mx.nd.ones(shape=(10, 30)),
    ),
    (
        scaler.NOPScaler(),
        mx.nd.random.normal(shape=(10, 20, 30)),
        mx.nd.zeros(shape=(10, 20, 30)),
        mx.nd.ones(shape=(10, 30)),
    ),
]


@pytest.mark.parametrize("s, target, observed, expected_scale", test_cases)
def test_scaler(s, target, observed, expected_scale):
    target_scaled, scale = s(target, observed)

    assert np.allclose(
        expected_scale.asnumpy(), scale.asnumpy()
    ), "mismatch in the scale computation"

    if s.keepdims:
        expected_target_scaled = mx.nd.broadcast_div(target, expected_scale)
    else:
        expected_target_scaled = mx.nd.broadcast_div(
            target, expected_scale.expand_dims(axis=1)
        )

    assert np.allclose(
        expected_target_scaled.asnumpy(), target_scaled.asnumpy()
    ), "mismatch in the scaled target computation"


@pytest.mark.parametrize("target, observed", [])
def test_nopscaler(target, observed):
    s = scaler.NOPScaler()
    target_scaled, scale = s(target, observed)

    assert mx.nd.norm(target - target_scaled) == 0
    assert mx.nd.norm(mx.nd.ones_like(target).mean(axis=1) - scale) == 0
