from typing import Optional
from pydantic.decorator import Callable
from gluonts.evaluation.metrics import (
    abs_target_mean,
    abs_target_sum,
    mase,
    mape,
    quantile_loss,
    smape,
    mse,
    abs_error,
    coverage,
)
import numpy as np
import pytest


ZEROES = np.array([0.0] * 5)
LINEAR = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
EXPONENTIAL = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
CONSTANT = np.array([0.4] * 5)


# TODO remove `"exclude_zero_denominator": True` when metrics are refactored
@pytest.mark.parametrize(
    "target, forecast, metrics",
    [
        (
            ZEROES,
            ZEROES,
            [
                (
                    mase,
                    np.nan,
                    {"seasonal_error": 0.0, "exclude_zero_denominator": False},
                ),
                (
                    mase,
                    0.0,
                    {"seasonal_error": 1.0, "exclude_zero_denominator": False},
                ),
                (mse, 0.0, {}),
                (abs_error, 0.0, {}),
                (quantile_loss, 0.0, {"q": 0.5}),
                (coverage, 0.0, {}),
                (mape, np.nan, {"exclude_zero_denominator": False}),
                (smape, np.nan, {"exclude_zero_denominator": False}),
            ],
        ),
        (
            LINEAR,
            ZEROES,
            [
                (
                    mase,
                    0.2 / 10e-10,
                    {
                        "seasonal_error": 10e-10,
                        "exclude_zero_denominator": False,
                    },
                ),
                (
                    mase,
                    np.inf,
                    {"seasonal_error": 0.0, "exclude_zero_denominator": False},
                ),
                (mse, 0.06, {}),
                (abs_error, 1.0, {}),
                (quantile_loss, 0.2, {"q": 0.1}),
                (quantile_loss, 1.0, {"q": 0.5}),
                (quantile_loss, 1.8, {"q": 0.9}),
                (coverage, 0.0, {}),
                (mape, np.nan, {"exclude_zero_denominator": False}),
                (smape, np.nan, {"exclude_zero_denominator": False}),
            ],
        ),
        (
            LINEAR,
            CONSTANT,
            [
                (
                    mase,
                    0.2,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.06, {}),
                (abs_error, 1.0, {}),
                (quantile_loss, 1.8, {"q": 0.1}),
                (quantile_loss, 1.0, {"q": 0.5}),
                (quantile_loss, 0.2, {"q": 0.9}),
                (coverage, 0.8, {}),
                (mape, np.inf, {"exclude_zero_denominator": False}),
                (
                    smape,
                    0.8304761904761906,
                    {"exclude_zero_denominator": False},
                ),
            ],
        ),
        (
            ZEROES,
            EXPONENTIAL,
            [
                (
                    mase,
                    0.022222,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.00202020202, {}),
                (abs_error, 0.11111, {}),
                (quantile_loss, 0.199998, {"q": 0.1}),
                (quantile_loss, 0.11111, {"q": 0.5}),
                (quantile_loss, 0.0222219, {"q": 0.9}),
                (coverage, 1.0, {}),
                (mape, np.nan, {}),
                (mape, np.inf, {"exclude_zero_denominator": False}),
                (smape, 2.0, {}),
                (smape, 2.0, {"exclude_zero_denominator": False}),
            ],
        ),
        (
            np.array([1.0, 2.0, 3.0, 0.0, 0.0]),
            np.array([2.0, 3.0, 4.0, 1.0, 0.0]),
            [
                (
                    mase,
                    0.8,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.8, {}),
                (abs_error, 4.0, {}),
                (quantile_loss, 7.2, {"q": 0.1}),
                (quantile_loss, 4.0, {"q": 0.5}),
                (quantile_loss, 0.7999999, {"q": 0.9}),
                (coverage, 0.8, {}),
                (mape, 0.6111111, {}),
                (mape, np.nan, {"exclude_zero_denominator": False}),
                (smape, 0.8380952, {}),
                (smape, np.nan, {"exclude_zero_denominator": False}),
            ],
        ),
        (
            CONSTANT,
            EXPONENTIAL,
            [
                (
                    mase,
                    0.377778,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.14424260202, {}),
                (abs_error, 1.88889, {}),
                (quantile_loss, 0.377778, {"q": 0.1}),
                (quantile_loss, 1.88889, {"q": 0.5}),
                (quantile_loss, 3.400002, {"q": 0.9}),
                (coverage, 0.0, {}),
                (mape, 0.944445, {"exclude_zero_denominator": False}),
                (smape, 1.8182728, {"exclude_zero_denominator": False}),
            ],
        ),
    ],
)
def test_metrics(target, forecast, metrics):
    for metric, expected, kwargs in metrics:
        np.testing.assert_almost_equal(
            metric(target, forecast, **kwargs), expected
        )


@pytest.mark.parametrize(
    "target, metric, expected",
    [
        (ZEROES, abs_target_sum, 0.0),
        (LINEAR, abs_target_sum, 1.0),
        (ZEROES, abs_target_mean, 0.0),
        (LINEAR, abs_target_mean, 0.2),
    ],
)
def test_target_metrics(target, metric, expected):
    np.testing.assert_almost_equal(metric(target), expected)
