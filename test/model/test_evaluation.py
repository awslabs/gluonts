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

from typing import Union, Tuple

import numpy as np
import pytest
import pandas as pd

from gluonts.dataset.split import split, TestData
from gluonts.itertools import prod
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model import (
    SampleForecast,
    evaluate_forecasts,
    evaluate_model,
)
from gluonts.ev import (
    MSE,
    RMSE,
    NRMSE,
    MAPE,
    SMAPE,
    MASE,
    MSIS,
    ND,
    MeanWeightedSumQuantileLoss,
)


_test_univariate_dataset = [
    {
        "item_id": k,
        "start": pd.Period("2022-06-12", freq="D"),
        "target": np.ones((100)),
    }
    for k in range(5)
]

_test_multivariate_dataset = [
    {
        "item_id": k,
        "start": pd.Period("2022-06-12", freq="D"),
        "target": np.ones((7, 100)),
    }
    for k in range(5)
]

_test_metrics = [
    MSE(),
    RMSE(),
    NRMSE(),
    MAPE(),
    SMAPE(),
    MASE(),
    MSIS(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.5, 0.9]),
]


def infer_metrics_df_shape(
    metrics,
    test_data: TestData,
    axis: Union[int, Tuple[int]],
) -> Tuple[int]:
    """
    Infer expected shape of metrics data frame.
    """
    labels = list(test_data.label)
    test_data_shape = (len(labels), *labels[0]["target"].shape)

    if axis is None:
        axis = tuple(range(len(test_data_shape)))
    if isinstance(axis, int):
        axis = (axis,)

    return (
        prod(
            test_data_shape[d]
            for d in range(len(test_data_shape))
            if d not in axis
        ),
        len(metrics),
    )


@pytest.mark.parametrize(
    "metrics, test_data, axis",
    [
        (
            _test_metrics,
            split(_test_univariate_dataset, offset=-12)[1].generate_instances(
                prediction_length=3, windows=4
            ),
            None,
        ),
        (
            _test_metrics,
            split(_test_univariate_dataset, offset=-12)[1].generate_instances(
                prediction_length=3, windows=4
            ),
            (0, 1),
        ),
        (
            _test_metrics,
            split(_test_univariate_dataset, offset=-12)[1].generate_instances(
                prediction_length=3, windows=4
            ),
            0,
        ),
        (
            _test_metrics,
            split(_test_univariate_dataset, offset=-12)[1].generate_instances(
                prediction_length=3, windows=4
            ),
            1,
        ),
        (
            _test_metrics,
            split(_test_multivariate_dataset, offset=-12)[
                1
            ].generate_instances(prediction_length=3, windows=4),
            None,
        ),
        (
            _test_metrics,
            split(_test_multivariate_dataset, offset=-12)[
                1
            ].generate_instances(prediction_length=3, windows=4),
            (0, 1, 2),
        ),
        (
            _test_metrics,
            split(_test_multivariate_dataset, offset=-12)[
                1
            ].generate_instances(prediction_length=3, windows=4),
            (0, 2),
        ),
        (
            _test_metrics,
            split(_test_multivariate_dataset, offset=-12)[
                1
            ].generate_instances(prediction_length=3, windows=4),
            (0, 1),
        ),
        (
            _test_metrics,
            split(_test_multivariate_dataset, offset=-12)[
                1
            ].generate_instances(prediction_length=3, windows=4),
            (1, 2),
        ),
    ],
)
def test_evaluation_shape(
    metrics,
    test_data: TestData,
    axis: Union[int, Tuple[int]],
):
    num_samples = 31

    sample_forecasts = [
        SampleForecast(
            samples=np.zeros((num_samples, *label["target"].T.shape)),
            start_date=label["start"],
            item_id=label["item_id"],
        )
        for label in test_data.label
    ]

    metrics_df = evaluate_forecasts(
        sample_forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
    )

    assert metrics_df.shape == infer_metrics_df_shape(metrics, test_data, axis)


def test_evaluate_model_vs_forecasts():
    dataset = [
        {
            "item_id": k,
            "start": pd.Period("2022-06-12", freq="D"),
            "target": np.random.normal(size=50),
        }
        for k in range(2)
    ]

    test_data = split(dataset, offset=-12)[1].generate_instances(
        prediction_length=3, windows=4
    )

    model = SeasonalNaivePredictor(
        freq="D", prediction_length=3, season_length=1
    )

    forecasts = list(model.predict(test_data.input))

    for axis in [None, 0, 1, (0, 1)]:
        df1 = evaluate_forecasts(
            forecasts=forecasts,
            test_data=test_data,
            metrics=_test_metrics,
            axis=axis,
        )
        df2 = evaluate_model(
            model=model, test_data=test_data, metrics=_test_metrics, axis=axis
        )
        assert (df1 == df2).all().all()


def test_data_nan():
    target = np.random.normal(size=50)
    target[0] = np.nan
    target[-1] = np.nan
    dataset = [
        {
            "item_id": k,
            "start": pd.Period("2022-06-12", freq="D"),
            "target": target,
        }
        for k in range(2)
    ]

    test_data = split(dataset, offset=-12)[1].generate_instances(
        prediction_length=3, windows=4
    )

    model = SeasonalNaivePredictor(
        freq="D", prediction_length=3, season_length=1
    )

    forecasts = list(model.predict(test_data.input))

    for axis in [None, 0, 1, (0, 1)]:
        df1 = evaluate_forecasts(
            forecasts=forecasts,
            test_data=test_data,
            metrics=_test_metrics,
            axis=axis,
        )
        assert not np.any(df1.isna())
